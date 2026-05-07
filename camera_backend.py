import cv2
import time
import threading
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import sys

app = FastAPI()

# Global state for cameras
# { cam_index: { "cap": VideoCapture, "lock": Lock, "last_frame": frame } }
cameras = {}
_cameras_lock = threading.Lock()

def get_camera(idx):
    with _cameras_lock:
        if idx not in cameras:
            print(f"[Camera-Backend] 📸 Attempting to open hardware camera {idx}...")
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"[Camera-Backend] ❌ FAILED to open hardware camera {idx}")
                return None
            
            print(f"[Camera-Backend] ✅ Camera {idx} hardware opened successfully")
            # Set lower resolution for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            cameras[idx] = {
                "cap": cap,
                "lock": threading.Lock(),
                "last_frame": None,
                "running": True,
                "last_access": time.time()
            }
            
            # Start background capture thread
            thread = threading.Thread(target=capture_loop, args=(idx,), daemon=True)
            thread.start()
            
        return cameras[idx]

def capture_loop(idx):
    cam = cameras[idx]
    while cam["running"]:
        with cam["lock"]:
            ret, frame = cam["cap"].read()
            if ret:
                cam["last_frame"] = frame
            else:
                print(f"Error reading from camera {idx}")
                cam["running"] = False
        time.sleep(0.03) # ~30 FPS

def generate_frames(idx):
    boundary = "frame"
    while True:
        cam = get_camera(idx)
        if not cam or not cam["running"]:
            time.sleep(1)
            continue
            
        with cam["lock"]:
            cam["last_access"] = time.time() # Mark as active
            frame = cam["last_frame"]
            
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03) # Match ~30 FPS capture rate
        
        # If camera died, try to recover it after a delay
        if cam and not cam["running"]:
            print(f"[Camera-Backend] 🔄 Attempting to recover Camera {idx}...")
            with _cameras_lock:
                del cameras[idx]
            time.sleep(2)

@app.get("/video_feed/{idx}")
async def video_feed(idx: int):
    print(f"[Camera-Backend] 📡 New stream request for Camera {idx}")
    return StreamingResponse(generate_frames(idx), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/health")
async def health():
    print(f"[Camera-Backend] 💓 Health check received")
    return {"status": "Camera Backend Online", "cameras_open": list(cameras.keys())}

def cleanup_idle_cameras():
    """Background task to release cameras that haven't been accessed for 30s."""
    while True:
        time.sleep(10)
        now = time.time()
        to_delete = []
        
        with _cameras_lock:
            for idx, cam in cameras.items():
                if now - cam["last_access"] > 30:
                    print(f"[Camera-Backend] 💤 Camera {idx} is idle. Releasing hardware...")
                    cam["running"] = False
                    with cam["lock"]:
                        if cam["cap"]:
                            cam["cap"].release()
                    to_delete.append(idx)
            
            for idx in to_delete:
                del cameras[idx]

if __name__ == "__main__":
    # Start cleanup thread
    threading.Thread(target=cleanup_idle_cameras, daemon=True).start()
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
