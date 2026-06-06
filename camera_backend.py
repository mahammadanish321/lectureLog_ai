import cv2
import time
import threading
import subprocess
from fastapi import FastAPI, Response, Query
from fastapi.responses import StreamingResponse
import uvicorn
import sys

app = FastAPI()

# Global state for cameras
# { cam_index: { "cap": VideoCapture, "lock": Lock, "last_frame": frame } }
cameras = {}
_cameras_lock = threading.Lock()

def find_camera_index_by_label(target_label):
    """
    Finds the hardware index by matching the friendly name.
    Ensures that 'Camera A' always maps to the same device even if indices shift.
    """
    if not target_label:
        return None

    print(f"[Camera-Backend] 🔍 Searching hardware for: {target_label}")

    try:
        # Get list of devices in the order the OS assigns them
        cmd = ["powershell", "-Command", "Get-PnpDevice -Class Image,Camera,Video -Status OK | Select-Object FriendlyName"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip() and '---' not in line and 'FriendlyName' not in line]

            # Map unique names to their OS index
            unique_devices = []
            for name in lines:
                if name not in unique_devices: unique_devices.append(name)

            for idx, name in enumerate(unique_devices):
                # Check for exact or close match
                if target_label.lower() in name.lower() or name.lower() in target_label.lower():
                    print(f"[Camera-Backend] 🎯 Resolved '{target_label}' to hardware index {idx}")
                    return idx
    except Exception as e:
        print(f"[Camera-Backend] ⚠️ Label lookup failed: {e}")

    return None

def get_camera(idx):
    with _cameras_lock:
        if idx not in cameras:
            print(f"[Camera-Backend] 📸 Attempting to open hardware camera {idx}...")
            cap = cv2.VideoCapture(idx)
            if str(idx).startswith(("http://", "https://", "rtsp://")):
                cap = cv2.VideoCapture(idx, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"[Camera-Backend] ❌ FAILED to open hardware camera {idx}")
                print(f"[Camera-Backend] ❌ FAILED to open camera/stream {idx}")
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

        time.sleep(0.03)

        # Recovery logic
        if cam and not cam["running"]:
            print(f"[Camera-Backend] 🔄 Attempting to recover Camera {idx}...")
            with _cameras_lock:
                if idx in cameras: del cameras[idx]
            time.sleep(2)

@app.get("/video_feed/{idx}")
async def video_feed(idx: str, label: str = Query(None)):
    """
    Stream video. 
    1. If a 'label' is provided, we use it to FIND the correct hardware index.
    2. If no label or label not found, we use 'idx' as fallback (could be RTSP URL or Index).
    """
    actual_idx = idx

    # Priority 1: Use label to find hardware index (handles Windows index shifting)
    if label and label.strip():
        matched_idx = find_camera_index_by_label(label)
        if matched_idx is not None:
            actual_idx = matched_idx
        elif idx.isdigit():
            actual_idx = int(idx)
    # Priority 2: Use provided idx directly
    elif idx.isdigit():
        actual_idx = int(idx)

    print(f"[Camera-Backend] 📡 Stream request: Source={actual_idx} (Requested ID={idx}, Label={label})")
    return StreamingResponse(generate_frames(actual_idx), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/health")
async def health():
    return {"status": "Camera Backend Online", "cameras_active": list(cameras.keys())}

@app.get("/list_cameras")
async def list_cameras():
    """Returns a list of all detected hardware cameras with their current indices."""
    try:
        cmd = ["powershell", "-Command", "Get-PnpDevice -Class Image,Camera,Video -Status OK | Select-Object FriendlyName"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip() and '---' not in line and 'FriendlyName' not in line]
            unique_names = []
            for name in lines:
                if name not in unique_names: unique_names.append(name)

            # Return as objects for easier mapping
            return {"cameras": [{"index": i, "name": name} for i, name in enumerate(unique_names)]}
    except Exception as e:
        return {"error": str(e)}
    return {"cameras": []}

@app.post("/release/{idx}")
async def release_camera(idx: str):
    """Explicitly release a hardware camera or stream."""
    actual_idx = int(idx) if idx.isdigit() else idx
    with _cameras_lock:
        if actual_idx in cameras:
            cam = cameras[actual_idx]
            print(f"[Camera-Backend] 🛑 Explicit release requested for Source {actual_idx}...")
            cam["running"] = False
            # Wait a moment for threads to notice running=False
            time.sleep(0.1)
            with cam["lock"]:
                if cam["cap"]:
                    cam["cap"].release()
            del cameras[idx]
            return {"status": f"Camera {idx} released"}
    return {"status": "Camera not active"}

def cleanup_idle_cameras():
    """Background task to release cameras that haven't been accessed for 5s."""
    while True:
        time.sleep(2)
        now = time.time()
        to_delete = []

        with _cameras_lock:
            for idx, cam in cameras.items():
                if now - cam["last_access"] > 5:
                    print(f"[Camera-Backend] 💤 Camera {idx} is idle. Releasing hardware...")
                    cam["running"] = False
                    with cam["lock"]:
                        if cam["cap"]:
                            cam["cap"].release()
                    to_delete.append(idx)

            for idx in to_delete:
                del cameras[idx]

if __name__ == "__main__":
    threading.Thread(target=cleanup_idle_cameras, daemon=True).start()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)