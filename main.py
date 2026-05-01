from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import cv2
import numpy as np
from deepface import DeepFace
import requests
import os
import time
from dotenv import load_dotenv
import threading
import uvicorn
from scipy.spatial.distance import cosine
import shutil
import json
from datetime import datetime

load_dotenv()

# ── Configuration ──────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api/recognition")
STUDENTS_API = os.getenv("STUDENTS_API", "http://localhost:5000/api/students")
CLASSROOMS_API = os.getenv("CLASSROOMS_API", "http://localhost:5000/api/classrooms")
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "30"))
AI_PORT = int(os.getenv("AI_PORT", "8001"))
RECOGNITION_INTERVAL = float(os.getenv("RECOGNITION_INTERVAL", "3.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
FRAME_SCALE = float(os.getenv("FRAME_SCALE", "0.4"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "60"))
CAMERA_IDLE_TIMEOUT = int(os.getenv("CAMERA_IDLE_TIMEOUT", "60"))

# ── Logging Helpers ────────────────────────────────────────
def log(icon, tag, msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"info": "\033[0m", "success": "\033[92m", "warn": "\033[93m", "error": "\033[91m", "dim": "\033[90m"}
    c = colors.get(level, "\033[0m")
    print(f"{c}[{ts}] {icon} [{tag}] {msg}\033[0m")

# ── Global State ───────────────────────────────────────────
system_active = False          # Start INACTIVE — no camera access until needed
student_cache = []
last_marked = {}               # {student_id: timestamp}
current_session_info = None    # List of active sessions or None
camera_workers = {}            # {cam_index: CameraWorker}
_state_lock = threading.Lock()

# ── Helper Functions ───────────────────────────────────────
def get_camera_index(cam_info):
    """Convert a camera_url string to an integer index."""
    cam_info = str(cam_info).strip()
    if cam_info.isdigit():
        return int(cam_info)
    if "Smart Connect" in cam_info or "Phone" in cam_info:
        return 1
    return 0

def refresh_student_cache():
    """Fetch active sessions and matching students from backend."""
    global student_cache, current_session_info, system_active

    try:
        log("🔄", "SYNC", "Refreshing system context...")

        # ── Step 1: Fetch active sessions ──
        active_sessions = []
        try:
            sess_resp = requests.get("http://localhost:5000/api/sessions", timeout=5)
            if sess_resp.status_code == 200:
                sessions = sess_resp.json()
                active_sessions = [s for s in sessions if s.get('status') == 'active']
        except Exception as e:
            log("❌", "SYNC", f"Cannot reach backend for sessions: {e}", "error")
            return

        # ── Step 2: Update session info ──
        if active_sessions:
            log("📡", "SYNC", f"Found {len(active_sessions)} active session(s):", "success")
            for s in active_sessions:
                log("  ", "SYNC", f"  → {s.get('subject_name', '?')} | Year {s.get('year', '?')} {s.get('stream', '')} | Room: {s.get('classroom_name', '?')} | Cam: {s.get('camera_url', '?')}", "dim")
            current_session_info = active_sessions
            system_active = True
        else:
            log("💤", "SYNC", "No active sessions. System is idle.", "dim")
            current_session_info = None
            system_active = False
            student_cache = []
            # Stop idle cameras
            _cleanup_idle_cameras()
            return

        # ── Step 3: Fetch students ──
        try:
            response = requests.get(STUDENTS_API, timeout=5)
            if response.status_code != 200:
                log("❌", "SYNC", f"Students API returned {response.status_code}", "error")
                return
        except Exception as e:
            log("❌", "SYNC", f"Cannot reach students API: {e}", "error")
            return

        all_students = response.json()
        all_valid = []
        skipped = 0
        for s in all_students:
            if s.get('face_embedding'):
                if isinstance(s['face_embedding'], str):
                    try:
                        s['face_embedding'] = json.loads(s['face_embedding'])
                    except:
                        skipped += 1
                        continue
                all_valid.append(s)
            else:
                skipped += 1

        if skipped > 0:
            log("⚠️", "SYNC", f"{skipped} student(s) skipped (no face embedding)", "warn")

        # ── Step 4: Filter students by active session groups ──
        new_cache = []
        for sess in active_sessions:
            y, st = sess.get('year'), sess.get('stream')
            if y and st:
                matches = [s for s in all_valid if str(s.get('year')) == str(y) and str(s.get('stream', '')).lower() == str(st).lower()]
                new_cache.extend(matches)
            else:
                new_cache.extend(all_valid)

        # Deduplicate
        seen_ids = set()
        student_cache = []
        for s in new_cache:
            if s['id'] not in seen_ids:
                student_cache.append(s)
                seen_ids.add(s['id'])

        log("👤", "SYNC", f"Cache updated: {len(student_cache)} students ready for recognition", "success")

    except Exception as e:
        log("❌", "SYNC", f"Critical refresh error: {e}", "error")

def _cleanup_idle_cameras():
    """Stop camera workers that are no longer needed."""
    with _state_lock:
        for idx in list(camera_workers.keys()):
            worker = camera_workers[idx]
            worker.stop()
            del camera_workers[idx]
            log("📷", "CLEANUP", f"Camera {idx} released (no active sessions)", "dim")

def _ensure_camera(cam_index):
    """Start a camera worker if not already running."""
    with _state_lock:
        if cam_index not in camera_workers:
            log("📸", "CAMERA", f"Opening camera {cam_index}...", "info")
            camera_workers[cam_index] = CameraWorker(cam_index)
        return camera_workers[cam_index]

# ── Camera Worker (Optimized) ──────────────────────────────
class CameraWorker:
    def __init__(self, index):
        self.index = index
        self.cap = None
        self.latest_frame = np.zeros((360, 480, 3), dtype=np.uint8)  # Smaller default
        self._raw_frame = None      # Raw frame for recognition (no overlays)
        self.running = True
        self.last_recognition_results = []  # [{name, confidence, area}]
        self._recognition_lock = threading.Lock()
        self._recognition_busy = False
        self._last_active_time = time.time()

        threading.Thread(target=self._capture_loop, daemon=True, name=f"cam-{index}-capture").start()
        threading.Thread(target=self._recognition_loop, daemon=True, name=f"cam-{index}-recog").start()

    def stop(self):
        """Gracefully stop the worker and release the camera."""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            log("📷", "CAMERA", f"Camera {self.index} hardware released", "dim")

    def _capture_loop(self):
        log("📸", "CAMERA", f"Capture thread started for Camera {self.index}")
        self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            log("❌", "CAMERA", f"Cannot open Camera {self.index}! Is it connected?", "error")
            self.running = False
            return

        # Set lower resolution to reduce memory
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        fail_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 20:
                    log("❌", "CAMERA", f"Camera {self.index} disconnected after 20 failures", "error")
                    self.running = False
                    break
                time.sleep(0.5)
                continue

            fail_count = 0
            self._raw_frame = frame  # Store raw for recognition

            # Build display frame with overlays
            display = frame.copy()

            # Draw recognition results on display frame
            for result in self.last_recognition_results:
                area = result.get('area', {})
                x, y, w, h = area.get('x', 0), area.get('y', 0), area.get('w', 0), area.get('h', 0)
                name = result.get('name', '')
                confidence = result.get('confidence', 0)

                if name == "UNKNOWN":
                    color = (0, 0, 200)  # Red
                    label = "UNKNOWN"
                elif name == "ANALYZING":
                    color = (0, 200, 200)  # Yellow
                    label = "ANALYZING..."
                else:
                    color = (0, 200, 0)  # Green
                    label = f"{name.upper()} ({confidence:.0%})"

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Status bar
            status_text = "MONITORING" if system_active else "IDLE (No Session)"
            status_color = (0, 200, 0) if system_active else (128, 128, 128)
            cv2.putText(display, f"CAM {self.index}: {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if system_active and student_cache:
                cv2.putText(display, f"Students: {len(student_cache)} | Marked: {len(last_marked)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            self.latest_frame = display
            time.sleep(0.066)  # ~15 FPS display

    def _recognition_loop(self):
        log("🧠", "RECOG", f"Recognition thread started for Camera {self.index}")

        while self.running:
            if not system_active or not student_cache or self._raw_frame is None:
                time.sleep(1)
                continue

            if self._recognition_busy:
                time.sleep(0.5)
                continue

            self._recognition_busy = True
            scan_start = time.time()

            try:
                # Resize for faster processing
                small = cv2.resize(self._raw_frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

                # Run DeepFace
                objs = DeepFace.represent(
                    img_path=small,
                    model_name="VGG-Face",
                    enforce_detection=False,
                    detector_backend="opencv"
                )

                if not objs:
                    self.last_recognition_results = []
                    log("👁️", f"CAM-{self.index}", "No faces detected in frame", "dim")
                    continue

                results = []
                scale_inv = 1.0 / FRAME_SCALE

                for obj in objs:
                    fa = obj.get("facial_area", {})
                    # Scale facial area back to original frame coords
                    area = {
                        'x': int(fa.get('x', 0) * scale_inv),
                        'y': int(fa.get('y', 0) * scale_inv),
                        'w': int(fa.get('w', 0) * scale_inv),
                        'h': int(fa.get('h', 0) * scale_inv)
                    }

                    # Skip tiny faces (likely false positives)
                    if area['w'] < 30 or area['h'] < 30:
                        continue

                    embedding = obj["embedding"]

                    # Find best match
                    best_match = None
                    min_dist = 1.0
                    for s in student_cache:
                        d = cosine(embedding, s['face_embedding'])
                        if d < min_dist:
                            min_dist = d
                            best_match = s

                    confidence = 1.0 - min_dist

                    if best_match and min_dist < CONFIDENCE_THRESHOLD:
                        student_id = best_match['id']
                        student_name = best_match['name']
                        current_time = time.time()

                        results.append({'name': student_name, 'confidence': confidence, 'area': area})

                        # Check cooldown before marking
                        if student_id not in last_marked or (current_time - last_marked[student_id]) > COOLDOWN_PERIOD:
                            log("✅", f"CAM-{self.index}", f"MATCH: {student_name.upper()} (confidence: {confidence:.1%})", "success")

                            try:
                                requests.post(BACKEND_URL, json={
                                    "student_id": student_id,
                                    "session_id": "active",
                                    "confidence": confidence
                                }, timeout=3)
                                last_marked[student_id] = current_time
                                log("📝", f"CAM-{self.index}", f"  → Attendance marked for {student_name}", "success")
                            except Exception as e:
                                log("❌", f"CAM-{self.index}", f"  → Failed to mark attendance: {e}", "error")
                    elif best_match and min_dist < 0.65:
                        # Low confidence — face detected but not sure
                        results.append({'name': "ANALYZING", 'confidence': confidence, 'area': area})
                        log("🔍", f"CAM-{self.index}", f"Low confidence face: closest to {best_match['name']} ({confidence:.1%}) — below threshold", "warn")
                    else:
                        results.append({'name': "UNKNOWN", 'confidence': 0, 'area': area})

                self.last_recognition_results = results

                elapsed = time.time() - scan_start
                log("⏱️", f"CAM-{self.index}", f"Scan complete: {len(objs)} face(s) processed in {elapsed:.1f}s", "dim")

            except Exception as e:
                log("❌", f"CAM-{self.index}", f"Recognition error: {e}", "error")
            finally:
                self._recognition_busy = False

            time.sleep(RECOGNITION_INTERVAL)

# ── Background Maintenance ─────────────────────────────────
def run_maintenance():
    """Periodically refresh student cache and manage camera workers."""
    log("🚀", "SYSTEM", "AI Service maintenance loop started")

    while True:
        refresh_student_cache()

        # Start camera workers for active sessions
        if system_active and current_session_info:
            needed = set()
            for s in current_session_info:
                idx = get_camera_index(s.get('camera_url', '0'))
                needed.add(idx)

            for idx in needed:
                _ensure_camera(idx)

            # Stop cameras no longer needed
            with _state_lock:
                for idx in list(camera_workers.keys()):
                    if idx not in needed:
                        camera_workers[idx].stop()
                        del camera_workers[idx]
                        log("📷", "CLEANUP", f"Camera {idx} stopped (session ended)", "dim")

        time.sleep(30)

# ── FastAPI ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    log("🚀", "SYSTEM", "LectureLog AI Service starting...")
    log("📋", "SYSTEM", f"Config: threshold={CONFIDENCE_THRESHOLD}, cooldown={COOLDOWN_PERIOD}s, scale={FRAME_SCALE}, interval={RECOGNITION_INTERVAL}s")
    threading.Thread(target=run_maintenance, daemon=True, name="maintenance").start()
    yield
    log("🛑", "SYSTEM", "AI Service shutting down...")
    for w in camera_workers.values():
        w.stop()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── API Endpoints ──────────────────────────────────────────

@app.get("/system/status")
async def get_status():
    return {
        "active": system_active,
        "cameras_open": list(camera_workers.keys()),
        "students_cached": len(student_cache),
        "students_marked": len(last_marked),
        "active_sessions": len(current_session_info) if current_session_info else 0
    }

@app.post("/system/toggle")
async def toggle_system():
    global system_active
    system_active = not system_active
    if not system_active:
        _cleanup_idle_cameras()
    log("🔄", "SYSTEM", f"System toggled: {'ACTIVE' if system_active else 'PAUSED'}", "info")
    return {"active": system_active}

@app.post("/system/refresh")
async def refresh_system():
    refresh_student_cache()
    return {"message": "AI Cache Refreshed"}

@app.get("/cameras")
async def list_cameras():
    """Return all registered classroom cameras from the backend."""
    try:
        resp = requests.get(CLASSROOMS_API, timeout=3)
        if resp.status_code == 200:
            classrooms = resp.json()
            cameras = []
            for c in classrooms:
                cam_url = c.get('camera_url', '0')
                cameras.append({
                    "classroom_id": c['id'],
                    "classroom_name": c['name'],
                    "camera_index": get_camera_index(cam_url),
                    "camera_url": cam_url
                })
            return cameras
        return []
    except:
        return []

@app.get("/video_feed")
async def video_feed(v: str = "default", cam: str = None):
    """
    Stream video from a camera.
    - ?v=<session_id>  → Show camera linked to that session
    - ?cam=<index>     → Show a specific camera by index (idle browsing mode)
    - default          → Show first available camera
    """
    target_idx = 0

    if cam is not None:
        # Direct camera index mode (idle browsing)
        try:
            target_idx = int(cam)
        except:
            target_idx = 0
    elif v != "default" and current_session_info:
        # Session-based camera selection
        target_sess = next((s for s in current_session_info if str(s.get('id')) == v), None)
        if target_sess:
            target_idx = get_camera_index(target_sess.get('camera_url', '0'))

    # Ensure camera is open
    _ensure_camera(target_idx)

    def frame_generator():
        while True:
            worker = camera_workers.get(target_idx)
            if worker and worker.running:
                ret, buffer = cv2.imencode('.jpg', worker.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # Send a blank frame with error message
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(blank, f"Camera {target_idx} unavailable", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.066)  # ~15 FPS

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    """Generate a face embedding from an uploaded image."""
    temp_path = f"temp_{int(time.time())}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            objs = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=True)
        except:
            objs = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=False)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if objs:
            return {"embedding": objs[0]["embedding"]}
        return {"error": "Could not find a valid face signature."}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"error": str(e)}

# ── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=AI_PORT)
