import os
# Suppress TensorFlow logs before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import cv2
import numpy as np
import requests
import time
from dotenv import load_dotenv
import threading
import uvicorn
import logging
import signal
import sys
from scipy.spatial.distance import cosine
import shutil
import json
from datetime import datetime
import sys

# Lazy loading for DeepFace
DeepFace = None

load_dotenv()

# ── Configuration ──────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api/recognition")
STUDENTS_API = os.getenv("STUDENTS_API", "http://localhost:5000/api/students")
SESSIONS_API = os.getenv("SESSIONS_API", "http://localhost:5000/api/sessions")
CLASSROOMS_API = os.getenv("CLASSROOMS_API", "http://localhost:5000/api/classrooms")
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "30"))
AI_PORT = int(os.getenv("AI_PORT", "8001"))
STREAM_BACKEND_URL = os.getenv("STREAM_BACKEND_URL", "http://localhost:8002/video_feed")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID") # Used for multi-tenancy isolation
RECOGNITION_INTERVAL = float(os.getenv("RECOGNITION_INTERVAL", "3.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
FRAME_SCALE = float(os.getenv("FRAME_SCALE", "0.4"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "60"))
CAMERA_IDLE_TIMEOUT = int(os.getenv("CAMERA_IDLE_TIMEOUT", "60"))
# CAMERA_BACKEND_URL removed - now using direct hardware access

# ── Logging Helpers ────────────────────────────────────────
def log(icon, tag, msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {"info": "\033[0m", "success": "\033[92m", "warn": "\033[93m", "error": "\033[91m", "dim": "\033[90m"}
    c = colors.get(level, "\033[0m")
    print(f"{c}[{ts}] {icon} [{tag}] {msg}\033[0m")

log("🖥️", "SYSTEM", f"Python Executable: {sys.executable}")
log("🆔", "SYSTEM", f"Python Version: {sys.version}")
log("📂", "SYSTEM", f"Working Directory: {os.getcwd()}")
log("⚙️", "SYSTEM", f"Platform: {sys.platform}")

# ── Global State ───────────────────────────────────────────
system_active = False          # Global system active flag
student_cache = []
last_marked = {}               # {student_id: timestamp}
current_session_info = None    # List of active sessions or None
camera_workers = {}            # {cam_index: CameraWorker}
global_error = None            # Track global service errors
scanner_enabled = True         # Toggle to enable/disable scanning globally
_state_lock = threading.Lock()

# ── Helper Functions ───────────────────────────────────────
def find_camera_index_by_label(target_label):
    """Tries to find the hardware index by poking devices directly using PowerShell."""
    if not target_label: return None
    log("🔍", "CAMERA", f"Searching hardware for label: {target_label}")
    try:
        import subprocess
        cmd = ["powershell", "-Command", "Get-PnpDevice -Class Image,Camera,Video -Status OK | Select-Object FriendlyName"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip() and '---' not in line and 'FriendlyName' not in line]
            unique_names = []
            for name in lines:
                if name not in unique_names: unique_names.append(name)
            for idx, name in enumerate(unique_names):
                if target_label.lower() in name.lower() or name.lower() in target_label.lower():
                    log("🎯", "CAMERA", f"Matched label '{target_label}' to hardware index {idx}")
                    return idx
    except Exception as e:
        log("⚠️", "CAMERA", f"Label lookup failed: {e}", "warn")
    return None

def get_camera_url(cam_info, label=None):
    """Convert a camera_url string or index to a full stream URL or hardware index."""
    cam_info = str(cam_info).strip()
    
    # If it's already a full URL, use it
    if cam_info.startswith("http://") or cam_info.startswith("https://") or cam_info.startswith("rtsp://"):
        return cam_info
        
    # If a label is provided, try to find the hardware index by label first for accuracy
    if label and len(str(label)) > 3:
        matched_idx = find_camera_index_by_label(label)
        if matched_idx is not None:
            return matched_idx

    # Fallback to the digit index from DB
    if cam_info.isdigit():
        return int(cam_info)
        
    # Default to camera 0
    return 0

def refresh_student_cache():
    """Fetch active sessions and matching students from backend."""
    global student_cache, current_session_info, system_active

    try:
        log("🔄", "SYNC", "Refreshing system context...")

        # ── Step 1: Fetch active sessions ──
        active_sessions = []
        try:
            params = {"organization_id": ORGANIZATION_ID} if ORGANIZATION_ID else {}
            sess_resp = requests.get(SESSIONS_API, params=params, timeout=5)
            if sess_resp.status_code == 200:
                sessions = sess_resp.json()
                active_sessions = [s for s in sessions if s.get('status') == 'active']
                global_error = None
        except Exception as e:
            log("❌", "SYNC", f"Cannot reach backend for sessions: {e}", "error")
            global_error = "Backend service unreachable"
            return

        # ── Step 2: Update session info ──
        if active_sessions:
            log("📡", "SYNC", f"Found {len(active_sessions)} active session(s):", "success")
            for s in active_sessions:
                log("  ", "SYNC", f"  → {s.get('subject_name', '?')} | Year {s.get('year', '?')} {s.get('stream', '')} | Room: {s.get('classroom_name', '?')} | Cam: {s.get('camera_url', '?')}", "dim")
            current_session_info = active_sessions
            system_active = True
        else:
            if system_active:
                log("💤", "SYNC", "No active sessions. System transitioning to idle.", "dim")
            current_session_info = None
            system_active = False
            student_cache = []
            return

        # ── Step 3: Fetch students ──
        try:
            params = {"organization_id": ORGANIZATION_ID} if ORGANIZATION_ID else {}
            response = requests.get(STUDENTS_API, params=params, timeout=5)
            if response.status_code != 200:
                log("❌", "SYNC", f"Students API returned {response.status_code}", "error")
                return
        except Exception as e:
            log("❌", "SYNC", f"Cannot reach students API: {e}", "error")
            # If we can't reach the backend, we should probably stop scanning for safety
            system_active = False
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

        # ── Step 5: Deduplicate ──
        seen_ids = set()
        student_cache = []
        for s in new_cache:
            if s['id'] not in seen_ids:
                student_cache.append(s)
                seen_ids.add(s['id'])

        # ── Step 6: Start workers for active sessions ──
        if active_sessions:
            for sess in active_sessions:
                url = get_camera_url(sess.get('camera_url', '0'), sess.get('camera_name'))
                worker = _ensure_camera(url)
                if not worker.is_scanning:
                    worker.is_scanning = True
                    log("🔍", "SYSTEM", f"Auto-started scanning for Camera {url}", "success")

        log("👤", "SYNC", f"Cache updated: {len(student_cache)} students ready for recognition", "success")

    except Exception as e:
        log("❌", "SYNC", f"Critical refresh error: {e}", "error")

def _cleanup_idle_cameras(exclude=None, force=False):
    """Stop camera workers that are no longer needed.
    Skips cameras that have an active viewer (streamed within the last 15s)
    unless force=True, and cameras in the exclude set."""
    exclude = exclude or set()
    now = time.time()
    with _state_lock:
        for idx in list(camera_workers.keys()):
            if idx in exclude:
                continue
            worker = camera_workers[idx]
            # Don't kill cameras with an active viewer unless forced
            if not force and (now - worker._last_stream_time) < 15:
                log("👁️", "CLEANUP", f"Camera {idx} kept alive (active viewer)", "dim")
                continue
            
            # Derivative: If it's a backend URL, tell the backend to release it too
            try:
                if str(idx).startswith(STREAM_BACKEND_URL):
                    # Derive index from URL: .../video_feed/0 -> 0
                    parts = str(idx).split('/')
                    cam_id = parts[-1].split('?')[0]
                    if cam_id.isdigit():
                        release_url = f"{STREAM_BACKEND_URL.replace('/video_feed', '')}/release/{cam_id}"
                        log("🛑", "CLEANUP", f"Signaling Camera Backend to release {cam_id}...")
                        requests.post(release_url, timeout=1)
            except: pass

            worker.stop()
            del camera_workers[idx]
            log("📷", "CLEANUP", f"Camera {idx} released (no active sessions)", "dim")

def _ensure_camera(cam_url):
    """Start a camera worker if not already running."""
    with _state_lock:
        if cam_url not in camera_workers:
            log("📸", "CAMERA", f"Initializing AI Worker for Stream: {cam_url}...", "info")
            camera_workers[cam_url] = CameraWorker(cam_url)
        return camera_workers[cam_url]

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
        self.is_scanning = False    # Controlled by backend
        self.status = "Initializing Camera..."
        self.error_message = None
        # self.stream_url removed - using self.index directly
        self._last_active_time = time.time()
        self._last_stream_time = time.time()  # Updated by video_feed viewers

        threading.Thread(target=self._capture_loop, daemon=True, name=f"cam-{index}-capture").start()
        threading.Thread(target=self._recognition_loop, daemon=True, name=f"cam-{index}-recog").start()

    def stop(self):
        """Gracefully stop the worker and release the camera."""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            log("📷", "CAMERA", f"Camera {self.index} hardware released", "dim")

    def _capture_loop(self):
        # Determine the correct source: If it's a hardware index, route through Camera Backend to avoid locks
        source = self.index
        is_hardware = False
        
        if isinstance(self.index, (int, float)) or (isinstance(self.index, str) and self.index.isdigit()):
            is_hardware = True
            # Pre-check: Is the camera backend even alive?
            try:
                # Use a short timeout to check health
                health_resp = requests.get("http://localhost:8002/health", timeout=1)
                if health_resp.status_code == 200:
                    source = f"{STREAM_BACKEND_URL}/{self.index}"
                    log("🔗", "CAPTURE", f"Routing hardware index {self.index} through Backend Stream: {source}")
                else:
                    is_hardware = False # Fallback to direct if health check fails but returns a code
            except:
                is_hardware = False # Fallback to direct if backend is totally offline
        
        if not is_hardware:
            log("📷", "CAPTURE", f"Connecting to Video Source: {source}")

        # Open the capture object with specific backend for stability
        # For URLs (HTTP), CAP_FFMPEG is the most stable on Windows
        if str(source).startswith("http"):
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            # Desperate fallback for hardware
            if is_hardware:
                log("⚠️", "CAMERA", "Backend stream failed, attempting direct hardware access...")
                self.cap = cv2.VideoCapture(int(self.index))
            
            if not self.cap.isOpened():
                log("❌", "CAMERA", f"Video source {source} could not be opened!", "error")
                self.status = "Hardware Error"
                self.error_message = f"Source {source} Failed"
                self.running = False
                return

        log("✅", "CAPTURE", f"Video source {source} Opened Successfully")
        self.status = "Camera Online"

        # Set lower resolution to reduce memory
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        fail_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                fail_count += 1
                if fail_count % 10 == 0:
                    log("⚠️", "CAPTURE", f"Stream glitch on Cam {self.index}, retrying... ({fail_count}/50)", "warn")
                if fail_count > 50:
                    log("❌", "CAMERA", f"Stream lost for Cam {self.index}. Reconnecting...", "error")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.index)
                    fail_count = 0
                time.sleep(0.1)
                continue

            fail_count = 0
            if frame is not None:
                self._raw_frame = frame  # Store raw for recognition
            else:
                continue

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
            status_text = self.status
            if self.is_scanning:
                status_text = "AI SCANNING ACTIVE"
            
            status_color = (0, 200, 0) if self.is_scanning else (128, 128, 128)
            if "Error" in self.status or "Failed" in self.status:
                status_color = (0, 0, 200)

            cv2.putText(display, f"CAM {self.index}: {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            if self.error_message:
                cv2.putText(display, f"Error: {self.error_message}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if system_active and student_cache:
                cv2.putText(display, f"Students: {len(student_cache)} | Marked: {len(last_marked)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            self.latest_frame = display
            time.sleep(0.066)  # ~15 FPS display

    def _recognition_loop(self):
        global DeepFace
        log("🧠", "RECOG", f"Recognition thread started for Camera {self.index}")

        # Lazy load DeepFace if not already loaded
        if DeepFace is None:
            log("⏳", "AI", "Loading DeepFace & TensorFlow (First run, may take a few seconds)...", "warn")
            from deepface import DeepFace as DF
            DeepFace = DF
            log("✅", "AI", "AI Models loaded and ready.")

        while self.running:
            if not system_active or not student_cache or self._raw_frame is None or not self.is_scanning:
                time.sleep(1)
                continue

            if self._recognition_busy:
                time.sleep(0.5)
                continue

            self._recognition_busy = True
            log("🔍", f"SCAN-{self.index}", "Analyzing frame for students...", "dim")
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

                    if best_match and min_dist < (CONFIDENCE_THRESHOLD + 0.05): # Slightly more lenient
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
                    elif best_match and min_dist < 0.75: # Increased range for 'Analyzing' feedback
                        # Low confidence — face detected but not sure
                        results.append({'name': "ANALYZING", 'confidence': confidence, 'area': area})
                        log("🔍", f"CAM-{self.index}", f"Low confidence face: closest to {best_match['name']} ({confidence:.1%}) — trying to confirm...", "warn")
                    else:
                        results.append({'name': "UNKNOWN", 'confidence': 0, 'area': area})

                self.last_recognition_results = results

                elapsed = time.time() - scan_start
                log("⏱️", f"CAM-{self.index}", f"Scan complete: {len(objs)} face(s) processed in {elapsed:.1f}s", "dim")
                
                # If we reached here, recognition is working
                self.status = "AI Scanning Active"
                self.error_message = None

            except Exception as e:
                log("❌", f"CAM-{self.index}", f"Recognition error: {e}", "error")
                self.status = "Recognition Failed"
                self.error_message = str(e)
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
                url = get_camera_url(s.get('camera_url', '0'), s.get('camera_name'))
                needed.add(url)

            for url in needed:
                _ensure_camera(url)

            # Stop cameras no longer needed by sessions (but keep browsed ones)
            _cleanup_idle_cameras(exclude=needed)
        else:
            # No active sessions — clean up everything immediately
            _cleanup_idle_cameras(force=True)

        time.sleep(10)

# ── FastAPI ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global DeepFace
    log("🚀", "SYSTEM", "LectureLog AI Service starting...")
    log("📋", "SYSTEM", f"Config: threshold={CONFIDENCE_THRESHOLD}, cooldown={COOLDOWN_PERIOD}s, scale={FRAME_SCALE}, interval={RECOGNITION_INTERVAL}s")
    
    # Start a background thread to warm up DeepFace so it doesn't lag the first scan
    def warmup():
        global DeepFace
        try:
            log("🔥", "AI", "Warming up AI models in background...", "dim")
            from deepface import DeepFace as DF
            DeepFace = DF
            # Pre-load VGG-Face by doing a dummy representation
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(dummy, model_name="VGG-Face", enforce_detection=False)
            log("✨", "AI", "Background warmup complete. Scanning will be fast.", "success")
        except Exception as e:
            log("⚠️", "AI", f"Warmup failed: {e}", "warn")

    threading.Thread(target=warmup, daemon=True).start()

    threading.Thread(target=run_maintenance, daemon=True, name="Maintenance").start()
    yield
    log("🛑", "SYSTEM", "AI Service shutting down...")
    for w in camera_workers.values():
        w.stop()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── API Endpoints ──────────────────────────────────────────

@app.get("/system/status")
async def get_status():
    cam_status = {}
    for idx, worker in camera_workers.items():
        cam_status[idx] = {
            "status": worker.status,
            "error": worker.error_message,
            "is_scanning": worker.is_scanning,
            "busy": worker._recognition_busy
        }

    return {
        "active": system_active,
        "scanner_enabled": scanner_enabled,
        "global_error": global_error,
        "cameras": cam_status,
        "cameras_open": list(camera_workers.keys()),
        "students_cached": len(student_cache),
        "students_marked": len(last_marked),
        "active_sessions": len(current_session_info) if current_session_info else 0
    }

@app.get("/system/hardware_cameras")
async def get_hardware_cameras():
    """Returns all available camera devices using system-level detection."""
    try:
        import subprocess
        cmd = ["powershell", "-Command", "Get-PnpDevice -Class Image,Camera,Video -Status OK | Select-Object FriendlyName"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip() and '---' not in line and 'FriendlyName' not in line]
            unique_names = []
            for name in lines:
                if name not in unique_names: unique_names.append(name)
            
            cameras = []
            for idx, name in enumerate(unique_names):
                cameras.append({"id": str(idx), "name": name})
            return cameras
    except Exception as e:
        log("⚠️", "SYSTEM", f"Hardware detection failed: {e}", "warn")
    return []

@app.post("/system/toggle")
async def toggle_system():
    global system_active
    system_active = not system_active
    if not system_active:
        _cleanup_idle_cameras()
    log("🔄", "SYSTEM", f"System toggled: {'ACTIVE' if system_active else 'PAUSED'}", "info")
    return {"active": system_active}

@app.post("/system/refresh")
async def refresh_system(scan: bool = True):
    """
    Called by backend when a session starts OR ends.
    - scan=True: Automatically start scanning on all active cameras.
    """
    refresh_student_cache()
    
    if system_active and current_session_info:
        needed = set()
        for sess in current_session_info:
            url = get_camera_url(sess.get('camera_url', '0'), sess.get('camera_name'))
            needed.add(url)
            worker = _ensure_camera(url)
            if scan:
                worker.is_scanning = True
                log("🔍", "SYSTEM", f"Scanning started for Stream {url}", "success")
        
        # Stop any cameras that were running but are no longer in the active sessions
        _cleanup_idle_cameras(exclude=needed)
    else:
        # System became idle — force stop everything immediately
        log("💤", "SYSTEM", "No active sessions after refresh. Cleaning up cameras.", "info")
        _cleanup_idle_cameras(force=True)

    return {"message": "AI Cache Refreshed", "system_active": system_active}

@app.post("/scanner/start")
async def start_scanner(cam: int = 0):
    worker = _ensure_camera(cam)
    worker.is_scanning = True
    return {"status": "Scanning started", "camera": cam}

@app.post("/scanner/stop")
async def stop_scanner(cam: int = 0):
    if cam in camera_workers:
        camera_workers[cam].is_scanning = False
        return {"status": "Scanning stopped", "camera": cam}
    return {"error": "Camera not open"}

@app.get("/cameras")
async def list_cameras():
    """Return all registered classroom cameras from the backend."""
    try:
        resp = requests.get(CLASSROOMS_API, timeout=3)
        if resp.status_code == 200:
            classrooms = resp.json()
            cameras = []
            for c in classrooms:
                cam_url_val = c.get('camera_url', '0')
                cameras.append({
                    "classroom_id": c['id'],
                    "classroom_name": c['name'],
                    "camera_index": cam_url_val,
                    "camera_url": get_camera_url(cam_url_val)
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
    target_url = None

    if cam is not None:
        # Direct camera index mode (idle browsing)
        target_url = get_camera_url(cam)
    elif v and v != "default":
        # Session-based camera selection
        target_sess = None
        if current_session_info:
            target_sess = next((s for s in current_session_info if str(s.get('id')) == v), None)
        
        if target_sess:
            target_url = get_camera_url(target_sess.get('camera_url', '0'), target_sess.get('camera_name'))
        
        # If still no URL or fallback needed
        if not target_url:
            log("📡", "STREAM", f"Session {v} lookup failed. Searching for ANY active camera...", "warn")
            if camera_workers:
                # Prioritize a worker that is already running
                target_url = next(iter(camera_workers.keys()))
                log("✅", "STREAM", f"Fallback found active worker: {target_url}")
            elif current_session_info:
                # If no workers, use the first session in the list
                s = current_session_info[0]
                target_url = get_camera_url(s.get('camera_url', '0'), s.get('camera_name'))
                log("✅", "STREAM", f"Fallback found pending session: {target_url}")
    
    if not target_url:
         # No target (Idle state) — send an idle placeholder instead of opening hardware
         async def idle_gen():
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(blank, "System Idle", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
         return StreamingResponse(idle_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # Ensure camera is open
    _ensure_camera(target_url)

    def frame_generator():
        while True:
            worker = camera_workers.get(target_url)
            if not worker or not worker.running:
                # Camera was cleaned up — re-open it since viewer is still active
                _ensure_camera(target_url)
                worker = camera_workers.get(target_url)
            if worker and worker.running:
                worker._last_stream_time = time.time()  # Keep-alive for cleanup
                ret, buffer = cv2.imencode('.jpg', worker.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # Send a blank frame with error message
                blank = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(blank, f"Stream unavailable", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.putText(blank, f"{target_url}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                ret, buffer = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.066)  # ~15 FPS

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    """Generate a face embedding from an uploaded image."""
    temp_path = os.path.join(os.getcwd(), f"temp_{int(time.time())}_{file.filename}")
    log("📁", "EMBED", f"Processing embedding request for: {file.filename}")
    
    try:
        # Securely write the file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        log("🔍", "EMBED", f"File saved to {temp_path}, starting DeepFace analysis...")
        
        try:
            # Try with detection first
            objs = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=True, detector_backend="opencv")
        except Exception as detection_err:
            log("⚠️", "EMBED", f"Face detection failed: {detection_err}. Retrying without enforcement...", "warn")
            # Fallback: retry without enforcement
            objs = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=False, detector_backend="opencv")
            
        # Clean up immediately
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if objs and len(objs) > 0:
            log("✅", "EMBED", "Embedding generated successfully", "success")
            return {"embedding": objs[0]["embedding"]}
        
        log("❌", "EMBED", "No face detected in the image", "error")
        return {"error": "Could not find a valid face signature. Ensure the photo is clear and contains a face."}
        
    except Exception as e:
        log("❌", "EMBED", f"Critical embedding error: {str(e)}", "error")
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        return {"error": f"AI Engine Error: {str(e)}"}

# ── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    # Suppress noisy uvicorn/asyncio shutdown tracebacks
    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    try:
        uvicorn.run(app, host="0.0.0.0", port=AI_PORT, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        log("👋", "SYSTEM", "AI Service stopped. Goodbye!", "info")
        for w in camera_workers.values():
            w.stop()
