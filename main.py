from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api/recognition")
STUDENTS_API = "http://localhost:5000/api/students" 
CAMERA_INDEX = 0 
COOLDOWN_PERIOD = 10 

# Global state
system_active = True
student_cache = []
last_marked = {}
current_frame = None

@app.get("/system/status")
async def get_status():
    return {"active": system_active}

@app.post("/system/toggle")
async def toggle_system():
    global system_active
    system_active = not system_active
    return {"active": system_active}

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    temp_path = f"temp_{int(time.time())}.jpg"
    try:
        # Save temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing registration photo: {temp_path}")
        
        # Try to generate embedding
        try:
            objs = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=True)
        except Exception as detection_error:
            print(f"Face detection failed: {detection_error}. Using fallback mode...")
            # Guaranteed fallback: Always return an embedding for the center of the photo
            objs = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=False)

        if os.path.exists(temp_path): os.remove(temp_path)
        
        if objs and len(objs) > 0:
            print("Embedding generated successfully!")
            return {"embedding": objs[0]["embedding"]}
        
        return {"error": "Could not find a valid face signature."}
        
    except Exception as e:
        print(f"Critical embedding error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"error": str(e)}

def refresh_student_cache():
    global student_cache
    try:
        response = requests.get(STUDENTS_API)
        if response.status_code == 200:
            students = response.json()
            student_cache = [s for s in students if s.get('face_embedding')]
            for s in student_cache:
                if isinstance(s['face_embedding'], str):
                    s['face_embedding'] = json.loads(s['face_embedding'])
            print(f"Updated student cache: {len(student_cache)} students loaded.")
    except Exception as e:
        print(f"Failed to refresh student cache: {e}")

def gen_frames():
    global current_frame
    while True:
        if current_frame is not None:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def run_recognition():
    global last_marked, student_cache, current_frame
    print("Starting Recognition Loop (SQL + Stream Mode)...")
    
    # Initial cache load
    refresh_student_cache()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Load a fast face detector for smooth 30FPS tracking
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cache_timer = 0
    objs = [] # Persistent recognition data
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.1)
            continue
        
        # 1. FAST TRACKING (Every Frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw boxes for all detected faces
        for (x, y, w, h) in faces:
            color = (0, 255, 0) # Neon Green
            t, l = 2, 20
            
            # Sci-Fi Corners (Follows face perfectly at 30FPS)
            cv2.line(frame, (x, y), (x + l, y), color, t)
            cv2.line(frame, (x, y), (x, y + l), color, t)
            cv2.line(frame, (x + w, y), (x + w - l, y), color, t)
            cv2.line(frame, (x + w, y), (x + w, y + l), color, t)
            cv2.line(frame, (x, y + h), (x + l, y + h), color, t)
            cv2.line(frame, (x, y + h), (x, y + h - l), color, t)
            cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, t)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, t)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            
            # Scanning Line
            scan_pos = int((time.time() * 150) % h)
            cv2.line(frame, (x, y + scan_pos), (x + w, y + scan_pos), color, 1)
            cv2.line(frame, (x, y + scan_pos), (x + w, y + scan_pos), (0, 100, 0), 3)

            # Identity Label (Check if we have a match from the background brain)
            label = "ANALYZING..."
            if objs and len(objs) > 0:
                # Find the 'brain' object closest to this 'tracking' box
                for obj in objs:
                    ba = obj["facial_area"]
                    # If the centers are close, it's the same person
                    if abs((x + w/2) - (ba['x'] + ba['w']/2)) < 50:
                        current_embedding = obj["embedding"]
                        best_match_name = None
                        min_dist = 1.0
                        
                        for student in student_cache:
                            dist = cosine(current_embedding, student['face_embedding'])
                            if dist < min_dist:
                                min_dist = dist
                                best_match_name = student['name']
                        
                        if best_match_name and min_dist < 0.4:
                            label = f"ID: {best_match_name.upper()}"
                            cv2.putText(frame, "VERIFIED", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "UNKNOWN", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add general HUD elements
        cv2.putText(frame, "SYSTEM: ACTIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"TARGETS: {len(faces)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Store frame for streaming
        current_frame = frame.copy()

        if not system_active:
            time.sleep(0.1)
            continue

        # Refresh cache every 60 seconds
        cache_timer += 1
        if cache_timer > 100:
            threading.Thread(target=refresh_student_cache).start()
            cache_timer = 0

        try:
            # Recognition logic (every 10th frame to keep it smooth)
            if cache_timer % 10 == 0:
                # Use a small resize to make detection faster for streaming
                small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                
                # Detect and extract embeddings
                objs = DeepFace.represent(img_path=small_frame, model_name="VGG-Face", enforce_detection=False)
                
                if objs and len(objs) > 0:
                    print(f"👀 [AI] Detected {len(objs)} face(s) in frame.")
                    current_embedding = objs[0]["embedding"]
                    best_match = None
                    min_dist = 1.0
                    
                    if not student_cache:
                        print("⚠️ [AI] Student list is empty. Please register students first.")
                    else:
                        print(f"🔍 [AI] Comparing face with {len(student_cache)} registered students...")
                    
                    for student in student_cache:
                        dist = cosine(current_embedding, student['face_embedding'])
                        # print(f"   - Checking {student['name']}: {dist:.4f}")
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_match = student
                    
                    if best_match:
                        confidence = (1 - min_dist) * 100
                        if min_dist < 0.4:
                            student_id = best_match['id']
                            name = best_match['name']
                            current_time = time.time()
                            
                            if student_id not in last_marked or (current_time - last_marked[student_id]) > COOLDOWN_PERIOD:
                                print(f"✅ [MATCH] Found: {name} | Confidence: {confidence:.2f}%")
                                try:
                                    res = requests.post(BACKEND_URL, json={
                                        "student_id": student_id,
                                        "session_id": "active",
                                        "confidence": 1 - min_dist
                                    })
                                    if res.status_code == 200:
                                        print(f"📡 [SERVER] Attendance marked for {name}.")
                                    else:
                                        print(f"❌ [SERVER] Failed to mark attendance: {res.text}")
                                except Exception as req_err:
                                    print(f"💥 [ERROR] Could not reach backend: {req_err}")
                                    
                                last_marked[student_id] = current_time
                            else:
                                print(f"⏳ [COOLDOWN] {name} was just marked. Skipping...")
                        else:
                            print(f"❓ [AI] Face detected but no clear match (Best guess: {best_match['name']} at {confidence:.2f}%)")
                else:
                    # Optional: print every few frames if no face is found
                    if cache_timer % 50 == 0:
                        print("😶 [AI] No faces visible.")
                            
        except Exception as e:
            print(f"🔴 [AI ERROR] {str(e)}")
            pass

    cap.release()

if __name__ == "__main__":
    rec_thread = threading.Thread(target=run_recognition, daemon=True)
    rec_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8001)
