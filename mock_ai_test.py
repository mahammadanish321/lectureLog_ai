import requests
import json
import time

BACKEND_URL = "http://localhost:5000/api/recognition"

def test_recognition(student_id, session_id):
    print(f"--- Mock AI: Sending recognition for Student {student_id} in Session {session_id} ---")
    data = {
        "student_id": student_id,
        "session_id": session_id,
        "confidence": 0.95
    }
    
    try:
        response = requests.post(BACKEND_URL, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test values
    sid = input("Enter Student ID to mark: ")
    sess_id = input("Enter Active Session ID: ")
    test_recognition(int(sid), int(sess_id))
