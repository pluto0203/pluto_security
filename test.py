# main.py


import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "backend/models/bytetrack/ByteTrack"))

from backend.models.detections.smoking_zones import SmokingTracker
import cv2




tracker = SmokingTracker()
cap = cv2.VideoCapture(r"D:\FPTUniversity\Practice_Python\pluto_security\data\test_data\hutthuoc_5.mp4")

while True:
    ret, frame = cap.read()
    if not ret: break

    result_frame, events = tracker.process_frame(frame)

    cv2.imshow("Smoking Violation Detection", result_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()