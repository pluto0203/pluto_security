import cv2
from ultralytics import YOLO
from pathlib import Path

# Load model
data_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/data/smoking_data/data.yaml")
model_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/backend/models/detections/weights/yolo11n.pt")
checkpoint_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/backend/models/detections/runs/detect/yolo11_smoking_hand_mouth/weights/last.pt")
best_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/backend/models/detections/runs/detect/yolo11_smoking_hand_mouth/weights/best.pt")


if __name__ == '__main__':

    model = YOLO(best_path)

    # Input video
    # video_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/data/test_data/hutthuoc_5.mp4")
    video_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/data/test_data/vid1.mp4")
    cap = cv2.VideoCapture(video_path)

    # Output video
    # output_path = r"D:\FPTUniversity\Practice_Python\pluto_security\videos\output_smoking.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, imgsz=640, conf=0.4)

        # Visualize predictions
        annotated_frame = results[0].plot()

        # Show
        cv2.imshow("Smoking Detection", annotated_frame)

        # Save output
        # out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
