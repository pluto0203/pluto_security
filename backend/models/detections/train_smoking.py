# train_smoking.py
from ultralytics import YOLO
from pathlib import Path

data_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/data/smoking_data/data.yaml")
model_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/backend/models/detections/weights/yolo11n.pt")
checkpoint_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/backend/models/detections/runs/detect/yolo11_smoking_hand_mouth/weights/last.pt")
best_path = Path("D:/FPTUniversity/Practice_Python/pluto_security/backend/models/detections/runs/detect/yolo11_smoking_hand_mouth/weights/best.pt")


if __name__ == '__main__':

    # model = YOLO(checkpoint_path)    # n = nhẹ nhất, nhanh nhất
    #
    # model.train(
    #     data=data_path,
    #     epochs=100,           # 50–100 là đủ
    #     imgsz=640,
    #     batch=8,             # tăng lên 32 nếu GPU mạnh
    #     device= 0,             # 0 = GPU đầu tiên, "cpu" nếu không có
    #     patience=20,          # early stopping
    #     name="yolo11_smoking_hand_mouth",
    #     # project="runs/smoking",
    #     exist_ok=True,
    #     augment=True,         # bật augmentation (rất quan trọng!)
    #     lr0=0.01,
    #     optimizer="AdamW",
    #     close_mosaic=10,      # tắt mosaic 10 epoch cuối
    #     amp=True,
    #     workers = 0,
    #     resume = True,
    #     cache=False
    # )

    evaluate = YOLO(best_path)

    evaluate.val(
        data = data_path,
        split = "test",
        batch=16,
        conf=0.4,
        iou=0.5,
        plots=True,
        save_json=True,
        name="test_python_result"
    )