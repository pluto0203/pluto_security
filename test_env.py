import cv2
import pandas as pd
from pathlib import Path
import logging

from ultralytics.data import YOLODataset

from backend.models.detections import yolo_detector
from backend.models.detections.yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    yolo_detector = YOLODetector(
        model_path="backend/models/detections/weights/yolo11n.pt"
    )
    img_path = Path(r"D:\FPTUniversity\Practice_Python\pluto_security\data\MOT20\train\MOT20-01\img1\000001.jpg")
    out_dir = Path("data/out_put")

    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error(f"Could not read {img_path}")

    detections = yolo_detector.detect(frame, save_img=True, save_path=out_dir)
