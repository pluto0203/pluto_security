from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(
            self,
            model_path: Optional[Union[Path, str]] = None,
            confidence_threshold: float = 0.4,
            img_size: int = 640,
            device: str = "cuda"
    ) -> None:

        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.device = device

        if model_path is None:
            model_path = "yolo11n.pt"
            logger.info(f"Using default model: {model_path}")

        logger.info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        logger.info(f"YOLODetector ready | conf: {confidence_threshold} | imgsz: {img_size}")

    def detect_frame(
            self,
            frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in a single frame
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []

        h, w = frame.shape[:2]

        # ĐÚNG CÁCH: truyền tham số trực tiếp vào predict
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.img_size,
            classes=None,   # None = detect all, hoặc [0] nếu chỉ person
            verbose=False,
            device=self.device
        )[0]

        detections: List[Dict[str, Any]] = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = results.names[cls_id]

                # Normalized bbox [0,1]
                norm_bbox = [x1/w, y1/h, x2/w, y2/h]
                pixel_bbox = [int(x1), int(y1), int(x2), int(y2)]

                detections.append({
                    "bbox": norm_bbox,
                    "bbox_pixels": pixel_bbox,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                })

        logger.debug(f"Frame {getattr(self, 'frame_idx', '?')}: Detected {len(detections)} objects")
        return detections

    def detect_video(
            self,
            video_path: Union[str, Path],
            save_annotated: bool = True,
            output_video: Optional[Path] = None,
    ) -> List[List[Dict[str, Any]]]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {video_path.name} | {width}x{height} | {total_frames} frames")

        writer = None
        if save_annotated and output_video:
            output_video = Path(output_video)
            output_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        all_detections: List[List[Dict[str, Any]]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_frame(frame)
            all_detections.append(detections)

            if save_annotated and writer:
                # Dùng results.plot() đúng cách
                results = self.model(frame, conf=self.confidence_threshold, imgsz=self.img_size, verbose=False)[0]
                annotated = results.plot()  # Đây là cách đúng!
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Annotated video saved → {output_video}")

        return all_detections

    def detect_stream(
            self,
            source: Union[int, str] = 0,
    ) -> Generator[tuple[int, np.ndarray, List[Dict[str, Any]]], None, None]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open stream: {source}")

        logger.info(f"Starting real-time detection on source: {source}")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_frame(frame)

            # Tạo annotated frame
            results = self.model(frame, conf=self.confidence_threshold, imgsz=self.img_size, verbose=False)[0]
            annotated_frame = results.plot()

            yield frame_idx, annotated_frame, detections
            frame_idx += 1

        cap.release()

    def to_json(self, detections: List[Dict[str, Any]]) -> str:
        return json.dumps(detections, indent=2, ensure_ascii=False)

    def __del__(self) -> None:
        if hasattr(self, "model"):
            del self.model
            logger.debug("YOLODetector model cleaned up")