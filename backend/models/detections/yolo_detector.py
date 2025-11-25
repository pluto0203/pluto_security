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
            model_path = "yolov11n.pt"
            logger.info(f"Using default model: {model_path}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.model.overrides['conf'] = confidence_threshold
        self.model.overrides['imgsz'] = img_size

        logger.info(f"YOLODetector initialized with confidence threshold: {self.confidence_threshold}, image size: {self.img_size}")

    def detect_frame(
            self,
            frame: np.ndarray,
            save_img: bool = False,
            save_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:

        if frame is None or frame.size == 0:
            logger.warning(f"Frame size is empty, returning empty frame")
            return []

        results = self.model(frame, verbose=False)[0]
        detections: List[Dict[str, Any]] = []

        for box in results.boxes:
            if box.conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results.names[cls_id]

            h,w = frame.shape[:2]
            norm_bbox = [
                x1/w, y1/h, x2/w, y2/h
            ]

            detection = {
                "bbox": norm_bbox,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": float(conf),
            }
            detections.append(detection)
            logger.info(f"Detected box: {norm_bbox}, class: {cls_name}, confidence: {conf}")

            if save_img and save_path:
                annotated = results.plot()
                cv2.imwrite(str(save_path / f"{cls_name}.png"), annotated)
                logger.debug(f"Saved annotated image to {save_path}")

            logger.debug(f"Detected {len(detections)} detections")
            return detections

    def detect_video(
            self,
            video_path: Union[str, Path],
            save_annotated: bool = False,
            output_video: Optional[Path] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect toàn bộ video, trả về list[per-frame detections]
        Dùng cho: upload video → xử lý backend (Week 2)
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing video: {video_path.name} | {total_frames} frames | {fps:.1f} FPS")

        # Setup video writer nếu cần lưu
        writer = None
        if save_annotated and output_video:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        all_detections: List[List[Dict[str, Any]]] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect(frame)
            all_detections.append(detections)

            # Annotate + save nếu cần
            if save_annotated:
                annotated_frame = self.model.plot(frame=frame, boxes=detections)  # custom plot nếu muốn
                if writer:
                    writer.write(annotated_frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Annotated video saved → {output_video}")

        logger.info(f"Video detection completed: {len(all_detections)} frames")
        return all_detections

    def detect_stream(
            self,
            source: Union[int, str] = 0,  # 0 = webcam, hoặc RTSP URL
    ) -> Generator[tuple[int, np.ndarray, List[Dict[str, Any]]], None, None]:
        """
        Real-time detection generator.
        Dùng cho: webcam demo (Week 3) và livestream dashboard
        Yields: (frame_idx, annotated_frame, detections_json)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open stream: {source}")

        logger.info(f"Starting real-time detection on source: {source}")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Stream ended or error")
                break

            detections = self.detect(frame)
            annotated_frame = self.model.plot(frame=frame)  # hoặc tự vẽ bbox đẹp hơn

            yield frame_idx, annotated_frame, detections
            frame_idx += 1

        cap.release()
        logger.info("Stream detection stopped")


    def to_json(self, detections: List[Dict[str, Any]]) -> str:
        return json.dumps(detections, indent=2)

    def __del__(self) -> None:
        """Clean up model"""
        if hasattr(self, "model"):
            del self.model
            logger.debug(f"Deleted YOLODetector model")