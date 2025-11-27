from __future__ import annotations

import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from types import SimpleNamespace

from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)

class YOLOByteTracker:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.3,
        img_size: int = 640,
        device: str = "cuda",
        track_thresh: float = 0.2,
        track_buffer: int = 120,
        match_thresh: float = 0.7,
        fps: int = 30
    ) -> None:
        logger.info(f"[INIT] Khởi tạo YOLOByteTracker | Device: {device} | Model: {model_path}")

        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.device = device

        # Load YOLO
        logger.info(f"[INIT] Đang load YOLO model: {model_path}")
        self.detector = YOLO(model_path)
        self.detector.to(device)
        logger.info("[INIT] YOLO loaded thành công!")

        # ByteTrack args
        args = SimpleNamespace()
        args.track_thresh = track_thresh
        args.track_buffer = track_buffer
        args.match_thresh = match_thresh
        args.mot20 = False
        args.with_reid = False

        self.tracker = BYTETracker(args, frame_rate=fps)
        logger.info(f"[INIT] ByteTrack khởi tạo: thresh={track_thresh}, buffer={track_buffer}")

        self.frame_idx = 0
        logger.info("[INIT] YOLOByteTracker sẵn sàng!")

    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        self.frame_idx += 1
        h, w = frame.shape[:2]

        logger.debug(f"\n--- Frame {self.frame_idx} | Kích thước: {w}x{h} ---")

        # === 1. YOLO DETECTION ===
        results = self.detector(
            frame,
            conf=self.conf_threshold,
            classes=0,  # only person
            imgsz=self.img_size,
            verbose=False,
            device=self.device
        )[0]

        num_dets = len(results.boxes) if results.boxes is not None else 0
        logger.info(f"[DETECT] Phát hiện {num_dets} person")

        if num_dets > 0:
            boxes = results.boxes.xyxy.cpu()  # [N,4] frame gốc
            scores = results.boxes.conf.cpu()  # [N]
            dets_for_tracker = torch.cat([boxes, scores.unsqueeze(1)], dim=1)  # [N,5]
            logger.debug(f"[INPUT] dets_for_tracker shape: {dets_for_tracker.shape}")
            logger.debug(f"[INPUT] Sample dets_for_tracker[0]: {dets_for_tracker[0].tolist()}")
        else:
            dets_for_tracker = torch.empty((0, 5), dtype=torch.float32)
            logger.warning("[INPUT] Không có detection → gửi tensor rỗng vào tracker")

        # === 2. ByteTrack update ===
        online_targets = self.tracker.update(
            output_results=dets_for_tracker,
            img_info=(h, w),
            img_size=(self.img_size, self.img_size)  # tracker cần img_size nhưng input đã frame gốc
        )

        logger.info(f"[TRACK] ByteTrack trả về {len(online_targets)} track đang active")

        # === 3. Format output ===
        tracks = []
        for t in online_targets:
            tlwh = t.tlwh  # tracker output, đã ở frame gốc
            track_id = t.track_id
            score = t.score

            x1, y1, bw, bh = tlwh
            x2, y2 = x1 + bw, y1 + bh

            bbox_pixels = [
                max(0, int(x1)),
                max(0, int(y1)),
                min(w - 1, int(x2)),
                min(h - 1, int(y2))
            ]

            logger.debug(f"[OUTPUT] Track ID {track_id} tlwh: {tlwh}, pixels: {bbox_pixels}, score: {score}")

            tracks.append({
                "track_id": int(track_id),
                "bbox": [bbox_pixels[0] / w, bbox_pixels[1] / h, bbox_pixels[2] / w, bbox_pixels[3] / h],
                "bbox_pixels": bbox_pixels,
                "confidence": float(score),
                "frame": self.frame_idx,
                "class_name": "person"
            })

        return tracks

    def process_video(self, video_path: Union[str, Path], save_annotated: bool = True, output_path: Optional[Path] = None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Không mở được video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video: {w}x{h}, {fps} FPS")

        writer = None
        if save_annotated:
            output_path = output_path or Path("output") / f"tracked_{Path(video_path).name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            logger.info(f"Đang ghi video output → {output_path}")

        all_tracks = []
        self.reset()

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            tracks = self.update(frame)
            all_tracks.append(tracks)

            if save_annotated and writer:
                frame_to_write = frame.copy()
                if frame_to_write.dtype != np.uint8:
                    frame_to_write = np.clip(frame_to_write*255, 0, 255).astype(np.uint8)

                for t in tracks:
                    x1, y1, x2, y2 = t["bbox_pixels"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w-1, x2), min(h-1, y2)
                    cv2.rectangle(frame_to_write, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame_to_write, f"ID:{t['track_id']}", (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                writer.write(frame_to_write)

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Hoàn thành! Video lưu tại: {output_path}")

        logger.info(f"Done! Đã xử lý {frame_count} frame")
        return all_tracks

    def reset(self):
        self.frame_idx = 0
        self.tracker = BYTETracker(self.tracker.args, frame_rate=30)
        logger.info("Tracker đã được reset")
