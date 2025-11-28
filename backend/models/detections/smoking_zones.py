# smoking_zones.py
from collections import defaultdict
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
from datetime import datetime
from ultralytics import YOLO
from backend.models.tracking.yolo_bytetrack import YOLOByteTracker

# Cấu hình (configs.config.py)
from configs.config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class SmokingZoneManager:
    def __init__(self, json_path: Path = ZONE_FILE):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.zones = data.get("smoking_forbidden_zones", [])
        logger.info(f"Loaded {len(self.zones)} vùng cấm hút thuốc")

    def is_bbox_in_zone(self, bbox) -> Tuple[bool, Dict | None]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for zone in self.zones:
            pts = np.array(zone["points"], np.int32)
            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                return True, zone
        return False, None

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        for zone in self.zones:
            pts = np.array(zone["points"], np.int32)
            color = tuple(zone.get("color", [0, 255, 255]))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 3)
            cv2.putText(overlay, zone["name"], (pts[0][0] + 10, pts[0][1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


class SmokingTracker:
    def __init__(self):
        self.detector = YOLO(SMOKING_DETECTION_MODEL)
        self.zone_manager = SmokingZoneManager(ZONE_FILE)

        # Tracker chỉ track person
        self.person_tracker = YOLOByteTracker(
            model_path=SMOKING_DETECTION_MODEL,
            conf_threshold=CONFIDENCE_PERSON_THRESHOLD,
            track_thresh=0.1,
            track_buffer=90,
            match_thresh=0.8
        )

        # Bộ đếm vi phạm + thời gian track còn sống
        self.violation_counter = defaultdict(int)  # track_id → số frame vi phạm liên tục
        self.last_seen = {}  # track_id → frame_idx cuối cùng xuất hiện
        self.frame_idx = 0

        # Cấu hình dọn dẹp bộ nhớ
        self.MAX_LOST_FRAMES = 300  # nếu mất > 10 giây (30fps) → xóa khỏi bộ nhớ

    def _cleanup_lost_tracks(self):
        """Xóa các track_id đã mất quá lâu → tránh memory leak 24/7"""
        current_frame = self.frame_idx
        lost_tracks = [
            tid for tid, last_frame in self.last_seen.items()
            if current_frame - last_frame > self.MAX_LOST_FRAMES
        ]
        for tid in lost_tracks:
            self.violation_counter.pop(tid, None)
            self.last_seen.pop(tid, None)
        if lost_tracks:
            logger.debug(f"Đã dọn {len(lost_tracks)} track lost")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        self.frame_idx += 1
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        result_frame = self.zone_manager.draw_zones(result_frame)

        # === 1. Track person ===
        person_tracks = self.person_tracker.update(frame)

        # === 2. Detect hand + mouth (toàn frame) ===
        results = self.detector(frame, conf=0.3, verbose=False)[0]
        hand_mouth_dets = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in (0, 1):  # hand hoặc mouth = đang hút thuốc
                    conf = float(box.conf[0])
                    if conf >= CONF_HAND_MOUTH_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        hand_mouth_dets.append((x1, y1, x2, y2, cls_id))

        # === 3. Kiểm tra từng track hiện tại ===
        current_track_ids = {t["track_id"] for t in person_tracks}
        violation_events = []

        for track in person_tracks:
            track_id = track["track_id"]
            bbox = track["bbox_pixels"]
            x1, y1, x2, y2 = bbox

            # Cập nhật thời gian xuất hiện
            self.last_seen[track_id] = self.frame_idx

            in_zone, zone_info = self.zone_manager.is_bbox_in_zone(bbox)

            # Kiểm tra có thuốc trong người không
            has_smoking = False
            for hx1, hy1, hx2, hy2, cls_id in hand_mouth_dets:
                hc_x, hc_y = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                if x1 <= hc_x <= x2 and y1 <= hc_y <= y2:
                    has_smoking = True
                    color = (0, 0, 255) if cls_id == 0 else (0, 255, 255)
                    cv2.rectangle(result_frame, (hx1, hy1), (hx2, hy2), color, 3)
                    cv2.putText(result_frame, "THUOC", (hx1, hy1 - 10), 0, 0.8, color, 2)
                    break

            # Cập nhật counter
            if in_zone and has_smoking:
                self.violation_counter[track_id] += 1
            else:
                self.violation_counter[track_id] = 0

            # Vẽ bbox person
            color = (0, 0, 255) if (in_zone and has_smoking) else \
                (0, 255, 255) if in_zone else (255, 255, 0)
            thick = 6 if (in_zone and has_smoking) else 3
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(result_frame, f"ID:{track_id}", (x1, y1 - 10), 0, 1, (0, 255, 0), 3)

            # BÁO ĐỘNG
            if self.violation_counter[track_id] >= MIN_FRAMES:
                zone_name = zone_info["name"] if zone_info else "Khu vực cấm"
                cv2.putText(result_frame, f"VI PHAM HUT THUOC ID {track_id}", (80, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                cv2.putText(result_frame, zone_name, (80, 200), 0, 1.8, (0, 0, 255), 4)
                logger.warning(f"VI PHẠM HÚT THUỐC | ID: {track_id} | Khu vực: {zone_name}")
                violation_events.append({
                    "track_id": track_id,
                    "zone": zone_name,
                    "frame": self.frame_idx,
                    "timestamp": datetime.now().isoformat()
                })

        # === 4. DỌN DẸP TRACK LOST (QUAN TRỌNG CHO 24/7) ===
        self._cleanup_lost_tracks()

        # Thống kê
        active = len(current_track_ids)
        violating = sum(1 for c in self.violation_counter.values() if c >= MIN_FRAMES)
        cv2.putText(result_frame, f"Active: {active} | Vi pham: {violating}", (50, h - 40),
                    0, 1, (255, 255, 255), 2)

        return result_frame, violation_events

    def reset(self):
        self.person_tracker.reset()
        self.violation_counter.clear()
        self.last_seen.clear()
        self.frame_idx = 0
        logger.info("SmokingTracker đã được reset hoàn toàn")