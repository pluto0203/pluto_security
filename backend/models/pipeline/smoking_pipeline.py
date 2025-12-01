import cv2
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from configs.config import *
from backend.models.detections.smoking_zones import SmokingTracker
from backend.models.tracking.yolo_bytetrack import YOLOByteTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("smoking_detection_pipeline.log"),
        logging.StreamHandler
    ]
)
logger = logging.getLogger("SMOKING_DETECTION_PIPELINE")


class SmokingDetectionPipeline:
    def __init__(self):
        logger.info(f"Smoking detection pipeline started.")
        
        self.person_tracker = YOLOByteTracker(
            model_path=SMOKING_DETECTION_MODEL,
            conf_threshold= CONFIDENCE_PERSON_THRESHOLD,
            track_thresh = 0.1,
            track_buffer = 90,
            fps = 20
        )
        
        self.smoking_tracker = SmokingTracker()
        self.output_dir = Path("D:/FPTUniversity/Practice_Python/pluto_security/data/out_put/pipeline_W1")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Smoking detection pipeline ready at {self.output_dir}")

    def run(
            self,
            video_path: Path,
            save_video: bool = True,
            show_video: bool = True
    ) -> Dict:
        """
        :param video_path: duong dan den video can detect
        :param save_video: TRUE or FALSE
        :param show_video: TRUE or FALSE
        :return:
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video {video_path} not found")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")


        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing video {video_path.name} | fps: {fps} | width: {width} | height: {height} | total_frames: {total_frames}")

        writer = None
        if save_video:
            output_video = self.output_dir / f"RES_w1_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height) )
            logger.info(f"Saving video {output_video}")

        self.person_tracker.reset()
        self.smoking_tracker.reset()

        stats = {
            "video_path": video_path,
            "process_frames": 0,
            "violations": [],
            "total_violations": 0,
            "start_time": datetime.now().isoformat()
        }

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            person_tracks = self.person_tracker.update(frame)
            result_frame, events = self.smoking_tracker.process_frame(frame)

            if save_video and writer:
                writer.write(result_frame)

            if show_video:
                cv2.imshow("SMOKING_DETECTION_PIPELINE", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info(f"SMOKING_DETECTION_PIPELINE ended by user!")
                    break

            for event in events:
                event["frame"] = frame_idx
                event["timestamp"] = datetime.now().isoformat()
                stats["violations"].append(event)

                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx} frames | TOTAL violations: {stats['total_violations']}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        stats["process_frames"] = frame_idx
        stats["total_violations"] = len(set(e["track_id"] for e in stats["violations"]))
        stats["end_time"] = datetime.now().isoformat()

        logger.info(f"COMPLETED SMOKING_DETECTION_PIPELINE!")
        logger.info(f"Processed {stats["process_frames"]} frames!")
        logger.info(f"Total violated person: {stats['total_violations']}")
        # logger.info(f"Total violations: {len(stats['total_violations'])}")

        if save_video:
            logger.info(f"Saving video {output_video}")

        return stats

    def run_demo(self):
        logger.info(f"Run demo!")
        stats = self.run(VIDEO_TEST, save_video=False, show_video=True)
        print('\n THONG KE VI PHAM ')
        for k, v in stats.items():
            if k != 'violations':
                print(f"{k}: {v}")

        return stats



















