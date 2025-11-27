import logging
import sys
sys.path.append(r"D:\FPTUniversity\Practice_Python\pluto_security\backend\models\bytetrack\ByteTrack")

from backend.models.tracking.yolo_bytetrack import YOLOByteTracker
from pathlib import Path
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Khởi tạo tracker
    tracker = YOLOByteTracker(
        model_path="backend/models/detections/weights/yolo11n.pt",
        device="cuda",
        fps=20
    )

    # Test trên 1 video mẫu
    video_path = "data/test_data/vid1.mp4"       # thay bằng video của bạn

    all_tracks = tracker.process_video(
        video_path=video_path,
        save_annotated=True,            # sẽ lưu video có vẽ bbox + ID
        output_path=Path("data/out_put/my_tracking.mp4")
    )

    print(f"Done! Đã xử lý {len(all_tracks)} frame")
    print("Video kết quả lưu ở: output/tracked_test_video.mp4")