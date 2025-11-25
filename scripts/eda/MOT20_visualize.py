import cv2
import pandas as pd
from pathlib import Path

def visualize_one_frame(seq_path: Path, frame_idx: int = 1):
    img_path = seq_path / "img1" / f"{frame_idx:06d}.jpg"
    gt_path = seq_path / "gt" / "gt.txt"

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    # Load GT
    df = pd.read_csv(
        gt_path, header=None,
        names=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    )

    # Filter đúng frame
    rows = df[df["frame"] == frame_idx]

    # Vẽ từng bbox
    for _, row in rows.iterrows():
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        track_id = int(row["id"])

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"ID {track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )

    cv2.imshow(f"GT Frame {frame_idx}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    seq = Path(r"D:\FPTUniversity\Practice_Python\pluto_security\data\MOT20\train\MOT20-01")
    visualize_one_frame(seq, frame_idx=1)
