#!/usr/bin/env python
# scripts/eda_mot20_det_only.py

"""
EDA thuần det.txt cho MOT20 (hoặc bất kỳ file detection nào theo format MOT Challenge)
Ví dụ path: .../MOT20-04/det/det.txt
Dùng cho Week 1 - Day 2 khi bạn chỉ có kết quả detector (YOLO, CenterNet, v.v.)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ============================== LOGGING ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eda_det")


# ============================== CORE ==============================
def load_det_file(det_path: Path) -> pd.DataFrame:
    """
    Load file det.txt (10 cột) theo chuẩn MOT Challenge.

    """
    df = pd.read_csv(
        det_path,
        header=None,
        names=[
            "frame", "id", "left", "top", "width", "height",
            "conf", "class", "visibility", "extra"
        ],
    )
    # Lọc người + confidence hợp lệ
    df = df[df["conf"] >= 0.0]
    df["area"] = df["width"] * df["height"]
    df["height"] = df["height"].astype(float)

    log.info(f"Loaded {det_path.parent.name}/{det_path.name} → {len(df):,} detections")
    return df


def analyze_det_file(det_path: Path) -> dict:
    df = load_det_file(det_path)

    return {
        "sequence": det_path.parts[-3],           # MOT20-04
        "total_frames": int(df["frame"].max()),
        "total_detections": len(df),
        "avg_density": round(len(df) / df["frame"].max(), 2),
        "max_density": int(df.groupby("frame").size().max()),
        "avg_conf": round(df["conf"].mean(), 4),
        "low_conf_rate (<0.5)": round((df["conf"] < 0.5).mean(), 4),
        "avg_height": round(df["height"].mean(), 1),
        "p5_height": round(np.percentile(df["height"], 5), 1),
        "p95_height": round(np.percentile(df["height"], 95), 1),
    }


def run_det_only_eda(root_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả det.txt
    det_files = list(root_dir.rglob("det.txt"))
    log.info(f"Tìm thấy {len(det_files)} file det.txt")

    results = []
    all_heights = []

    for det_path in tqdm(det_files, desc="EDA det files"):
        stats = analyze_det_file(det_path)
        results.append(stats)

        df = load_det_file(det_path)
        all_heights.extend(df["height"].tolist())

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "det_summary.csv", index=False)

    # ============================== PLOTS ==============================
    plt.figure(figsize=(14, 9))

    plt.subplot(2, 3, 1)
    sns.barplot(data=summary_df.sort_values("avg_density", ascending=False),
                x="sequence", y="avg_density",hue="sequence", palette="viridis")
    plt.title("Avg Detections per Frame")
    plt.xticks(rotation=45)

    plt.subplot(2, 3, 2)
    sns.barplot(data=summary_df.sort_values("max_density", ascending=False),
                x="sequence", y="max_density",hue="sequence", palette="Reds")
    plt.title("Peak Crowd (max detections in 1 frame)")
    plt.xticks(rotation=45)

    plt.subplot(2, 3, 3)
    plt.hist(all_heights, bins=80, color="#66b3ff", edgecolor="black", alpha=0.8)
    plt.axvline(np.percentile(all_heights, 5), color="red", linestyle="--", label="5th percentile")
    plt.title("Bbox Height Distribution → small object detection challenge")
    plt.xlabel("Height (px)")
    plt.legend()

    plt.subplot(2, 3, 4)
    sns.boxplot(data=summary_df, x="sequence", y="avg_conf")
    plt.title("Average Confidence Score")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "det_eda_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ============================== REPORT ==============================
    report = {
        "total_sequences": len(summary_df),
        "global_avg_density": round(summary_df["avg_density"].mean(), 2),
        "global_peak": int(summary_df["max_density"].max()),
        "most_crowded_seq": summary_df.loc[summary_df["max_density"].idxmax(), "sequence"],
        "small_object_ratio (<40px)": round((np.array(all_heights) < 40).mean(), 3),
        "recommend": {
            "yolo_input_size": "1280x1280" if (np.array(all_heights) < 50).mean() > 0.2 else "640x640",
            "conf_threshold": 0.25,
            "nms_threshold": 0.45,
            "note": "Rất nhiều small objects → dùng YOLOv8x + img_sz=1280 nếu muốn mAP cao"
        }
    }

    with open(output_dir / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"EDA det.txt hoàn tất! → {output_dir.resolve()}")


# ============================== CLI ==============================
if __name__ == "__main__":
    root_dir = Path(r'D:\FPTUniversity\Practice_Python\pluto_security\data\MOT20Labels\test')
    out_dir = Path(r'D:\FPTUniversity\Practice_Python\pluto_security\scripts\eda\eda_results')
    run_det_only_eda(root_dir, out_dir)
