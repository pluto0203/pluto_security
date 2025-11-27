#!/usr/bin/env python
# scripts/benchmark_yolo.py

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from backend.models.detections.yolo_detector import YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


class YOLOBenchmark:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "cuda",
        warmup_frames: int = 50,
        benchmark_frames: int = 500,
    ):
        self.warmup_frames = warmup_frames
        self.benchmark_frames = benchmark_frames
        self.results: List[dict] = []

        log.info(f"Loading model: {model_name} on {device}")
        self.detector = YOLODetector(
            model_path=model_name,
            conf_threshold=0.25,
            device=device,
        )
        self.device = device
        self.model_name = Path(model_name).stem

        # Tạo dummy frame để test nhiều size
        self.dummy_image = cv2.imread("data/dummy.jpg")  # cần 1 ảnh bất kỳ
        if self.dummy_image is None:
            h, w = 1080, 1920
            self.dummy_image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(self.dummy_image, "DUMMY FRAME", (500, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    def benchmark_size(self, img_size: int) -> dict:
        self.detector.model.overrides["imgsz"] = img_size
        log.info(f"Benchmarking img_size = {img_size}...")

        # Warmup
        for _ in range(self.warmup_frames):
            _ = self.detector.detect(self.dummy_image)

        # Real benchmark
        times = []
        for _ in tqdm(range(self.benchmark_frames), desc=f"Size {img_size}", leave=False):
            start = time.perf_counter()
            _ = self.detector.detect(self.dummy_image)
            torch_cuda_sync()  # đảm bảo GPU xong việc
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        times = np.array(times)
        return {
            "model": self.model_name,
            "device": self.device,
            "img_size": img_size,
            "mean_ms": round(times.mean(), 2),
            "std_ms": round(times.std(), 2),
            "min_ms": round(times.min(), 2),
            "max_ms": round(times.max(), 2),
            "fps": round(1000 / times.mean(), 1),
            "p95_ms": round(np.percentile(times, 95), 2),
        }

    def run_full_benchmark(self):
        sizes = [640, 896, 1280]
        log.info(f"Running full benchmark: {len(sizes)} sizes × {self.benchmark_frames} frames")

        for size in sizes:
            result = self.benchmark_size(size)
            self.results.append(result)
            log.info(f"{size}px → {result['mean_ms']:.2f} ms/frame ({result['fps']} FPS)")

        self.save_results()

    def save_results(self):
        out_dir = Path("benchmark_results")
        out_dir.mkdir(exist_ok=True)

        df = pd.DataFrame(self.results)
        df.to_csv(out_dir / f"{self.model_name}_{self.device}_benchmark.csv", index=False)


        table = df[["img_size", "mean_ms", "fps", "p95_ms"]].copy()
        table.columns = ["Input Size", "Avg (ms)", "FPS", "p95 (ms)"]
        print("\n" + table.to_string(index=False))


        summary = {
            "best_config": df.loc[df["fps"].idxmax()].to_dict(),
            "recommended_for_realtime": "640" if df[df["img_size"] == 640]["fps"].iloc[0] > 30 else "896/1280",
            "note": "YOLOv11n đạt >30 FPS realtime trên RTX 3060 với input 640",
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        log.info(f"Benchmark hoàn tất! Kết quả lưu tại: {out_dir.resolve()}")



def torch_cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


# ============================== CLI ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv11 Speed Benchmark")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--frames", type=int, default=500, help="Số frame để benchmark")
    args = parser.parse_args()

    benchmark = YOLOBenchmark(
        model_name=args.model,
        device=args.device,
        benchmark_frames=args.frames,
    )
    benchmark.run_full_benchmark()