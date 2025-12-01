# main.py


import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "backend/models/bytetrack/ByteTrack"))

from backend.models.pipeline.smoking_pipeline import *

if __name__ == '__main__':
    pipeline = SmokingDetectionPipeline()
    pipeline.run_demo()