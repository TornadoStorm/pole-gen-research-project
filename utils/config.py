import os

import yaml
from dotenv import load_dotenv
from segments import SegmentsClient

load_dotenv()

SEGMENTS_CLIENT = SegmentsClient(os.getenv("SEGMENTS_AI_API_KEY"))

with open("config.yaml") as f:
    config = yaml.safe_load(f)

N_POINTS: int = config.get("n_points", 1024)
SEED: int = config.get("seed", 42)

ts = config.get("train_data", {})
TRAIN_DATA_SIZE: int = ts.get("size", 65536)
TRAIN_DATA_PATH: str = ts.get("path", "data/train")
TRAIN_DATA_BATCH_SIZE: int = ts.get("batch_size", 32)
TRAIN_DATA_WORKERS: int = ts.get("workers", 4)

ts = config.get("test_data", {})
TEST_DATA_PATH: str = ts.get("path", "data/test")
TEST_DATA_BATCH_SIZE: int = ts.get("batch_size", 32)
TEST_DATA_WORKERS: int = ts.get("workers", 4)

ts = config.get("valid_data", {})
VALID_DATA_SPLIT: float = ts.get("split", 0.2)
VALID_DATA_BATCH_SIZE: int = ts.get("batch_size", 32)
VALID_DATA_WORKERS: int = ts.get("workers", 4)

ts = config.get("train", {})
TRAIN_EPOCHS: int = ts.get("epochs", 100)
TRAIN_PATH: str = ts.get("path", "data/pointnet")
TRAIN_BEST_MODEL_PATH: str = ts.get("best_model_path", "data/pointnet/best_model.pth")

del ts
del config
