import os
import random
import warnings

import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from ai.pointnet_seg.model import PointNetSeg
from electrical_poles.data import download_data
from models.dataset import PointCloudDataset
from pole_gen.data import generate_data
from pole_gen.models import UtilityPoleLabel
from utils.config import *
from utils.logging import warning_format

CLASSES: list = [l.name for l in UtilityPoleLabel]
N_CLASSES: int = len(CLASSES)
torch.set_float32_matmul_precision("medium")

warnings.formatwarning = warning_format

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    warnings.warn("CUDA is not available. Running on CPU.")
    print(f"Using device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Seed: {SEED}")

# Load training data

needs_validation = False
if not os.path.exists(TRAIN_DATA_PATH) or len(os.listdir(TRAIN_DATA_PATH)) == 0:
    print("Directory is empty or does not exist. New testing data will be generated.")
    generate_data(
        n_samples=TRAIN_DATA_SIZE,
        n_points=N_POINTS,
        out_dir=TRAIN_DATA_PATH,
        jitter=0.02,
    )
    needs_validation = True
else:
    print("Data directory found. Using existing training data.")

file_paths = [os.path.join(TRAIN_DATA_PATH, f) for f in os.listdir(TRAIN_DATA_PATH)]
train_dataset = PointCloudDataset(
    file_paths=file_paths,
    n_points=N_POINTS,
    n_classes=N_CLASSES,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=TRAIN_DATA_BATCH_SIZE,
    shuffle=True,
    num_workers=TRAIN_DATA_WORKERS,
    persistent_workers=True,
)

if needs_validation:
    train_dataset.validate()

print(f"Training dataset size: {len(train_dataset)}")

# Testing & validation data

needs_validation = False
if not os.path.exists(TEST_DATA_PATH) or len(os.listdir(TEST_DATA_PATH)) == 0:
    print(
        "Testing data directory is empty or does not exist. New testing data will be downloaded."
    )
    download_data(out_dir=TEST_DATA_PATH, n_points=N_POINTS)
    needs_validation = True
else:
    print("Testing data directory found. Using existing testing data.")

real_data = PointCloudDataset(
    file_paths=[os.path.join(TEST_DATA_PATH, f) for f in os.listdir(TEST_DATA_PATH)],
    n_points=N_POINTS,
    n_classes=N_CLASSES,
)

if needs_validation:
    real_data.validate()

# Split the real data into test and validation datasets
valid_size = int(VALID_DATA_SPLIT * len(real_data))
test_dataset, valid_dataset = torch.utils.data.random_split(
    real_data, [len(real_data) - valid_size, valid_size]
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=TEST_DATA_BATCH_SIZE,
    shuffle=False,
    num_workers=TEST_DATA_WORKERS,
    persistent_workers=True,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=VALID_DATA_BATCH_SIZE,
    shuffle=False,
    num_workers=VALID_DATA_WORKERS,
    persistent_workers=True,
)

del real_data

print(f"Testing dataset size: {len(test_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")

# Trainiing

data_path = "data/pointnet"
model_fname = "bestmodel"
log_fname = "log.csv"

segmenter = PointNetSeg(n_classes=N_CLASSES)

model_path = os.path.join(data_path, model_fname)
log_path = os.path.join(data_path, log_fname)

try:
    segmenter.load_state_dict(torch.load(f"{model_path}.pth"))
    print("Model loaded")
except FileNotFoundError:
    print("Training new model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer = L.Trainer(
        # fast_dev_run=True,
        max_epochs=15,
        default_root_dir=data_path,
    )
    trainer.fit(
        model=segmenter,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    trainer.test(model=segmenter, dataloaders=test_dataloader)

# Testing

trainer = L.Trainer(default_root_dir=data_path)
trainer.test(model=segmenter, dataloaders=test_dataloader)
