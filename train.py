import os
import random
import warnings

import numpy as np
import pytorch_lightning as L
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from ai.pointnet_seg.model import PointNetSeg
from models.dataset import PointCloudDataset
from pole_gen.data import generate_data
from utils.config import *
from utils.logging import warning_format
from utils.sample_pcd import sample_pcd_files

torch.set_float32_matmul_precision("medium")

# Odd workaround to fix the cuda not initialized error
torch.cuda.empty_cache()
s = 32
dev = torch.device("cuda")
torch.nn.functional.conv2d(
    torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
)

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
    sample_pcd_files("gt_data", TEST_DATA_PATH, N_POINTS)
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

# Training

segmenter = PointNetSeg(n_classes=N_CLASSES)
trainer = L.Trainer(
    max_epochs=15,
    precision="16-mixed",
    accumulate_grad_batches=2,
    logger=MLFlowLogger(
        experiment_name="PointNetSeg_Synthetic",
        log_model=True,
    ),
)
trainer.fit(
    model=segmenter,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)

# Testing

test_trainer = L.Trainer(devices=1)
test_trainer.test(model=segmenter, dataloaders=test_dataloader)
