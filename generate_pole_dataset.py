import argparse
import os

from torch.utils.data import DataLoader

from models.dataset import PointCloudDataset
from pole_gen.data import generate_data
from utils.config import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate pole dataset.")
parser.add_argument(
    "-p",
    "--npoints",
    type=int,
    required=False,
    help="Number of points in each point cloud",
    default=N_POINTS,
)
parser.add_argument(
    "-n",
    "--size",
    type=int,
    required=False,
    help="Number of samples to generate",
    default=TRAIN_DATA_SIZE,
)
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    required=False,
    help="Directory to save the generated data",
    default=TRAIN_DATA_PATH,
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    required=False,
    help="Random seed for data generation",
    default=SEED,
)
args = parser.parse_args()

N_POINTS = args.npoints
TRAIN_DATA_SIZE = args.size
TRAIN_DATA_PATH = args.dir
SEED = args.seed

if not os.path.exists(TRAIN_DATA_PATH) or len(os.listdir(TRAIN_DATA_PATH)) == 0:
    print("Directory is empty or does not exist. New testing data will be generated.")
    print(f"Generating {TRAIN_DATA_SIZE} samples with {N_POINTS} points each.")
    generate_data(
        n_samples=TRAIN_DATA_SIZE,
        n_points=N_POINTS,
        out_dir=TRAIN_DATA_PATH,
        jitter=TRAIN_DATA_JITTER,
    )
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

print(f"Training dataset size: {len(train_dataset)}")
