from __future__ import print_function

import os.path
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
from utils.file import read_points

class PointCloudDataset(data.Dataset):
    """Dataset for point cloud data."""

    file_paths: List[str]
    """List of file paths to the source files."""

    def __init__(self, file_paths: List[str] = []) -> None:
        self.file_paths = file_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[index]
        label, num_points, points = read_points(file_path)
        point_set_tensor = torch.from_numpy(np.asarray(points).astype(np.float32))
        label_tensor = torch.from_numpy(np.array([label]).astype(np.int64))
        return point_set_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.file_paths)
