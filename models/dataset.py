from __future__ import print_function

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
from tqdm.auto import tqdm


class PointCloudDataset(data.Dataset):
    """Dataset for point cloud data."""

    file_paths: List[str]
    """List of file paths to the source files."""

    n_points: int | None

    def __init__(self, file_paths: List[str], n_points=None) -> None:
        self.file_paths = file_paths
        self.n_points = n_points

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[index]
        pc = o3d.t.io.read_point_cloud(file_path)
        n = len(pc.point.positions)

        if self.n_points is not None:
            indices = np.random.choice(
                n,
                self.n_points,
                replace=n < self.n_points,
            )
        else:
            indices = np.arange(n)

        points = torch.from_numpy(
            pc.point.positions.numpy()[indices].astype(np.float32)
        )
        points -= points.mean(dim=0)
        points /= points.abs().max()

        labels = torch.from_numpy(pc.point.labels.numpy()[indices].astype(np.uint8))

        return points, labels

    def __len__(self) -> int:
        return len(self.file_paths)
