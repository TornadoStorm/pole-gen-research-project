from __future__ import print_function

from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data


class PointCloudDataset(data.Dataset):
    """Dataset for point cloud data."""

    file_paths: List[str]
    """List of file paths to the source files."""

    def __init__(self, file_paths: List[str] = []) -> None:
        self.file_paths = file_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[index]
        pc = o3d.t.io.read_point_cloud(file_path)

        points = torch.from_numpy(pc.point.positions.numpy().astype(np.float32))
        points -= points.mean(dim=0)
        points /= points.abs().max()

        labels = torch.from_numpy(pc.point.labels.numpy().astype(np.uint8))

        return points, labels

    def __len__(self) -> int:
        return len(self.file_paths)
