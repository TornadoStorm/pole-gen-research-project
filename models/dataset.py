from __future__ import print_function

import warnings
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
from tqdm.auto import trange


class PointCloudDataset(data.Dataset):
    """Dataset for point cloud data."""

    file_paths: List[str]
    """List of file paths to the source files."""

    n_points: int | None
    n_classes: int

    def __init__(self, file_paths: List[str], n_points: int, n_classes: int) -> None:
        self.file_paths = file_paths
        self.n_points = n_points
        self.n_classes = n_classes

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

        labels = torch.from_numpy(
            pc.point.labels.numpy()[indices].squeeze().astype(np.int64)
        )

        return points, labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def validate(self) -> bool:
        for i in trange(len(self), desc="Checking dataset..."):
            sample = self[i]
            n = len(sample[0].numpy())
            if n != self.n_points:
                warnings.warn(
                    f"A sample has {n} points (Expected {self.n_points}). Check if the data was generated correctly!"
                )
                return False
            labels = sample[1].numpy()
            for label in labels:
                if label < 0 or label >= self.n_classes:
                    warnings.warn(
                        f"Invalid label {label} found in the dataset (File: {self.file_paths[i]}). Check if the data was generated correctly!"
                    )
                    return False

            # Check for NaN values
            if torch.any(torch.isnan(sample[0])):
                warnings.warn(
                    f"NaN values found in the dataset (File: {self.file_paths[i]}). Check if the data was generated correctly!\nValues: {sample[0]}"
                )
                return False
        return True
