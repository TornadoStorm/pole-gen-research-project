from __future__ import print_function

import os.path
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
from segments.utils import load_pointcloud_from_url

from segments import SegmentsDataset
from utils.config import SEGMENTS_CLIENT as sg_client


class PointCloudDataset(data.Dataset):
    """Dataset for point cloud data."""

    file_paths: List[str]
    """List of file paths to the source files."""

    def __init__(self, file_paths: List[str] = []) -> None:
        self.file_paths = file_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract points from file
        file_path = self.file_paths[index]
        pc = o3d.io.read_point_cloud(file_path)
        point_set_tensor = torch.from_numpy(np.asarray(pc.points).astype(np.float32))

        # Center and normalize points
        point_set_tensor -= point_set_tensor.mean(dim=0)
        point_set_tensor /= point_set_tensor.abs().max()

        # Extract label from filename
        filename = os.path.basename(file_path)
        label_str = filename.split("_")[0]
        label = int(label_str)
        label_tensor = torch.from_numpy(np.array([label]).astype(np.int64))

        return point_set_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.file_paths)


class ElectricalPolesDataset(data.Dataset):
    DATASET_ID = "TeaStorm/Electrical_poles"
    LABELSET = "ground-truth"

    sample_uuids: List[str]
    npoints: int

    def __init__(self, npoints: int = 1024) -> None:
        super().__init__()
        self.npoints = npoints
        self.sample_uuids = list(
            map(
                lambda x: x.uuid,
                sg_client.get_samples(
                    self.DATASET_ID,
                    labelset=self.LABELSET,
                ),
            )
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = sg_client.get_sample(self.sample_uuids[index], labelset=self.LABELSET)

        # Extract points from file
        pcd: o3d.geometry.PointCloud = load_pointcloud_from_url(
            sample.attributes.pcd.url
        )

        # Sample points and labels
        indices = np.random.choice(
            len(pcd.points),
            self.npoints,
            replace=False,
        )

        points = np.asarray(pcd.points)[indices]
        points -= points.mean(axis=0)  # Center
        points /= np.abs(points).max()  # Normalize

        labels = np.asarray(sample.label.attributes.point_annotations)[indices]

        points_tensor = torch.from_numpy(np.asarray(points).astype(np.float32))
        label_tensor = torch.from_numpy(np.asarray(labels).astype(np.int64))

        return points_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.sample_uuids)
