from __future__ import print_function

import os
import os.path
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data

# PCArray = np.ndarray[Any, np.dtype[np.float32]]


# class PCAugmentation:

#     def augment(self, point_set: PCArray, seed: Any) -> None:
#         raise NotImplementedError


# class RotateAugment(PCAugmentation):

#     def augment(self, point_set: PCArray, seed: Any) -> None:
#         np.random.seed(seed)
#         theta = np.random.uniform(0, np.pi * 2)
#         rotation_matrix = np.array(
#             [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
#         )
#         point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)


# class JitterAugment(PCAugmentation):
#     amount: float = 0.02
#     """_summary_: Amount of jitter to apply to the point cloud. Default is 0.02."""

#     def augment(self, point_set: PCArray, seed: Any) -> None:
#         np.random.seed(seed)
#         point_set += np.random.normal(0, self.amount, size=point_set.shape)


def check_off_file(file_path: str) -> None:
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        if first_line == "OFF":
            return

        remaining_lines = f.readlines()

    with open(file_path, "w") as f:
        if first_line.startswith("OFF") and len(first_line) > 3:
            print("Fixing OFF file format for", file_path)
            first_line_remainder = first_line[3:].strip()
            f.writelines(["OFF\n", f"{first_line_remainder}\n"] + remaining_lines)


# TODO: Augmentations (only for training data)
class ModelNetDataset(data.Dataset):
    npoints: int
    """Number of points to sample from the point cloud."""
    split: str
    """Split of the dataset to use. Can be either 'train' or 'test'."""
    files: List[Tuple[str, int]]
    """List of file paths to the source files paired with their class index."""
    classes: List[str]
    """List of class names."""

    seed = 69420

    def __init__(
        self,
        root,
        npoints=2500,
        split="train",
    ):

        # Find real root folder
        root_folder = root
        while True:
            listdir = os.listdir(root_folder)
            if len(listdir) > 0 and listdir[0].startswith("ModelNet"):
                root_folder = os.path.join(root_folder, listdir[0])
            else:
                break

        self.npoints = npoints
        self.split = split

        self.files = []
        self.classes = []

        self.classes = os.listdir(root_folder)

        for i in range(len(self.classes)):
            cl = self.classes[i]
            for fn in os.listdir(os.path.join(root_folder, cl, self.split)):
                file_path = os.path.join(root_folder, cl, self.split, fn)
                self.files.append((file_path, i))
                check_off_file(file_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, cls = self.files[index]
        mesh = o3d.io.read_triangle_mesh(file_path)

        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty mesh found at {file_path}")

        pts = np.asarray(mesh.vertices)
        np.random.seed(self.seed + index)
        # TODO: Figure out how to add more points
        choice = np.random.choice(
            len(pts), self.npoints, replace=(self.npoints > len(pts))
        )
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
        point_set = point_set / dist  # scale

        point_set_tensor = torch.from_numpy(point_set.astype(np.float32))
        cls_tensor = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set_tensor, cls_tensor

    def __len__(self) -> int:
        return len(self.files)
