import os

import numpy as np
import open3d as o3d
from tqdm.auto import tqdm


# Little tool to sample pcd files with a specified number of points and save them over to a new directory.
def sample_pcd_files(src_dir: str, out_dir: str, n_points: int):
    files = os.listdir(src_dir)
    files = [f for f in files if f.endswith(".pcd")]

    os.makedirs(out_dir, exist_ok=True)

    for f in tqdm(files, desc="Sampling pcd files"):
        pc = o3d.t.io.read_point_cloud(os.path.join(src_dir, f))
        n = len(pc.point.positions)
        indices = np.random.choice(n, n_points, replace=n_points > n)
        pc.point.positions = pc.point.positions[indices]
        pc.point.labels = pc.point.labels[indices]
        o3d.t.io.write_point_cloud(os.path.join(out_dir, f), pc)
