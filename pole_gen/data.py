import os

import numpy as np
import open3d as o3d
from tqdm.auto import tqdm, trange

from pole_gen.pole_gen import generate_utility_pole_mesh
from pole_gen.raycasting import scan_geometry


def _pc_data_filename(n: int) -> str:
    return f"{n:06d}.pcd"


def generate_data(
    n_samples: int,
    n_points: int,
    out_dir: str,
    min_scanner_distance: int = 3,
    max_scanner_distance: int = 20,
    clear_dir: bool = True,
    jitter: float = 0.0,
):
    # Delete directory if it already exists
    if clear_dir and os.path.exists(out_dir):
        for file in tqdm(os.listdir(out_dir), desc="Clearing output directory"):
            os.remove(os.path.join(out_dir, file))

    os.makedirs(out_dir, exist_ok=True)

    sample_id = 0  # For filename

    for _ in trange(n_samples, desc="Generating training data"):
        state = generate_utility_pole_mesh()
        r = np.random.uniform(min_scanner_distance, max_scanner_distance)
        a = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(a)
        y = r * np.sin(a)
        pc = scan_geometry(
            state, n_points=n_points, sensor_pos=(x, y, 1), jitter=jitter
        )
        out_path: str
        while True:
            out_path = os.path.join(out_dir, _pc_data_filename(sample_id))
            if not os.path.exists(out_path):
                break
            sample_id += 1
        o3d.t.io.write_point_cloud(out_path, pc)
