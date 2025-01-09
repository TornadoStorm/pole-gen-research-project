import os

import numpy as np
import open3d as o3d
from segments.utils import load_pointcloud_from_url
from tqdm.auto import tqdm

from utils.config import SEGMENTS_CLIENT as sg_client

DATASET_ID = "TeaStorm/Electrical_poles"
LABELSET = "ground-truth"


def _pc_data_filename(n: int) -> str:
    return f"{n:06d}.pcd"


def download_data(
    n_points: int,
    out_dir: str,
    clear_dir: bool = True,
):
    # Delete directory if it already exists
    if clear_dir and os.path.exists(out_dir):
        for file in tqdm(os.listdir(out_dir), desc="Clearing output directory"):
            os.remove(os.path.join(out_dir, file))

    if not os.path.exists(out_dir):
        print("Creating output directory")
        os.makedirs(out_dir)

    sample_uuids = list(
        map(
            lambda x: x.uuid,
            sg_client.get_samples(
                DATASET_ID,
                labelset=LABELSET,
            ),
        )
    )

    sample_id = 0  # For filename

    for uuid in tqdm(sample_uuids, desc="Downloading data"):
        sample = sg_client.get_sample(uuid, labelset=LABELSET)

        # Extract points from file
        pc: o3d.geometry.PointCloud = load_pointcloud_from_url(
            sample.attributes.pcd.url
        )

        # Sample points and labels
        indices = np.random.choice(
            len(pc.points),
            n_points,
            replace=len(pc.points) < n_points,
        )

        points = np.asarray(pc.points)[indices]
        labels = np.asarray(sample.label.attributes.point_annotations)[indices]

        out_pc = o3d.t.geometry.PointCloud()
        out_pc.point.positions = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
        out_pc.point.labels = o3d.core.Tensor(
            np.asarray(labels, dtype=np.uint32).reshape(-1, 1)
        )

        out_path: str
        while True:
            out_path = os.path.join(out_dir, _pc_data_filename(sample_id))
            if not os.path.exists(out_path):
                break
            sample_id += 1
        o3d.t.io.write_point_cloud(out_path, out_pc)
