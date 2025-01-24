# This code was used to pull data from the Segments.ai API and save it to disk.
# It is no longer needed as the sample data is now stored in this repository (./gt_data).
# I left it here for reference.

import os

import numpy as np
import open3d as o3d
from dotenv import load_dotenv
from segments import SegmentsClient
from segments.utils import load_pointcloud_from_url
from tqdm.auto import tqdm

load_dotenv()

SEGMENTS_CLIENT = SegmentsClient(os.getenv("SEGMENTS_AI_API_KEY"))
DATASET_ID = "TeaStorm/Electrical_poles"
LABELSET = "ground-truth"


def _pc_data_filename(n: int) -> str:
    return f"{n:06d}.pcd"


def download_data(
    out_dir: str,
    clear_dir: bool = True,
    include_unlabeled: bool = False,
):
    # Delete directory if it already exists
    if clear_dir and os.path.exists(out_dir):
        for file in tqdm(os.listdir(out_dir), desc="Clearing output directory"):
            os.remove(os.path.join(out_dir, file))

    os.makedirs(out_dir, exist_ok=True)

    sample_uuids = list(
        map(
            lambda x: x.uuid,
            SEGMENTS_CLIENT.get_samples(
                DATASET_ID,
                labelset=LABELSET,
            ),
        )
    )

    sample_id = 0  # For filename

    for uuid in tqdm(sample_uuids, desc="Downloading data"):
        sample = SEGMENTS_CLIENT.get_sample(uuid, labelset=LABELSET)

        # Extract points from file
        pc: o3d.geometry.PointCloud = load_pointcloud_from_url(
            sample.attributes.pcd.url
        )

        # Sample points and labels
        points = np.asarray(pc.points)

        label_map: dict[int, int] = {}
        label_map[0] = 0
        for a in sample.label.attributes.annotations:
            label_map[a.id] = a.category_id

        labels = np.asarray(
            [label_map[i] for i in sample.label.attributes.point_annotations],
            dtype=np.uint32,
        )

        # Remove points with label 0 if include_unlabeled is False
        if not include_unlabeled:
            mask = labels != 0
            points = points[mask]
            labels = labels[mask]

        out_pc = o3d.t.geometry.PointCloud()
        out_pc.point.positions = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
        out_pc.point.labels = o3d.core.Tensor(labels.reshape(-1, 1))

        out_path: str
        while True:
            out_path = os.path.join(out_dir, _pc_data_filename(sample_id))
            if not os.path.exists(out_path):
                break
            sample_id += 1
        o3d.t.io.write_point_cloud(out_path, out_pc)
