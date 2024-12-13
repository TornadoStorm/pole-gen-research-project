import cProfile
import os
import tempfile
import zipfile
from typing import List, Tuple

import numpy as np
import torch.utils.data as data
from m2p.convert import meshToPointCloud
from tqdm import tqdm

import utils.file
from models.dataset import PointCloudDataset


class DataSource:
    def download(self) -> Tuple[data.Dataset, data.Dataset]:
        pass


# TODO: Augmentations for training data
class ModelNet40(DataSource):
    URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    CHECKSUM = "42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac"

    @classmethod
    def download(
        cls,
        npoints: int = 2500,
        train_outdir: str = "data/train",
        test_outdir: str = "data/test",
    ) -> Tuple[data.Dataset, data.Dataset]:
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, "ModelNet40.zip")

        if os.path.exists(zip_path) and utils.file.checksum(zip_path) == cls.CHECKSUM:
            print("Zip already downloaded.")
        else:
            utils.file.download_file(cls.URL, zip_path)
            print("Zip downloaded.")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print("Creating dataset...")

            if not os.path.exists(train_outdir):
                os.makedirs(train_outdir)

            if not os.path.exists(test_outdir):
                os.makedirs(test_outdir)

            train_files: List[str] = []
            test_files: List[str] = []

            classes: List[str] = []
            class_index_map: dict[str, int] = {}

            namelist = zip_ref.namelist()
            i = 0

            data_count = 0

            # Collect classes
            while True:
                name = namelist[i]
                names = name.split("/")

                if len(names) <= 2:
                    i += 1
                    continue
                elif names[-1] == "":
                    # Directory
                    last_folder = names[-2]
                    if last_folder == "train" or last_folder == "test":
                        i += 1
                        continue
                    # Class name
                    label = last_folder
                    class_index_map[label] = len(classes)
                    classes.append(label)
                    i += 1
                    continue
                else:
                    i += 1
                    data_count = len(namelist) - i
                    print(f"Found {len(classes)} classes.")

                    pr = cProfile.Profile()
                    pr.enable()

                    with tqdm(desc="Processing data", total=data_count) as pbar:
                        for j in range(data_count):
                            name = namelist[i + j]
                            names = name.split("/")
                            split = names[-2]
                            label = names[-3]

                            with zip_ref.open(name) as file:
                                verts, faces = utils.file.read_off(file)
                                # points = meshToPointCloud(verts, faces, npoints)
                                indices = np.random.choice(
                                    len(verts), npoints, replace=True
                                )
                                points = np.array(verts)[indices]
                                points = points - np.expand_dims(
                                    np.mean(points, axis=0), 0
                                )  # center
                                dist = np.max(np.sqrt(np.sum(points**2, axis=1)), 0)
                                points = points / dist  # scale

                                if split == "train":
                                    target_outdir = train_outdir
                                    target_list = train_files
                                else:
                                    target_outdir = test_outdir
                                    target_list = test_files

                                o_path = os.path.join(
                                    target_outdir,
                                    f"{os.path.splitext(os.path.basename(name))[0]}.bin",
                                )

                                utils.file.save_points(
                                    o_path, points, class_index_map[label]
                                )
                                target_list.append(o_path)

                            pbar.update(1)
                            if pbar.n == 123:
                                pr.disable()
                                pr.dump_stats("profile.prof")
                                print("Profiling complete.")
                    break

        return classes, PointCloudDataset(train_files), PointCloudDataset(test_files)
