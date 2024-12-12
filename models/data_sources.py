import os
import shutil
import tempfile
import zipfile
from typing import List, Tuple

import numpy as np
import torch.utils.data as data
from m2p.convert import meshToPointCloud

import utils.file
from models.dataset import PointCloudDataset


class DataSource:
    url: str
    name: str

    def download(self) -> Tuple[data.Dataset, data.Dataset]:
        pass


# TODO: Augmentations for training data
class ModelNet40(DataSource):
    url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    name = "ModelNet40"
    checksum = "42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac"

    @classmethod
    def download(
        cls,
        npoints: int = 2500,
        train_outdir: str = "data/train",
        test_outdir: str = "data/test",
    ) -> Tuple[data.Dataset, data.Dataset]:
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, cls.name + ".zip")
        dataset_path = os.path.join(temp_dir, cls.name)

        if os.path.exists(zip_path) and utils.file.checksum(zip_path) == cls.checksum:
            print("Zip already downloaded.")
        else:
            utils.file.download_file(cls.url, zip_path)
            print("Zip downloaded.")

        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

        print("Extracting data...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
            print("Data extracted.")

        dataset_root = dataset_path
        while True:
            listdir = os.listdir(dataset_root)
            if len(listdir) > 0 and listdir[0].startswith("ModelNet"):
                dataset_root = os.path.join(dataset_root, listdir[0])
            else:
                break

        print("Creating dataset...")

        classes = os.listdir(dataset_root)

        if not os.path.exists(train_outdir):
            os.makedirs(train_outdir)

        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)

        train_files: List[str] = []
        test_files: List[str] = []

        for i in range(len(classes)):
            class_name = classes[i]
            for outdir, split, files in [
                (train_outdir, "train", train_files),
                (test_outdir, "test", test_files),
            ]:
                oudir_cls = os.path.join(outdir, class_name)
                if not os.path.exists(oudir_cls):
                    os.makedirs(oudir_cls)

                for fn in os.listdir(os.path.join(dataset_root, class_name, split)):
                    from_path = os.path.join(dataset_root, class_name, split, fn)
                    utils.file.check_off_file(from_path)

                    point_set = utils.file.mesh_to_points(
                        file_path=from_path, npoints=npoints
                    )
                    point_set = point_set - np.expand_dims(
                        np.mean(point_set, axis=0), 0
                    )  # center
                    dist = np.max(np.sqrt(np.sum(point_set**2, axis=1)), 0)
                    point_set = point_set / dist  # scale

                    to_path = os.path.join(oudir_cls, f"{os.path.splitext(fn)[0]}.bin")

                    utils.file.save_points(file_path=to_path, points=point_set, label=i)

                    files.append((to_path, i))

        print("Cleaning up...")

        shutil.rmtree(dataset_path)

        print("Done.")

        return classes, PointCloudDataset(train_files), PointCloudDataset(test_files)
