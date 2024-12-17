import json
import os
import tempfile
import zipfile
from typing import List

import open3d as o3d
from tqdm.auto import tqdm

import utils.file
from models.data_source import DataSource
from models.data_source_info import DataSourceInfo, DataSourceInfoEncoder
from models.dataset import PointCloudDataset


# TODO: Augmentations for training data
class ModelNet40(DataSource):
    URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

    INFO = DataSourceInfo(
        class_names=[
            "airplane",
            "bathtub",
            "bed",
            "bench",
            "bookshelf",
            "bottle",
            "bowl",
            "car",
            "chair",
            "cone",
            "cup",
            "curtain",
            "desk",
            "door",
            "dresser",
            "flower_pot",
            "glass_box",
            "guitar",
            "keyboard",
            "lamp",
            "laptop",
            "mantel",
            "monitor",
            "night_stand",
            "person",
            "piano",
            "plant",
            "radio",
            "range_hood",
            "sink",
            "sofa",
            "stairs",
            "stool",
            "table",
            "tent",
            "toilet",
            "tv_stand",
            "vase",
            "wardrobe",
            "xbox",
        ]
    )

    @classmethod
    def download(
        cls,
        npoints: int = 2500,
        train_outdir: str = "data/train",
        test_outdir: str = "data/test",
        info_outdir: str = "data",
    ):
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, "ModelNet40.zip")
        extract_dir = os.path.join(temp_dir, "ModelNet40")

        # Check if extracted folder exists and is valid
        print("Checking for existing source data...")
        if not os.path.exists(extract_dir):
            print("Extracted folder does not exist.")

            # Download and extract zip (if needed)
            if os.path.exists(zip_path):
                print("Zip already downloaded.")
            else:
                utils.file.download_file(cls.URL, zip_path)
                print("Zip downloaded.")

            print("Extracting zip...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            file_count = 0
            for _, _, filenames in os.walk(extract_dir):
                file_count += len(filenames)

            # Check all .off files for errors
            with tqdm(desc="Checking OFF files", total=file_count) as pbar:
                for dirpath, _, filenames in os.walk(extract_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        utils.file.check_off_file(fp)
                        pbar.update(1)
        else:
            file_count = 0
            for _, _, filenames in os.walk(extract_dir):
                file_count += len(filenames)

        if not os.path.exists(train_outdir):
            os.makedirs(train_outdir)

        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)

        train_files: List[str] = []
        test_files: List[str] = []

        classes: List[str] = []

        # Find root folder
        root_folder = extract_dir
        listdir = ""
        while True:
            listdir = os.listdir(root_folder)
            if len(listdir) != 0 and listdir[0].startswith("ModelNet"):
                root_folder = os.path.join(root_folder, listdir[0])
            else:
                break

        with tqdm(desc="Processing data", total=file_count) as pbar:
            for folder in sorted(listdir):
                label = len(classes)
                classes.append(folder)

                for split, file_list, out_dir in [
                    ("train", train_files, train_outdir),
                    ("test", test_files, test_outdir),
                ]:
                    file_no = 0
                    split_folder = os.path.join(root_folder, folder, split)
                    for file in os.listdir(split_folder):
                        file_path = os.path.join(split_folder, file)
                        # Annoying but we need to do this to fix some OFF files
                        mesh = o3d.io.read_triangle_mesh(file_path)
                        pcd = mesh.sample_points_uniformly(number_of_points=npoints)
                        pcd_center = pcd.get_center()
                        pcd.translate(-pcd_center)  # Center the point cloud
                        max_distance = max(pcd.get_max_bound() - pcd.get_min_bound())
                        pcd.scale(
                            1 / max_distance, center=(0, 0, 0)
                        )  # Scale to unit sphere
                        out_path = os.path.join(
                            out_dir, f"{label:02d}_{file_no:05d}.ply"
                        )
                        o3d.io.write_point_cloud(out_path, pcd)
                        file_no += 1
                        file_list.append(out_path)
                        pbar.update(1)

        with open(os.path.join(info_outdir, "data_info.json"), "w") as f:
            json.dump(cls.INFO, f, cls=DataSourceInfoEncoder)

        return cls.INFO, PointCloudDataset(train_files), PointCloudDataset(test_files)
