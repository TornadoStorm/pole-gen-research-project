import hashlib
import os
import struct
from typing import IO, List, Tuple

import numpy as np
import open3d as o3d
import requests
from tqdm import tqdm


def checksum(file_path: str, algorithm="sha256") -> str:
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte

    with open(output_path, "wb") as file, tqdm(
        desc=output_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)


def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


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


def save_points(
    file_path: str, points: List[Tuple[float, float, float]], label: int
) -> None:
    with open(file_path, "wb") as f:
        f.write(struct.pack("i", label))
        f.write(struct.pack("i", len(points)))
        for point in points:
            f.write(struct.pack("fff", *point))


def read_points(file_path: str) -> Tuple[int, int, List[Tuple[float, float, float]]]:
    with open(file_path, "rb") as f:
        label = struct.unpack("i", f.read(4))
        num_points = struct.unpack("i", f.read(4))
        points = []
        for i in range(num_points):
            point = struct.unpack("fff", f.read(12))
            points.append(point)
        return label, num_points, points


def read_points_header(file_path: str) -> Tuple[int, int]:
    with open(file_path, "rb") as f:
        label = struct.unpack("i", f.read(4))
        num_points = struct.unpack("i", f.read(4))
        return label, num_points


def mesh_to_points(file_path: str, npoints: int) -> List[Tuple[float, float, float]]:
    pc = o3d.io.read_triangle_mesh(file_path).sample_points_poisson_disk(
        number_of_points=npoints, init_factor=5
    )
    return np.asarray(pc.points)


def read_off(
    file: IO[bytes],
) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []

    # Find the first line that is not "OFF"
    first_line = ""
    while True:
        first_line = file.readline().strip().decode("utf-8")
        if first_line != "OFF" and first_line != "":
            break

    # Odd bug with ModelNet40 dataset where they put the OFF on the same line as the 2nd line
    if first_line.startswith("OFF") and len(first_line) > 3:
        first_line = first_line[3:]

    n_verts, n_faces, n_edges = map(int, first_line.split())

    lines = [line.strip().decode("utf-8") for line in file.readlines()]

    # Read the vertices
    vertices = [tuple(map(float, lines[i].split())) for i in range(n_verts)]

    # Read the faces
    faces = [
        list(map(int, lines[i].split()[1:])) for i in range(n_verts, n_verts + n_faces)
    ]

    return vertices, faces
