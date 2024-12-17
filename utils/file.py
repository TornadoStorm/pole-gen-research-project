import os
from typing import IO, List, Tuple

import requests
from tqdm.auto import tqdm


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
            # print("Fixing OFF file:", file_path)
            first_line_remainder = first_line[3:].strip()
            f.writelines(["OFF\n", f"{first_line_remainder}\n"] + remaining_lines)


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
