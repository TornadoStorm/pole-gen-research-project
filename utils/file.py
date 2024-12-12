import hashlib
import os

import requests
from tqdm import tqdm


def checksum(file_path, algorithm="sha256"):
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_download_size(url):
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


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
