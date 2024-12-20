import random

import numpy as np
import open3d as o3d

from pole_gen.models.state import State


def load_double_crossbar(variant: int) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(f"pole_gen/meshes/crossbar_double_{variant}.ply")
