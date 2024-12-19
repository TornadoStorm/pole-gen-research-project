from typing import Tuple

import numpy as np
import open3d as o3d


def create_quad(
    center: Tuple[float, float, float], x_scale: float, y_scale: float
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [center[0] - x_scale / 2, center[1] - y_scale / 2, center[2]],
            [center[0] + x_scale / 2, center[1] - y_scale / 2, center[2]],
            [center[0] + x_scale / 2, center[1] + y_scale / 2, center[2]],
            [center[0] - x_scale / 2, center[1] + y_scale / 2, center[2]],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    return mesh


def normalize_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    scale = max_bounds - min_bounds
    vertices -= center  # Center
    vertices /= scale.max()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)  # Normalize
