from typing import Tuple

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
