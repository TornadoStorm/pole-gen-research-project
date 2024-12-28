from typing import List, Tuple

import numpy as np
import open3d as o3d

from .models import State, UtilityPoleLabel


def scan_geometry(state: State):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_to_label: dict[int, UtilityPoleLabel] = {}
    for mesh, label in state.geometry.items():
        mesh_id = scene.add_triangles(mesh.vertices, mesh.triangles)
        mesh_to_label[mesh_id] = label

    # Simulate LiDAR sensor
    min_points = 800
    max_points = 103000

    # Sample random points on the geometry
    points = scene.sample_points_uniformly(min_points, max_points)
