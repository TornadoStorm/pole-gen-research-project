import random

import numpy as np
import open3d as o3d

from pole_gen.models.state import State


def add_lamp(mesh: o3d.geometry.TriangleMesh, state: State):
    if random.random() <= 0.5:
        lamp_mesh = o3d.io.read_triangle_mesh("pole_gen/meshes/lamp.ply")
        # Rotate to (main) road
        lamp_mesh.rotate(
            lamp_mesh.get_rotation_matrix_from_xyz(
                (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
            ),
            center=(0, 0, 0),
        )
        # Randomize rotation
        lamp_mesh.rotate(
            lamp_mesh.get_rotation_matrix_from_xyz(
                (0, np.deg2rad(random.uniform(-5.0, 0.0)), 0)
            ),
            center=(0, 0, 0),
        )
        # Randomize position
        lamp_height = max(
            state.traffic_light_heights[state.main_road]
            + 1,  # Always above traffic light
            min(7.5 + random.uniform(-1.0, 1.94), state.pole_scaled_height - 0.3),
        )
        lamp_mesh.translate([0, 0, lamp_height])
        state.lamp_height = lamp_height
        mesh += lamp_mesh
