import random

import numpy as np
import open3d as o3d

from pole_gen.components.lamp import add_lamp
from pole_gen.components.pole import add_pole
from pole_gen.components.traffic_lights import add_traffic_lights
from pole_gen.components.transformer import add_transformer
from pole_gen.models.state import State
from utils.mesh_tools import create_quad, normalize_mesh


# TODO DELETE ME WHEN DONE
def _add_road_meshes(mesh: o3d.geometry.TriangleMesh, state: State):
    road_meshes = []
    if state.road_presence[0] != 0:
        road_meshes.append(
            create_quad(
                (state.road_presence[0], 0, 0),
                1 if state.main_road == 0 else 0.5,
                4 if state.main_road == 0 else 2,
            )
        )
    if state.road_presence[1] != 0:
        road_meshes.append(
            create_quad(
                (0, state.road_presence[1], 0),
                4 if state.main_road == 1 else 2,
                1 if state.main_road == 1 else 0.5,
            )
        )

    for road_mesh in road_meshes:
        mesh += road_mesh


def generate_utility_pole():
    mesh = o3d.geometry.TriangleMesh()

    state = State()

    # Simulate having a road at a given side (1, -1) or not (0)
    # Index 0 = x axis, index 1 = z axis
    state.road_presence = [random.choice([1, 0, -1]), random.choice([1, 0, -1])]
    if all(road == 0 for road in state.road_presence):
        state.road_presence[random.choice([0, 1])] = random.choice([1, -1])
    # Pick one of the roads to be the "main" road
    state.main_road = random.choice(
        [i for i, v in enumerate(state.road_presence) if v != 0]
    )

    # Other shared variables
    state.intersection = len([road for road in state.road_presence if road != 0]) > 1
    state.pole_base_height = 8.45
    state.pole_scale = random.uniform(1.0, 1.955)
    state.pole_scaled_height = state.pole_base_height * state.pole_scale

    # Useful for rotating things to align with roads
    state.rot_indices = [0, 0]
    match state.road_presence[0]:
        case 1:
            state.rot_indices[0] = 0
        case -1:
            state.rot_indices[0] = 2
    match state.road_presence[1]:
        case 1:
            state.rot_indices[1] = 1
        case -1:
            state.rot_indices[1] = 3

    _add_road_meshes(mesh, state)
    add_pole(mesh, state)
    add_traffic_lights(mesh, state)
    add_lamp(mesh, state)
    add_transformer(mesh, state)

    # Scale and rotate everything a bit to give more variation
    mesh.scale(random.uniform(0.9, 1.1), center=(0, 0, 0))
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz(
            (
                np.deg2rad(random.uniform(-3, 3)),
                np.deg2rad(random.uniform(-3, 3)),
                random.uniform(0, 2 * np.pi),
            )
        )
    )

    # Finish up and return the mesh
    normalize_mesh(mesh)
    return mesh
