import numpy as np

from utils.mesh_tools import create_quad, normalize_mesh

from .components.crossbar import add_crossbar
from .components.lamp import add_lamp
from .components.pole import add_pole
from .components.signs import add_signs
from .components.traffic_lights import add_traffic_lights
from .components.transformer import add_transformer
from .models import State, UtilityPoleLabel


# TODO DELETE ME WHEN DONE
def _add_road_meshes(state: State):
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
        state.add_geometry(road_mesh, UtilityPoleLabel.UNLABELED)


def generate_utility_pole() -> State:
    state = State()

    # Simulate having a road at a given side (1, -1) or not (0)
    # Index 0 = x axis, index 1 = z axis
    state.road_presence = [np.random.choice([1, 0, -1]), np.random.choice([1, 0, -1])]
    if all(road == 0 for road in state.road_presence):
        state.road_presence[np.random.choice([0, 1])] = np.random.choice([1, -1])
    # Pick one of the roads to be the "main" road
    state.main_road = np.random.choice(
        [i for i, v in enumerate(state.road_presence) if v != 0]
    )

    # Other shared variables
    state.intersection = len([road for road in state.road_presence if road != 0]) > 1
    state.pole_base_height = 8.45
    state.pole_scale = np.random.uniform(1.0, 1.955)
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

    _add_road_meshes(state)
    add_pole(state)
    add_traffic_lights(state)
    add_lamp(state)
    add_transformer(state)
    add_crossbar(state)
    add_signs(state)

    # Scale and rotate everything a bit to give more variation
    # Also normalize the mesh to make sure it's centered and has a unit bounding box
    state.geometry.scale(np.random.uniform(0.9, 1.1), center=(0, 0, 0))
    state.geometry.rotate(
        state.geometry.get_rotation_matrix_from_xyz(
            (
                np.deg2rad(np.random.uniform(-3, 3)),
                np.deg2rad(np.random.uniform(-3, 3)),
                np.random.uniform(0, 2 * np.pi),
            )
        ),
        center=(0, 0, 0),
    )
    normalize_mesh(state.geometry)

    return state
