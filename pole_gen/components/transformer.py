import random

import numpy as np
import open3d as o3d

from pole_gen.components.crossbar import load_double_crossbar
from pole_gen.models.state import State

TRANSFORMER_SPAWN_CHANCE = 0.33
TRIPLE_TRANSFORMER_MIN_FREE_SPACE = 4.1


def add_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    # If we have enough space to add the crossbars for the triple transformer
    if (
        state.pole_scaled_height
        - max(np.max(state.traffic_light_heights), state.lamp_height)
        > TRIPLE_TRANSFORMER_MIN_FREE_SPACE
    ):
        _add_triple_transformer(mesh, state)
    return

    random_val = np.random.random()
    if random_val > TRANSFORMER_SPAWN_CHANCE:
        return

    if random_val <= TRANSFORMER_SPAWN_CHANCE / 3:
        _add_box_transformer(mesh, state)
    elif random_val <= 2 * TRANSFORMER_SPAWN_CHANCE / 3:
        _add_cylinder_transformer(mesh, state)
    else:
        # If we have enough space to add the crossbars for the triple transformer
        if (
            state.pole_scaled_height
            - max(np.max(state.traffic_light_heights), state.lamp_height)
            > TRIPLE_TRANSFORMER_MIN_FREE_SPACE
        ):
            _add_triple_transformer(mesh, state)
        else:
            _add_cylinder_transformer(mesh, state)


def _add_box_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    size = (0.417, 0.765, 0.573)
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0], height=size[1], depth=size[2]
    )
    box.translate(
        (
            random.uniform(0.45, 0.5) - (size[0] / 2),
            -(size[1] / 2),
            random.uniform(4.404, 4.73) - (size[2] / 2),
        )
    )
    box.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (
                0,
                0,
                np.deg2rad(
                    (90 * state.rot_indices[state.main_road]) + random.uniform(90, 270)
                ),
            )
        ),
        center=(0, 0, 0),
    )

    mesh += box
    return


def _create_cylinder_transformer(scale: bool = True):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.318, height=0.893, resolution=16
    )
    if scale:
        cylinder.scale(random.uniform(1.0, 1.426), center=(0, 0, 0))
    return cylinder


def _add_cylinder_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    cylinder = _create_cylinder_transformer()
    cylinder.translate(
        (
            random.uniform(0.5, 0.9),
            0,
            min(
                state.pole_scaled_height,  # Always below pole
                max(
                    max(  #  Always above street lamp & traffic lights
                        np.max(state.traffic_light_heights),
                        state.lamp_height,
                    ),
                    random.uniform(9.9, 10.1),
                ),
            ),
        )
    )
    cylinder.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (
                0,
                0,
                np.deg2rad(
                    90 * (state.rot_indices[state.main_road] + random.choice([-1, 1]))
                ),
            )
        ),
        center=(0, 0, 0),
    )

    mesh += cylinder


def _add_triple_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    z = min(
        state.pole_scaled_height,  # Always below pole
        max(
            max(  #  Always above street lamp & traffic lights
                np.max(state.traffic_light_heights),
                state.lamp_height,
            ),
            state.pole_scaled_height - random.uniform(4.1, 4.2),
        ),
    )

    for i in range(3):
        cylinder = _create_cylinder_transformer(scale=False)
        cylinder.translate(
            (
                random.uniform(0.65, 0.85),
                0,
                z,
            )
        )
        cylinder.rotate(
            R=o3d.geometry.get_rotation_matrix_from_xyz(
                (
                    0,
                    0,
                    np.deg2rad(90 * (state.rot_indices[state.main_road] + i + 1)),
                )
            ),
            center=(0, 0, 0),
        )
        mesh += cylinder

    end_z = max(z, state.pole_scaled_height - random.uniform(0.78, 1.2))

    crossbar_1 = load_double_crossbar(2)
    crossbar_1.translate((0, 0, np.interp(0.5, (0, 1), (z, end_z))))
    crossbar_1.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
        ),
        center=(0, 0, 0),
    )
    mesh += crossbar_1

    crossbar_2 = load_double_crossbar(1)
    crossbar_2.translate((0, 0, end_z))
    crossbar_2.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
        ),
        center=(0, 0, 0),
    )
    mesh += crossbar_2

    state.crossbars_placed = True
