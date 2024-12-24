import numpy as np
import open3d as o3d

from ..components.crossbar import create_double_crossbar
from ..models import State, UtilityPoleLabel

TRIPLE_TRANSFORMER_MIN_FREE_SPACE = 5.0


def add_transformer(state: State):
    if np.random.random() > 0.33:
        return

    configuration = np.random.randint(0, 3)

    match configuration:
        case 0:
            _add_box_transformer(state)
        case 1:
            _add_cylinder_transformer(state)
        case _:
            # If we have enough space to add the crossbars for the triple transformer
            if (
                state.pole_scaled_height
                - max(np.max(state.traffic_light_heights), state.lamp_height)
                > TRIPLE_TRANSFORMER_MIN_FREE_SPACE
            ):
                _add_triple_transformer(state)
            else:
                _add_cylinder_transformer(state)


def _add_box_transformer(state: State):
    size = (0.417, 0.765, 0.573)
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0], height=size[1], depth=size[2]
    )
    box.translate(
        (
            np.random.uniform(0.45, 0.5) - (size[0] / 2),
            -(size[1] / 2),
            np.random.uniform(4.404, 4.73) - (size[2] / 2),
        )
    )
    box.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (
                0,
                0,
                np.deg2rad(
                    (90 * state.rot_indices[state.main_road])
                    + np.random.uniform(90, 270)
                ),
            )
        ),
        center=(0, 0, 0),
    )

    state.geometry[box] = UtilityPoleLabel.TRANSFORMER


def _create_cylinder_transformer(scale: bool = True):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.318, height=0.893, resolution=16
    )
    if scale:
        cylinder.scale(np.random.uniform(1.0, 1.426), center=(0, 0, 0))
    return cylinder


def _add_cylinder_transformer(state: State):
    cylinder = _create_cylinder_transformer()
    cylinder.translate(
        (
            np.random.uniform(0.5, 0.9),
            0,
            min(
                state.pole_scaled_height,  # Always below pole
                max(
                    max(  #  Always above street lamp & traffic lights
                        np.max(state.traffic_light_heights),
                        state.lamp_height,
                    ),
                    np.random.uniform(9.9, 10.1),
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
                    90
                    * (state.rot_indices[state.main_road] + np.random.choice([-1, 1]))
                ),
            )
        ),
        center=(0, 0, 0),
    )

    state.geometry[cylinder] = UtilityPoleLabel.TRANSFORMER


def _add_triple_transformer(state: State):
    z = min(
        state.pole_scaled_height,  # Always below pole
        max(
            max(  #  Always above street lamp & traffic lights
                np.max(state.traffic_light_heights),
                state.lamp_height,
            ),
            state.pole_scaled_height - np.random.uniform(4.1, 4.2),
        ),
    )

    d = np.random.choice([-1, 1])
    for i in range(3):
        cylinder = _create_cylinder_transformer(scale=False)
        cylinder.translate(
            (
                np.random.uniform(0.65, 0.85),
                0,
                z,
            )
        )
        cylinder.rotate(
            R=o3d.geometry.get_rotation_matrix_from_xyz(
                (
                    0,
                    0,
                    np.deg2rad(90 * (state.rot_indices[state.main_road] + (i * d))),
                )
            ),
            center=(0, 0, 0),
        )
        state.geometry[cylinder] = UtilityPoleLabel.TRANSFORMER

    end_z = max(z, state.pole_scaled_height - np.random.uniform(0.78, 1.2))

    crossbar_1 = create_double_crossbar(2)
    crossbar_1.translate((0, 0, np.interp(0.5, (0, 1), (z, end_z))))
    crossbar_1.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
        ),
        center=(0, 0, 0),
    )
    state.geometry[crossbar_1] = UtilityPoleLabel.CROSSARM

    crossbar_2 = create_double_crossbar(1)
    crossbar_2.translate((0, 0, end_z))
    crossbar_2.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz(
            (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
        ),
        center=(0, 0, 0),
    )
    state.geometry[crossbar_2] = UtilityPoleLabel.CROSSARM
