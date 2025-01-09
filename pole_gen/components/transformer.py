import numpy as np
import open3d as o3d

from ..components.crossbar import create_double_crossbar
from ..models import Placement, PlacementClass, State, UtilityPoleLabel

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
    z_pos = np.random.uniform(4.404, 4.73)
    box.translate(
        (
            np.random.uniform(0.45, 0.5) - (size[0] / 2),
            -(size[1] / 2),
            z_pos - (size[2] / 2),
        )
    )
    z_rot = np.deg2rad(
        (90 * state.rot_indices[state.main_road]) + np.random.uniform(90, 270)
    )
    box.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, z_rot)),
        center=(0, 0, 0),
    )

    state.add_geometry(box, UtilityPoleLabel.TRANSFORMER)
    state.placements[PlacementClass.MISC].append(
        Placement(z_position=z_pos, z_rotation=z_rot, height=1.0)
    )


def _create_cylinder_transformer(scale: bool = True):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.318, height=0.893, resolution=16
    )
    if scale:
        cylinder.scale(np.random.uniform(1.0, 1.426), center=(0, 0, 0))
    return cylinder


def _add_cylinder_transformer(state: State):
    cylinder = _create_cylinder_transformer()
    z_pos = min(
        state.pole_scaled_height,  # Always below pole
        max(
            max(  #  Always above street lamp & traffic lights
                np.max(state.traffic_light_heights),
                state.lamp_height,
            )
            + 2.0,
            np.random.uniform(9.9, 10.1),
        ),
    )
    cylinder.translate((np.random.uniform(0.5, 0.9), 0, z_pos))
    z_rot = np.deg2rad(
        90 * (state.rot_indices[state.main_road] + np.random.choice([-1, 1]))
    )
    cylinder.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, z_rot)),
        center=(0, 0, 0),
    )

    state.add_geometry(cylinder, UtilityPoleLabel.TRANSFORMER)
    state.placements[PlacementClass.MISC].append(
        Placement(z_position=z_pos, z_rotation=z_rot, height=1.0)
    )


def _add_triple_transformer(state: State):
    z = min(
        state.pole_scaled_height,  # Always below pole
        max(
            max(  #  Always above street lamp & traffic lights
                np.max(state.traffic_light_heights),
                state.lamp_height,
            )
            + 2.0,
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
        z_rot = np.deg2rad(90 * (state.rot_indices[state.main_road] + (i * d)))
        cylinder.rotate(
            R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, z_rot)),
            center=(0, 0, 0),
        )
        state.add_geometry(cylinder, UtilityPoleLabel.TRANSFORMER)
        state.placements[PlacementClass.MISC].append(
            Placement(z_position=z, z_rotation=z_rot, height=1.0)
        )

    end_z = max(z, state.pole_scaled_height - np.random.uniform(0.78, 1.2))

    crossbar_1 = create_double_crossbar(2)
    crossbar_1.translate((0, 0, np.interp(0.5, (0, 1), (z, end_z))))
    z_rot = np.deg2rad(90 * state.rot_indices[state.main_road])
    crossbar_1.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, z_rot)),
        center=(0, 0, 0),
    )
    state.add_geometry(crossbar_1, UtilityPoleLabel.CROSSARM)
    state.placements[PlacementClass.MISC].append(
        Placement(z_position=z, z_rotation=z_rot, height=0.5)
    )
    state.placements[PlacementClass.MISC].append(
        Placement(z_position=z, z_rotation=z_rot + np.pi, height=0.5)
    )

    crossbar_2 = create_double_crossbar(1)
    crossbar_2.translate((0, 0, end_z))
    z_rot = np.deg2rad(90 * state.rot_indices[state.main_road])
    crossbar_2.rotate(
        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, z_rot)),
        center=(0, 0, 0),
    )
    state.add_geometry(crossbar_2, UtilityPoleLabel.CROSSARM)
    state.placements[PlacementClass.MISC].append(
        Placement(z_position=end_z, z_rotation=z_rot, height=0.5)
    )
    state.placements[PlacementClass.MISC].append(
        Placement(z_position=end_z, z_rotation=z_rot + np.pi, height=0.5)
    )
