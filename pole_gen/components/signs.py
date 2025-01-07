from math import inf
from typing import List

import numpy as np
import open3d as o3d

from ..models import Placement, State, UtilityPoleLabel

STOP_SIGN_HEIGHT: float = 1.0


def is_free_side_sign_position(state: State, placement: Placement) -> bool:
    if len(state.side_signs) == 0:
        return True

    h = placement.height or 0.0
    h_top = placement.z_position + (h / 2.0)
    h_bottom = placement.z_position - (h / 2.0)

    # Side signs never overlap in the z-axis
    if not all(
        h_bottom >= (sign.z_position + (sign.height or 0.0) / 2.0)
        or h_top <= (sign.z_position - (sign.height or 0.0) / 2.0)
        for sign in state.side_signs
    ):
        return False

    return True


def find_random_free_side_sign_position(
    state: State,
    sign_height: float,
    min_z_position: float,
    max_z_position: float,
    min_z_rotation: float,
    max_z_rotation: float,
) -> Placement | None:
    # Check every possible height and angle in random order, without replacement
    z_pos_list: List[float] = np.linspace(
        min_z_position,
        max_z_position,
        max(1, int((max_z_position - min_z_position) / 0.1)),
    )
    np.random.shuffle(z_pos_list)

    z_rot_list: List[float] = np.linspace(
        min_z_rotation,
        max_z_rotation,
        max(1, int(max_z_rotation - min_z_rotation)),
    )

    for z_pos in z_pos_list:
        np.random.shuffle(z_rot_list)
        for z_rot in z_rot_list:
            placement = Placement(z_pos, z_rot, sign_height)
            if is_free_side_sign_position(state, placement):
                return placement

    return None


def _create_side_street_sign(variant: int) -> o3d.geometry.TriangleMesh:
    if variant == 1:
        #  New main road sign, height = 0.4m
        return o3d.io.read_triangle_mesh("pole_gen/meshes/side_street_sign_1.ply")
    elif variant == 2:
        #  New side road sign, height = 0.4m
        return o3d.io.read_triangle_mesh("pole_gen/meshes/side_street_sign_2.ply")
    else:
        #  Old sign, height = 0.12m
        size = (0.76, 0.04, 0.12)
        mesh = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
        mesh.translate([0, -size[1] / 2, -size[2] / 2])
        return mesh


def add_signs(state: State):
    if np.random.random() > 0.75:
        return

    # Stop sign (only at intersections)
    if (
        state.is_intersection
        and not state.traffic_light_heights[1 - state.main_road] > 0.0
        and np.random.random() > 0.7
    ):
        # Rotate towards non-main road (with some randomization)
        r = 90 * state.rot_indices[1 - state.main_road]
        placement = find_random_free_side_sign_position(
            state,
            STOP_SIGN_HEIGHT,
            2.20,
            2.30,
            np.deg2rad(r - 7.0),
            np.deg2rad(r + 7.0),
        )

        if placement is not None:
            stop_sign_mesh = o3d.io.read_triangle_mesh("pole_gen/meshes/stop_sign.ply")
            stop_sign_mesh.rotate(
                stop_sign_mesh.get_rotation_matrix_from_xyz(
                    (0, 0, placement.z_rotation)
                ),
                center=(0, 0, 0),
            )
            stop_sign_mesh.translate([0, 0, placement.z_position])
            state.add_geometry(stop_sign_mesh, UtilityPoleLabel.SIGN)
            state.side_signs.append(placement)

    # Side street signs
    if state.is_intersection and True:
        for i in range(np.random.randint(1, 2)):
            variant = np.random.choice([1, 2, 3])
            side_street_sign = _create_side_street_sign(variant)
            z_rot = 90 * np.random.randint(0, 3)
            sign_height = 0.4 if variant < 3 else 0.12

            bo = (sign_height / 2.0) + 0.1  # Boundary offset
            ro = 20.0  # Rotation offset

            placement = find_random_free_side_sign_position(
                state,
                sign_height,
                max(
                    2.9,
                    (
                        (max(state.pedestrian_signal_heights) + bo)
                        if any(l > 0 for l in state.pedestrian_signal_heights)
                        else -inf
                    ),  # Below pedestrian signals
                ),
                min(
                    4.2,
                    (
                        (min([l for l in state.traffic_light_heights if l > 0]) - bo)
                        if any(l > 0 for l in state.traffic_light_heights)
                        else inf
                    ),  # Below traffic lights
                    (
                        (state.lamp_height - bo) if state.lamp_height > 0 else inf
                    ),  # Below lamps
                    state.pole_scaled_height - bo,  # Keep below pole
                ),
                np.deg2rad(z_rot - ro),
                np.deg2rad(z_rot + ro),
            )
            if placement is not None:
                # Offset to radius of pole
                side_street_sign.translate(
                    [state.pole_radius_at(placement.z_position), 0, 0]
                )
                side_street_sign.rotate(
                    side_street_sign.get_rotation_matrix_from_xyz(
                        (0, 0, placement.z_rotation)
                    ),
                    center=(0, 0, 0),
                )
                side_street_sign.translate([0, 0, placement.z_position])
                state.add_geometry(side_street_sign, UtilityPoleLabel.SIGN)
                state.side_signs.append(placement)

    pass


def add_signs_old(state: State):
    if np.random.random() > 0.5:
        return

    # Bigass rectangle
    if np.random.random() <= 0.3:
        SIZE = (2.28, 0.139, 0.469)
        mesh = o3d.geometry.TriangleMesh.create_box(SIZE[0], SIZE[1], SIZE[2])
        z = np.random.uniform(4.75, 5.0) - (SIZE[2] / 2)
        mesh.translate([-SIZE[0] / 2, state.pole_radius_at(z) + (SIZE[1] / 2), z])
        # Rotate a bit on all axes
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz(
                (
                    np.deg2rad(np.radians(np.random.uniform(-2, 2))),
                    np.deg2rad(np.radians(np.random.uniform(-2, 2))),
                    np.deg2rad(np.radians(np.random.uniform(-2, 2))),
                )
            )
        )
        # Rotate towards the road
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz(
                (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
            ),
            center=(0, 0, 0),
        )
        state.add_geometry(mesh, UtilityPoleLabel.SIGN)

    # Tiny square ad
    if np.random.random() <= 0.3:
        SIZE = (0.392, 0.139, 0.392)
        mesh = o3d.geometry.TriangleMesh.create_box(SIZE[0], SIZE[1], SIZE[2])
        z = np.random.uniform(2.75, 3.0) - (SIZE[2] / 2)
        mesh.translate([state.pole_radius_at(z) + (SIZE[0] / 2), -SIZE[1] / 2, z])
        # Rotate towards sidewalk
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz(
                (
                    0,
                    0,
                    np.deg2rad(
                        90 * state.rot_indices[state.main_road]
                        + np.random.uniform(135, 225)
                    ),
                )
            ),
            center=(0, 0, 0),
        )
        state.add_geometry(mesh, UtilityPoleLabel.SIGN)
        state.add_geometry(mesh, UtilityPoleLabel.SIGN)
