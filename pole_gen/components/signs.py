from math import inf
from typing import List

import numpy as np
import open3d as o3d

from utils.math import signed_angle_difference

from ..models import Placement, State, UtilityPoleLabel

STOP_SIGN_HEIGHT: float = 1.0


def _is_free_side_sign_position(state: State, placement: Placement) -> bool:
    if len(state.side_signs) == 0:
        return True

    h = placement.height or 0.0
    h_top = placement.z_position + (h / 2.0)
    h_bottom = placement.z_position - (h / 2.0)

    # Side signs never overlap in the z-axis
    if any(
        h_bottom < (sign.z_position + (sign.height or 0.0) / 2.0)
        and h_top > (sign.z_position - (sign.height or 0.0) / 2.0)
        for sign in state.side_signs
    ):
        return False

    # Side signs can overlap normal signs, but only if their angle difference is >= 90 degrees
    if any(
        h_bottom < (sign.z_position + (sign.height or 0.0) / 2.0)
        and h_top > (sign.z_position - (sign.height or 0.0) / 2.0)
        and abs(signed_angle_difference(sign.z_rotation, placement.z_rotation))
        < np.pi / 2.0
        for sign in state.normal_signs
    ):
        return False

    return True


def _find_random_free_side_sign_position(
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
            if _is_free_side_sign_position(state, placement):
                return placement

    return None


def _create_side_sign(
    length: float, height: float, thickness: float
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(length, thickness, height)
    mesh.translate([0, -thickness / 2, -height / 2])
    return mesh


def _create_side_street_sign(variant: int) -> o3d.geometry.TriangleMesh:
    if variant == 1:
        #  New main road sign, height = 0.4m
        return o3d.io.read_triangle_mesh("pole_gen/meshes/side_street_sign_1.ply")
    elif variant == 2:
        #  New side road sign, height = 0.4m
        return o3d.io.read_triangle_mesh("pole_gen/meshes/side_street_sign_2.ply")
    else:
        #  Old sign, height = 0.12m
        return _create_side_sign(length=0.76, height=0.12, thickness=0.04)


def _add_stop_sign(state: State):
    # Rotate towards non-main road (with some randomization)
    r = 90 * state.rot_indices[1 - state.main_road]
    placement = _find_random_free_side_sign_position(
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
            stop_sign_mesh.get_rotation_matrix_from_xyz((0, 0, placement.z_rotation)),
            center=(0, 0, 0),
        )
        stop_sign_mesh.translate([0, 0, placement.z_position])
        state.add_geometry(stop_sign_mesh, UtilityPoleLabel.SIGN)
        state.side_signs.append(placement)


def _add_side_street_signs(state: State):
    for i in range(np.random.randint(1, 2)):
        variant = np.random.choice([1, 2, 3])
        side_street_sign = _create_side_street_sign(variant)
        z_rot = 90 * np.random.randint(0, 3)
        sign_height = 0.4 if variant < 3 else 0.12

        bo = (sign_height / 2.0) + 0.1  # Boundary offset
        ro = 20.0  # Rotation offset

        placement = _find_random_free_side_sign_position(
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
            side_street_sign.translate(
                [state.pole_radius_at(placement.z_position), 0, placement.z_position]
            )
            side_street_sign.rotate(
                side_street_sign.get_rotation_matrix_from_xyz(
                    (0, 0, placement.z_rotation)
                ),
                center=(0, 0, 0),
            )
            state.add_geometry(side_street_sign, UtilityPoleLabel.SIGN)
            state.side_signs.append(placement)


def _add_small_rectangular_side_signs(state: State):
    MIN_H = 2.8
    MAX_H = 3.8

    spacing = 0.01
    min_occupied = inf
    max_occupied = -inf
    z_rot = 0.0
    initial_check_dir = np.random.choice([-1, 1])

    for i in range(np.random.randint(1, 3)):
        h = 0.3 if np.random.random() <= 0.75 else 0.6

        # First placement is random
        if i == 0:
            possible_rotations = []
            if state.is_intersection:
                possible_rotations = [
                    np.deg2rad(-45),
                    np.deg2rad((90 * state.rot_indices[state.main_road]) - 45),
                    np.deg2rad((180 * state.rot_indices[state.main_road]) - 45),
                    np.deg2rad((270 * state.rot_indices[state.main_road]) - 45),
                ]
            else:
                possible_rotations = [
                    np.deg2rad(-45),
                    np.deg2rad((180 * state.rot_indices[state.main_road]) - 45),
                ]
            np.random.shuffle(possible_rotations)

            rot_diff = np.deg2rad(2)

            for z_rot in possible_rotations:
                placement = _find_random_free_side_sign_position(
                    state=state,
                    sign_height=h,
                    min_z_position=MIN_H,
                    max_z_position=MAX_H,
                    min_z_rotation=z_rot - rot_diff,
                    max_z_rotation=z_rot + rot_diff,
                )
                if placement is not None:
                    z_rot = placement.z_rotation
                    break
        else:
            # Subsequent placements are based on the previous one
            pd = initial_check_dir
            for i in range(2):
                placement = Placement(
                    z_position=(
                        max(MIN_H, (min_occupied - ((h / 2.0) + spacing)))
                        if pd == 1
                        else min(MAX_H, (max_occupied + ((h / 2.0) + spacing)))
                    ),
                    z_rotation=z_rot,
                    height=h,
                )
                if not _is_free_side_sign_position(state, placement):
                    # Try the other direction
                    placement = None
                    pd *= -1
                else:
                    break

        if placement is None:
            break  # Welp, no more space I guess

        min_occupied = min(min_occupied, placement.z_position - (h / 2.0))
        max_occupied = max(max_occupied, placement.z_position + (h / 2.0))

        mesh = _create_side_sign(length=0.3, height=h, thickness=0.03)
        mesh.translate(
            [state.pole_radius_at(placement.z_position), 0, placement.z_position]
        )
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz((0, 0, z_rot)),
            center=(0, 0, 0),
        )
        state.add_geometry(mesh, UtilityPoleLabel.SIGN)
        state.side_signs.append(placement)


def add_signs(state: State):
    if np.random.random() > 0.75:
        return

    # Stop sign (only at intersections)
    if (
        state.is_intersection
        and not state.traffic_light_heights[1 - state.main_road] > 0.0
        and np.random.random() > 0.7
    ):
        _add_stop_sign(state)

    # Side street signs
    if state.is_intersection and np.random.random() <= 0.5:
        _add_side_street_signs(state)

    # Small rectangular side sign(s)
    if np.random.random() >= 0.5:
        _add_small_rectangular_side_signs(state)
