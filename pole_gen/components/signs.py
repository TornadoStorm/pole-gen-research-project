from math import inf
from typing import Callable, List

import numpy as np
import open3d as o3d

from utils.math import signed_angle_difference

from ..models import Placement, State, UtilityPoleLabel

STOP_SIGN_HEIGHT: float = 1.0


def _is_free_side_sign_position(state: State, placement: Placement) -> bool:
    h = placement.height or 0.0
    h_top = placement.z_position + (h / 2.0)
    h_bottom = placement.z_position - (h / 2.0)

    # Side signs never overlap in the z-axis
    if len(state.side_signs) > 0 and any(
        h_bottom < (sign.z_position + (sign.height or 0.0) / 2.0)
        and h_top > (sign.z_position - (sign.height or 0.0) / 2.0)
        for sign in state.side_signs
    ):
        return False

    # Side signs can overlap normal signs, but only if their angle difference is >= 90 degrees
    if len(state.normal_signs) > 0 and any(
        h_bottom < (sign.z_position + (sign.height or 0.0) / 2.0)
        and h_top > (sign.z_position - (sign.height or 0.0) / 2.0)
        and abs(signed_angle_difference(sign.z_rotation, placement.z_rotation))
        < np.pi / 2.0
        for sign in state.normal_signs
    ):
        return False

    return True


def _is_free_normal_sign_position(state: State, placement: Placement) -> bool:
    h = placement.height or 0.0
    h_top = placement.z_position + (h / 2.0)
    h_bottom = placement.z_position - (h / 2.0)

    # Normal signs never overlap in the z-axis
    if len(state.normal_signs) > 0 and any(
        h_bottom < (sign.z_position + (sign.height or 0.0) / 2.0)
        and h_top > (sign.z_position - (sign.height or 0.0) / 2.0)
        for sign in state.normal_signs
    ):
        return False

    # Normal signs can overlap side signs, but only if their angle difference is >= 90 degrees
    if len(state.side_signs) > 0 and any(
        h_bottom < (sign.z_position + (sign.height or 0.0) / 2.0)
        and h_top > (sign.z_position - (sign.height or 0.0) / 2.0)
        and abs(signed_angle_difference(sign.z_rotation, placement.z_rotation))
        < np.pi / 2.0
        for sign in state.side_signs
    ):
        return False

    return True


def _find_random_pos(
    state: State,
    sign_height: float,
    min_z_position: float,
    max_z_position: float,
    min_z_rotation: float,
    max_z_rotation: float,
    evaluation_function: Callable[[State, Placement], bool],
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
            if evaluation_function(state, placement):
                return placement

    return None


def _find_random_free_side_sign_position(
    state: State,
    sign_height: float,
    min_z_position: float,
    max_z_position: float,
    min_z_rotation: float,
    max_z_rotation: float,
) -> Placement | None:
    return _find_random_pos(
        state,
        sign_height,
        min_z_position,
        max_z_position,
        min_z_rotation,
        max_z_rotation,
        _is_free_side_sign_position,
    )


def _find_random_free_normal_sign_position(
    state: State,
    sign_height: float,
    min_z_position: float,
    max_z_position: float,
    min_z_rotation: float,
    max_z_rotation: float,
) -> Placement | None:
    return _find_random_pos(
        state,
        sign_height,
        min_z_position,
        max_z_position,
        min_z_rotation,
        max_z_rotation,
        _is_free_normal_sign_position,
    )


def _create_sign(
    width: float,
    height: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    z_rotation: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    """Creates a rectangular sign mesh.

    Args:
        width (float): Sign width.
        height (float): Sign height.
        x (float, optional): Horizontal offset. Defaults to 0.0.
        y (float, optional): Distance from the center of the pole. Defaults to 0.0.
        z (float, optional): Vertical position. Defaults to 0.0.
        z_rotation (float, optional): Z-rotation in radians. Defaults to 0.0.

    Returns:
        o3d.geometry.TriangleMesh: Procedurally generated sign mesh.
    """
    thickness = 0.03
    mesh = o3d.geometry.TriangleMesh.create_box(width, thickness, height)
    mesh.translate([x - (width / 2), thickness + y, z - (height / 2)])
    if z_rotation != 0.0:
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz((0, 0, z_rotation)),
            center=(0, 0, 0),
        )
    return mesh


def _create_side_sign(
    length: float,
    height: float,
    thickness: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    z_rotation: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(length, thickness, height)
    mesh.translate([x, y - (thickness / 2), z - (height / 2)])
    if z_rotation != 0.0:
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz((0, 0, z_rotation)),
            center=(0, 0, 0),
        )
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
    if not state.is_intersection:
        return

    if not state.traffic_light_heights[1 - state.main_road] > 0.0:
        return

    if np.random.random() > 0.3:
        return

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
    if not (state.is_intersection and np.random.random() <= 0.5):
        return

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
    if np.random.random() > 0.5:
        return

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

        mesh = _create_side_sign(
            length=0.3,
            height=h,
            thickness=0.03,
            x=state.pole_radius_at(placement.z_position),
            z=placement.z_position,
            z_rotation=z_rot,
        )
        state.add_geometry(mesh, UtilityPoleLabel.SIGN)
        state.side_signs.append(placement)


def _add_large_rectangular_street_sign(state: State):
    if not state.is_intersection:
        return

    if state.traffic_light_heights[state.main_road] > 0.0:
        return

    if np.random.random() > 0.3:
        return

    z_rot_deg = 90 * state.rot_indices[state.main_road]

    placement = _find_random_free_normal_sign_position(
        state=state,
        sign_height=0.4,
        min_z_position=4.5,
        max_z_position=5.0,
        min_z_rotation=np.deg2rad(z_rot_deg - 7.0),
        max_z_rotation=np.deg2rad(z_rot_deg + 7.0),
    )

    if placement is not None:
        w = np.random.uniform(1.8, 2.3)
        mesh = _create_sign(
            width=w,
            height=placement.height,
            x=np.random.uniform(0, w * (1 / 3)),
            y=state.pole_radius_at(0.0),
            z=placement.z_position,
            z_rotation=placement.z_rotation,
        )
        state.add_geometry(mesh, UtilityPoleLabel.SIGN)
        state.normal_signs.append(placement)


# Available rectangular sign dimensions
RECTANGULAR_SIGNS = [
    (1.0, 0.6),
    (0.5, 0.6),
    (0.3, 0.4),
    (0.4, 0.6),
    (0.4, 0.5),
    (0.6, 0.6),
    (0.6, 0.8),
    (0.7, 0.7),
]


def _add_rectangular_signs(state: State):
    # z_range: [2.4, 0.5]
    pass  # TODO Implement


def _add_large_rectangular_side_signs(state: State):
    # Size approx (0.4, 0.3) z >= 0.5
    pass  # TODO Implement


def add_signs(state: State):
    _add_stop_sign(state)
    _add_side_street_signs(state)
    _add_rectangular_signs(state)
    _add_small_rectangular_side_signs(state)
    _add_large_rectangular_side_signs(state)
    _add_large_rectangular_street_sign(state)
