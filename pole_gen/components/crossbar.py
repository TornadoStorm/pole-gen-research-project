import numpy as np
import open3d as o3d

from ..models import State, UtilityPoleLabel


def create_double_crossbar(variant: int) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(f"pole_gen/meshes/crossbar_double_{variant}.ply")


def create_top_crossbar() -> o3d.geometry.TriangleMesh:
    size = (2.93, 0.0829, 0.137)
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size[0], height=size[1], depth=size[2]
    )
    mesh.translate((-size[0] / 2, -size[1] / 2, -size[2] / 2))
    return mesh


def create_single_crossbar() -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(f"pole_gen/meshes/crossbar_single_1.ply")


def add_crossbar(state: State):
    if any(l == UtilityPoleLabel.CROSSARM for l in state.triangle_labels):
        return

    if np.random.random() > 0.4:
        return

    configuration = np.random.randint(0, 4)

    r = o3d.geometry.get_rotation_matrix_from_xyz(
        (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
    )

    z_min = max(state.lamp_height, state.traffic_light_heights[state.main_road]) + 1
    if z_min > state.pole_scaled_height:
        return

    match configuration:
        case 0:
            # 3 on side of (main) road (0.3m down, 1m spacing)
            z = state.pole_scaled_height - np.random.uniform(0.25, 0.35)
            for _ in range(3):
                if z < z_min:
                    break  # Stop if we reach the lamp or traffic light
                ca = create_single_crossbar()
                ca.translate((0, 0, z))
                ca.rotate(R=r, center=(0, 0, 0))
                z -= np.random.uniform(0.9, 1.1)
                state.add_geometry(ca, UtilityPoleLabel.CROSSARM)

            # May have one between street lamp and traffic light if they are present
            if (
                state.lamp_height > 0
                and state.traffic_light_heights[state.main_road] > 0
                and np.random.random() <= 0.5
            ):
                ca = create_single_crossbar()
                ca.translate(
                    (
                        0,
                        0,
                        np.interp(
                            0.5,
                            (0, 1),
                            (
                                state.lamp_height,
                                state.traffic_light_heights[state.main_road],
                            ),
                        ),
                    )
                )
                ca.rotate(R=r, center=(0, 0, 0))
                state.add_geometry(ca, UtilityPoleLabel.CROSSARM)
        case 1:
            # One big plank on top (up to 0.3m down)
            ca = create_top_crossbar()
            ca.scale(np.random.uniform(0.9, 1.1), center=(0, 0, 0))
            ca.translate((0, 0, state.pole_scaled_height - (np.random.random() * 0.35)))
            ca.rotate(R=r, center=(0, 0, 0))
            state.add_geometry(ca, UtilityPoleLabel.CROSSARM)
        case 2:
            # Two smalls near top (Like 0.62-0.9m from top)
            ca = create_double_crossbar(1)
            ca.translate(
                (0, 0, state.pole_scaled_height - np.random.uniform(0.62, 0.9))
            )
            ca.rotate(R=r, center=(0, 0, 0))
            state.add_geometry(ca, UtilityPoleLabel.CROSSARM)
        case 3:
            # Single on side of (main) road (~1.3-2.9m down)
            ca = create_single_crossbar()
            ca.translate(
                (
                    0,
                    0,
                    max(z_min, state.pole_scaled_height - np.random.uniform(1.3, 2.9)),
                )
            )
            ca.rotate(R=r, center=(0, 0, 0))
            state.add_geometry(ca, UtilityPoleLabel.CROSSARM)
        case 4:
            # 3 pairs (~1.2m spacing, ~0.3m down)
            z = state.pole_scaled_height - np.random.uniform(0.25, 0.35)
            for _ in range(3):
                if z < z_min:
                    break  # Stop if we reach the lamp or traffic light
                ca = create_double_crossbar(1)
                ca.translate((0, 0, z))
                ca.rotate(R=r, center=(0, 0, 0))
                z -= np.random.uniform(1.1, 1.3)
                state.add_geometry(ca, UtilityPoleLabel.CROSSARM)
            pass
