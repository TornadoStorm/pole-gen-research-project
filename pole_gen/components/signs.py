import numpy as np
import open3d as o3d

from ..models import State, UtilityPoleLabel


def add_signs(state: State):
    if np.random.random() > 0.5:
        return

    # Bigass rectangle
    if np.random.random() <= 0.3:
        H = 0.139
        mesh = o3d.geometry.TriangleMesh.create_box(2.28, H, 0.469)
        z = np.random.uniform(4.75, 5.0)
        mesh.translate([0, state.pole_radius_at(z) + (H / 2), z])
        # Rotate a bit on all axes
        mesh.rotate([0, 0, 0], np.radians(np.random.uniform(-3, 3)))
        # Rotate towards the road
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz(
                (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
            ),
            center=(0, 0, 0),
        )
        state.geometry[mesh] = UtilityPoleLabel.SIGN

    # Tiny square ad
    if np.random.random() <= 0.3:
        W = 0.392
        D = 0.139
        mesh = o3d.geometry.TriangleMesh.create_box(W, D, W)
        z = np.random.uniform(2.75, 3.0)
        mesh.translate([state.pole_radius_at(z) + (W / 2), 0, z])
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
            )
        ),
