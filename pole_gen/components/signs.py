import numpy as np
import open3d as o3d

from ..models import State, UtilityPoleLabel


def add_signs(state: State):
    if np.random.random() > 0.5:
        return

    # Bigass rectangle
    if np.random.random() <= 0.3:
        SIZE = (2.28, 0.139, 0.469)
        mesh = o3d.geometry.TriangleMesh.create_box(SIZE[0], SIZE[1], SIZE[2])
        z = np.random.uniform(4.75, 5.0) - (SIZE[2] / 2)
        mesh.translate([-SIZE[0] / 2, state.pole_radius_at(z), z])
        # Rotate a bit on all axes
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz(
                (
                    np.deg2rad(np.radians(np.random.uniform(-2, 2))),
                    np.deg2rad(np.radians(np.random.uniform(-2, 2))),
                    np.deg2rad(np.radians(np.random.uniform(-2, 2))),
                )
            ),
        )
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
        SIZE = (0.392, 0.139, 0.392)
        mesh = o3d.geometry.TriangleMesh.create_box(SIZE[0], SIZE[1], SIZE[2])
        z = np.random.uniform(2.75, 3.0) - (SIZE[2] / 2)
        mesh.translate([0, state.pole_radius_at(z) + SIZE[1], z])
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
        )
        state.geometry[mesh] = UtilityPoleLabel.SIGN
