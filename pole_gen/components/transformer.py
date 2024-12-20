import random

import numpy as np
import open3d as o3d

from pole_gen.models.state import State

TRANSFORMER_SPAWN_CHANCE = 0.33

CYLINDER_RESOLUTION = 16
CYLINDER_RADIUS = 0.1
CYLINDER_HEIGHT = 0.5


def add_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    # _add_cylinder_transformer(mesh, state)
    # return

    random_val = np.random.random()
    if random_val > TRANSFORMER_SPAWN_CHANCE:
        return

    if random_val <= TRANSFORMER_SPAWN_CHANCE / 3:
        _add_box_transformer(mesh, state)
    elif random_val <= 2 * TRANSFORMER_SPAWN_CHANCE / 3:
        _add_cylinder_transformer(mesh, state)
    else:
        _add_triple_transformer(mesh, state)


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
            (0, 0, random.uniform(0, 2 * np.pi))
        ),
        center=(0, 0, 0),
    )

    mesh += box
    return


def _add_cylinder_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    # TODO Implement
    pass


def _add_triple_transformer(mesh: o3d.geometry.TriangleMesh, state: State):
    # TODO Implement
    pass
