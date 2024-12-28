import numpy as np
import open3d as o3d

from utils.mesh_tools import create_cylinder

from ..models import State, UtilityPoleLabel


def add_pole(
    state: State,
):
    state.pole_base_radius = 0.143
    state.pole_top_radius = 0.0895

    pole_mesh = create_cylinder(
        resolution=16,
        base_radius=state.pole_base_radius,
        top_radius=state.pole_top_radius,
        depth=state.pole_base_height,
        top_cap=True,
        bottom_cap=False,
        position=(0, 0, state.pole_base_height / 2),
    )

    if state.pole_scale != 1.0:
        pole_mesh.scale(state.pole_scale, center=(0, 0, 0))

    state.add_geometry(pole_mesh, UtilityPoleLabel.POLE)
