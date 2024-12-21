import numpy as np
import open3d as o3d

from pole_gen.models import State, UtilityPoleLabel
from utils.mesh_tools import create_cylinder


def add_pole(
    state: State,
):
    pole_mesh = create_cylinder(
        resolution=16,
        base_radius=0.143,
        top_radius=0.0895,
        depth=state.pole_base_height,
        top_cap=True,
        bottom_cap=False,
        position=(0, 0, state.pole_base_height / 2),
    )

    if state.pole_scale != 1.0:
        pole_mesh.scale(state.pole_scale, center=(0, 0, 0))

    state.geometry[pole_mesh] = UtilityPoleLabel.POLE
