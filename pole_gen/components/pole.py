import numpy as np
import open3d as o3d

from pole_gen.models.state import State


def add_pole(
    mesh: o3d.geometry.TriangleMesh,
    state: State,
    slices: int = 0,
    circle_resolution: int = 16,
    base_radius: float = 0.143,
    top_radius: float = 0.0895,
):
    pole_mesh = o3d.geometry.TriangleMesh()
    slices += 1

    for i in range(slices + 1):
        z = state.pole_base_height * i / slices
        radius = base_radius + (top_radius - base_radius) * (i / slices)
        for j in range(circle_resolution):
            angle = 2 * np.pi * j / circle_resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pole_mesh.vertices.append([x, y, z])

    for i in range(slices):
        for j in range(circle_resolution):
            next_j = (j + 1) % circle_resolution
            pole_mesh.triangles.append(
                [
                    i * circle_resolution + j,
                    i * circle_resolution + next_j,
                    (i + 1) * circle_resolution + j,
                ]
            )
            pole_mesh.triangles.append(
                [
                    (i + 1) * circle_resolution + j,
                    i * circle_resolution + next_j,
                    (i + 1) * circle_resolution + next_j,
                ]
            )

    # Add bottom cap
    # bottom_center_idx = len(mesh.vertices)
    # mesh.vertices.append([0, 0, 0])
    # for j in range(circle_resolution):
    #     next_j = (j + 1) % circle_resolution
    #     mesh.triangles.append([bottom_center_idx, j, next_j])

    # Add top cap
    top_center_idx = len(pole_mesh.vertices)
    pole_mesh.vertices.append([0, 0, state.pole_base_height])
    for j in range(circle_resolution):
        next_j = (j + 1) % circle_resolution
        pole_mesh.triangles.append(
            [
                top_center_idx,
                slices * circle_resolution + j,
                slices * circle_resolution + next_j,
            ]
        )

    pole_mesh.compute_vertex_normals()
    if state.pole_scale != 1.0:
        pole_mesh.scale(state.pole_scale, center=(0, 0, 0))
    mesh += pole_mesh
