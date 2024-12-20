from typing import Tuple

import numpy as np
import open3d as o3d


def create_quad(
    center: Tuple[float, float, float], x_scale: float, y_scale: float
) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [center[0] - x_scale / 2, center[1] - y_scale / 2, center[2]],
            [center[0] + x_scale / 2, center[1] - y_scale / 2, center[2]],
            [center[0] + x_scale / 2, center[1] + y_scale / 2, center[2]],
            [center[0] - x_scale / 2, center[1] + y_scale / 2, center[2]],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    return mesh


def normalize_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    scale = max_bounds - min_bounds
    vertices -= center  # Center
    vertices /= scale.max()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)  # Normalize


# TODO Center mesh on position
# TODO Calculate rotation & positon
def create_cylinder(
    resolution: int,
    base_radius: float,
    top_radius: float,
    depth: float,
    slices: int = 0,
    top_cap=True,
    bottom_cap=True,
    position=(0, 0, 0),
    rotation=(0, 0, 0),
):
    mesh = o3d.geometry.TriangleMesh()
    slices += 1

    for i in range(slices + 1):
        t = i / slices
        z = depth * (t - 0.5)
        radius = base_radius + (top_radius - base_radius) * t
        for j in range(resolution):
            angle = 2 * np.pi * j / resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            mesh.vertices.append([x, y, z])

    for i in range(slices):
        for j in range(resolution):
            next_j = (j + 1) % resolution
            mesh.triangles.append(
                [
                    i * resolution + j,
                    i * resolution + next_j,
                    (i + 1) * resolution + j,
                ]
            )
            mesh.triangles.append(
                [
                    (i + 1) * resolution + j,
                    i * resolution + next_j,
                    (i + 1) * resolution + next_j,
                ]
            )

    # Add bottom cap
    if bottom_cap:
        bottom_center_idx = len(mesh.vertices)
        mesh.vertices.append([0, 0, -depth / 2])
        for j in range(resolution):
            next_j = (j + 1) % resolution
            mesh.triangles.append([bottom_center_idx, j, next_j])

    # Add top cap
    if top_cap:
        top_center_idx = len(mesh.vertices)
        mesh.vertices.append([0, 0, depth / 2])
        for j in range(resolution):
            next_j = (j + 1) % resolution
            mesh.triangles.append(
                [
                    top_center_idx,
                    slices * resolution + j,
                    slices * resolution + next_j,
                ]
            )

    mesh.compute_vertex_normals()

    # Transform
    mesh.rotate(R=o3d.geometry.get_rotation_matrix_from_xyz(rotation), center=(0, 0, 0))
    mesh.translate(position)

    return mesh
