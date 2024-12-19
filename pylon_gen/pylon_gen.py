import numpy as np
import open3d as o3d


def generate_pole(
    slices: int = 0,
    circle_resolution: int = 16,
    height: float = 10.5,
    base_radius: float = 0.11785,
    top_radius: float = 0.1115,
):
    mesh = o3d.geometry.TriangleMesh()
    slices += 1

    for i in range(slices + 1):
        z = height * i / slices
        radius = base_radius + (top_radius - base_radius) * (i / slices)
        for j in range(circle_resolution):
            angle = 2 * np.pi * j / circle_resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            mesh.vertices.append([x, y, z])

    for i in range(slices):
        for j in range(circle_resolution):
            next_j = (j + 1) % circle_resolution
            mesh.triangles.append(
                [
                    i * circle_resolution + j,
                    i * circle_resolution + next_j,
                    (i + 1) * circle_resolution + j,
                ]
            )
            mesh.triangles.append(
                [
                    (i + 1) * circle_resolution + j,
                    i * circle_resolution + next_j,
                    (i + 1) * circle_resolution + next_j,
                ]
            )

    # Add bottom cap
    bottom_center_idx = len(mesh.vertices)
    mesh.vertices.append([0, 0, 0])
    for j in range(circle_resolution):
        next_j = (j + 1) % circle_resolution
        mesh.triangles.append([bottom_center_idx, j, next_j])

    # Add top cap
    top_center_idx = len(mesh.vertices)
    mesh.vertices.append([0, 0, height])
    for j in range(circle_resolution):
        next_j = (j + 1) % circle_resolution
        mesh.triangles.append(
            [
                top_center_idx,
                slices * circle_resolution + j,
                slices * circle_resolution + next_j,
            ]
        )

    mesh.compute_vertex_normals()
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
