import random

import numpy as np
import open3d as o3d

from utils.mesh_tools import create_quad, normalize_mesh


class _State:
    intersection: bool = False
    road_presence: list[int] = [0, 0]
    main_road: int = 0
    rot_indices: list[int] = [0, 0]
    pole_base_height: float = 8.45
    pole_scale: float = 1.0
    pole_scaled_height: float = 8.45
    traffic_light_heights: list[float] = [0, 0]


def _add_road_meshes(mesh: o3d.geometry.TriangleMesh, state: _State):
    road_meshes = []
    if state.road_presence[0] != 0:
        road_meshes.append(
            create_quad(
                (state.road_presence[0], 0, 0),
                1 if state.main_road == 0 else 0.5,
                4 if state.main_road == 0 else 2,
            )
        )
    if state.road_presence[1] != 0:
        road_meshes.append(
            create_quad(
                (0, state.road_presence[1], 0),
                4 if state.main_road == 1 else 2,
                1 if state.main_road == 1 else 0.5,
            )
        )

    for road_mesh in road_meshes:
        mesh += road_mesh


def _add_pole(
    mesh: o3d.geometry.TriangleMesh,
    state: _State,
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


def _add_traffic_lights(mesh: o3d.geometry.TriangleMesh, state: _State):
    if state.intersection:
        # TODO Add pedestrian lights
        spawn_chances = [0.4 for _ in state.road_presence]
        spawn_chances[state.main_road] = 1.0

        for i in range(len(spawn_chances)):
            if random.random() > spawn_chances[i]:
                continue

            traffic_light_index = random.choice([1, 2, 3])
            state.traffic_light_heights[i] = random.uniform(4.17012, 5.2)

            match traffic_light_index:
                case 1:
                    traffic_light_mesh = o3d.io.read_triangle_mesh(
                        "pole_gen/meshes/traffic_light_1.ply"
                    )
                    if random.random() <= 0.5:
                        traffic_light_mesh += o3d.io.read_triangle_mesh(
                            "pole_gen/meshes/traffic_light_1_sign.ply"
                        )
                    traffic_light_mesh.rotate(
                        traffic_light_mesh.get_rotation_matrix_from_xyz(
                            (0, 0, np.deg2rad(90 * state.rot_indices[i]))
                        ),
                        center=(0, 0, 0),
                    )
                    traffic_light_mesh.translate([0, 0, state.traffic_light_heights[i]])
                    mesh += traffic_light_mesh
                case 2:
                    traffic_light_mesh = o3d.io.read_triangle_mesh(
                        "pole_gen/meshes/traffic_light_2.ply"
                    )
                    traffic_light_mesh.rotate(
                        traffic_light_mesh.get_rotation_matrix_from_xyz(
                            (0, 0, np.deg2rad(90 * state.rot_indices[i]))
                        ),
                        center=(0, 0, 0),
                    )
                    traffic_light_mesh.translate([0, 0, state.traffic_light_heights[i]])
                    mesh += traffic_light_mesh
                case 3:
                    traffic_light_mesh = o3d.io.read_triangle_mesh(
                        "pole_gen/meshes/traffic_light_3.ply"
                    )
                    traffic_light_mesh.rotate(
                        traffic_light_mesh.get_rotation_matrix_from_xyz(
                            (0, 0, np.deg2rad(90 * state.rot_indices[i]))
                        ),
                        center=(0, 0, 0),
                    )
                    traffic_light_mesh.translate([0, 0, state.traffic_light_heights[i]])
                    mesh += traffic_light_mesh


def _add_lamp(mesh: o3d.geometry.TriangleMesh, state: _State):
    if random.random() <= 0.8:
        lamp_mesh = o3d.io.read_triangle_mesh("pole_gen/meshes/lamp.ply")
        # Rotate to (main) road
        lamp_mesh.rotate(
            lamp_mesh.get_rotation_matrix_from_xyz(
                (0, 0, np.deg2rad(90 * state.rot_indices[state.main_road]))
            ),
            center=(0, 0, 0),
        )
        # Randomize rotation
        lamp_mesh.rotate(
            lamp_mesh.get_rotation_matrix_from_xyz(
                (0, np.deg2rad(random.uniform(-5.0, 0.0)), 0)
            ),
            center=(0, 0, 0),
        )
        # Randomize position
        lamp_mesh.translate(
            [
                0,
                0,
                max(
                    state.traffic_light_heights[state.main_road]
                    + 1,  # Always above traffic light
                    min(
                        7.5 + random.uniform(-1.0, 1.94), state.pole_scaled_height - 0.3
                    ),
                ),
            ]
        )
        mesh += lamp_mesh


def generate_utility_pole():
    mesh = o3d.geometry.TriangleMesh()

    state = _State()

    # Simulate having a road at a given side (1, -1) or not (0)
    # Index 0 = x axis, index 1 = z axis
    state.road_presence = [random.choice([1, 0, -1]), random.choice([1, 0, -1])]
    if all(road == 0 for road in state.road_presence):
        state.road_presence[random.choice([0, 1])] = random.choice([1, -1])
    # Pick one of the roads to be the "main" road
    state.main_road = random.choice(
        [i for i, v in enumerate(state.road_presence) if v != 0]
    )

    # Other shared variables
    state.intersection = len([road for road in state.road_presence if road != 0]) > 1
    state.pole_base_height = 8.45
    state.pole_scale = random.uniform(1.0, 1.955)
    state.pole_scaled_height = state.pole_base_height * state.pole_scale

    # Useful for rotating things to align with roads
    state.rot_indices = [0, 0]
    match state.road_presence[0]:
        case 1:
            state.rot_indices[0] = 0
        case -1:
            state.rot_indices[0] = 2
    match state.road_presence[1]:
        case 1:
            state.rot_indices[1] = 1
        case -1:
            state.rot_indices[1] = 3

    _add_road_meshes(mesh, state)
    _add_pole(mesh, state)
    _add_traffic_lights(mesh, state)
    _add_lamp(mesh, state)

    normalize_mesh(mesh)
    return mesh
