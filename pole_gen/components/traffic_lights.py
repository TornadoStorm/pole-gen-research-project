import random

import numpy as np
import open3d as o3d

from pole_gen.models.state import State


def add_traffic_lights(mesh: o3d.geometry.TriangleMesh, state: State):
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
