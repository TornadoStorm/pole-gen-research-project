import numpy as np
import open3d as o3d

from ..models import State, UtilityPoleLabel

# Stinky hack to decide on the pedestrian light model from its context.
PED_MAP = {
    0: {
        (1, 1): 2,
        (1, -1): 1,
        (-1, 1): 1,
        (-1, -1): 2,
    },
    1: {
        (1, 1): 1,
        (1, -1): 2,
        (-1, 1): 2,
        (-1, -1): 1,
    },
}

MIN_PED_H = 2.34
MAX_PED_H = 2.62


def _add_predestrian_light(
    height: float, road_index: int, state: State, variant: int = 0
):
    pedestrian_light_mesh = o3d.io.read_triangle_mesh(
        f"pole_gen/meshes/pedestrian_traffic_light_{PED_MAP[road_index][(state.road_presence[0], state.road_presence[1])] + (variant * 2)}.ply"
    )
    pedestrian_light_mesh.rotate(
        pedestrian_light_mesh.get_rotation_matrix_from_xyz(
            (0, 0, np.deg2rad(90 * state.rot_indices[road_index] + 180))
        ),
        center=(0, 0, 0),
    )
    pedestrian_light_mesh.translate([0, 0, height])
    state.add_geometry(pedestrian_light_mesh, UtilityPoleLabel.PEDESTRIAN_SIGNAL)


def add_traffic_lights(state: State):
    if not state.is_intersection:
        return

    if np.random.random() <= 0.2:
        # Pedestrian lights
        road_index = np.random.choice(
            [i for i, v in enumerate(state.road_presence) if v != 0]
        )
        pedestrian_light_height = np.random.uniform(MIN_PED_H, MAX_PED_H)
        _add_predestrian_light(
            pedestrian_light_height, road_index, state, np.random.choice([0, 1])
        )
        state.pedestrian_signal_heights[road_index] = pedestrian_light_height
    elif np.random.random() <= 0.5:
        # Traffic lights with pedestrian lights
        spawn_chances = [0.4 for _ in state.road_presence]
        spawn_chances[state.main_road] = 1.0

        pedestrian_light_height = np.random.uniform(MIN_PED_H, MAX_PED_H)

        for i in range(len(spawn_chances)):
            if np.random.random() > spawn_chances[i]:
                continue

            traffic_light_index = np.random.choice([1, 2, 3])
            state.traffic_light_heights[i] = np.random.uniform(4.17012, 5.2)
            traffic_light_mesh: o3d.geometry.TriangleMesh = None

            match traffic_light_index:
                case 1:
                    traffic_light_mesh = o3d.io.read_triangle_mesh(
                        "pole_gen/meshes/traffic_light_1.ply"
                    )
                    if np.random.random() <= 0.5:
                        traffic_light_mesh += o3d.io.read_triangle_mesh(
                            "pole_gen/meshes/traffic_light_1_sign.ply"
                        )
                case 2:
                    traffic_light_mesh = o3d.io.read_triangle_mesh(
                        "pole_gen/meshes/traffic_light_2.ply"
                    )
                case 3:
                    traffic_light_mesh = o3d.io.read_triangle_mesh(
                        "pole_gen/meshes/traffic_light_3.ply"
                    )

            if traffic_light_mesh is not None:
                traffic_light_mesh.rotate(
                    traffic_light_mesh.get_rotation_matrix_from_xyz(
                        (0, 0, np.deg2rad(90 * state.rot_indices[i]))
                    ),
                    center=(0, 0, 0),
                )
                traffic_light_mesh.translate([0, 0, state.traffic_light_heights[i]])
                state.add_geometry(traffic_light_mesh, UtilityPoleLabel.TRAFFIC_LIGHT)

            _add_predestrian_light(
                pedestrian_light_height, i, state, np.random.choice([0, 1])
            )
            state.pedestrian_signal_heights[i] = pedestrian_light_height
