import numpy as np
import open3d as o3d

from .models import LABEL_COLORS, State, UtilityPoleLabel


def scan_geometry(state: State):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        np.asarray(state.geometry.vertices, dtype=np.float32),
        np.asarray(state.geometry.triangles, dtype=np.uint32),
    )

    min_points = 800
    max_points = 103000
    sensor_pos = np.array([5, 7, 0])

    # Sample random points on the geometry
    pc: o3d.geometry.PointCloud = state.geometry.sample_points_uniformly(
        number_of_points=np.random.randint(min_points, max_points)
    )

    # Raycast to these points and only keep the points that are visible
    ray_dirs = np.array(pc.points) - sensor_pos
    rays = o3d.core.Tensor(
        [
            [sensor_pos[0], sensor_pos[1], sensor_pos[2], d[0], d[1], d[2]]
            for d in ray_dirs
        ],
        dtype=o3d.core.Dtype.Float32,
    )
    hits = scene.cast_rays(rays)

    points = []
    colors = []
    for i in range(len(hits["t_hit"])):
        primitive = hits["primitive_ids"][i]
        if primitive == o3d.t.geometry.RaycastingScene.INVALID_ID:
            continue

        primitive = int(primitive.item())

        label: UtilityPoleLabel = state.triangle_labels[primitive]
        if label == UtilityPoleLabel.UNLABELED:
            continue

        point = sensor_pos + ray_dirs[i] * float(hits["t_hit"][i].item())
        points.append(point)
        colors.append(LABEL_COLORS[label.value])

    result = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
    )
    result.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float32))

    return result
