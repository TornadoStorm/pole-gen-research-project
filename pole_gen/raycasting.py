from typing import Tuple

import numpy as np
import open3d as o3d

from .models import State, UtilityPoleLabel


def scan_geometry(
    state: State,
    npoints: int = 5000,
    sensor_pos: Tuple[float, float, float] = (0, 0, 0),
    jitter: float = 0.0,
) -> o3d.t.geometry.PointCloud:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        np.asarray(state.geometry.vertices, dtype=np.float32),
        np.asarray(state.geometry.triangles, dtype=np.uint32),
    )

    sensor_pos = np.array(sensor_pos)

    # Sample random points on the geometry
    pc: o3d.geometry.PointCloud = state.geometry.sample_points_uniformly(
        number_of_points=npoints
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
    labels = []
    for i in range(len(hits["t_hit"])):
        primitive = hits["primitive_ids"][i]
        if primitive == o3d.t.geometry.RaycastingScene.INVALID_ID:
            continue

        label: UtilityPoleLabel = state.triangle_labels[int(primitive.item())]

        point = sensor_pos + ray_dirs[i] * float(hits["t_hit"][i].item())
        if jitter != 0.0:
            offset = np.random.normal(size=3)
            offset = offset / np.linalg.norm(offset) * jitter
            point += offset
        points.append(point)
        labels.append(label.value)

    result = o3d.t.geometry.PointCloud()
    result.point.positions = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    result.point.labels = o3d.core.Tensor(
        np.asarray(labels, dtype=np.uint32).reshape(-1, 1)
    )

    return result
