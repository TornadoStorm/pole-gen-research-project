from enum import Enum
from typing import List

import numpy as np
import open3d as o3d


class UtilityPoleLabel(Enum):
    UNLABELED = 0
    POLE = 1
    LAMP = 2
    CROSSARM = 3
    TRANSFORMER = 4
    SIGN = 5
    TRAFFIC_LIGHT = 6
    PEDESTRIAN_SIGNAL = 7


class State:
    geometry: o3d.geometry.TriangleMesh
    triangle_labels: np.ndarray
    intersection: bool
    road_presence: list[int]
    main_road: int
    rot_indices: list[int]
    pole_base_radius: float
    pole_top_radius: float
    pole_base_height: float
    pole_scale: float
    pole_scaled_height: float
    traffic_light_heights: list[float]
    lamp_height: float

    def __init__(self):
        self.geometry = o3d.geometry.TriangleMesh()
        self.triangle_labels = np.array([], dtype=object)
        self.intersection = False
        self.road_presence = [0, 0]
        self.main_road = 0
        self.rot_indices = [0, 0]
        self.pole_base_radius = 0.143
        self.pole_top_radius = 0.0895
        self.pole_base_height = 8.45
        self.pole_scale = 1.0
        self.pole_scaled_height = 8.45
        self.traffic_light_heights = [0, 0]
        self.lamp_height = 0.0

    def pole_radius_at(self, height: float) -> float:
        return np.interp(
            height,
            [0, self.pole_scaled_height],
            [self.pole_top_radius, self.pole_base_radius],
        )

    def add_geometry(self, mesh: o3d.geometry.TriangleMesh, label: UtilityPoleLabel):
        self.geometry += mesh
        new_labels = np.full(len(mesh.triangles), label, dtype=object)
        self.triangle_labels = np.concatenate((self.triangle_labels, new_labels))
