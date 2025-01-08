from enum import Enum
from typing import List, Tuple

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


LABEL_COLORS = [
    [0.5, 0.5, 0.5],  # UNLABELED
    [1.0, 0.0, 0.0],  # POLE
    [0.0, 1.0, 0.0],  # LAMP
    [0.0, 0.0, 1.0],  # CROSSARM
    [1.0, 1.0, 0.0],  # TRANSFORMER
    [1.0, 0.0, 1.0],  # SIGN
    [0.0, 1.0, 1.0],  # TRAFFIC_LIGHT
    [0.5, 0.5, 1.0],  # PEDESTRIAN_SIGNAL
]


class Placement:
    z_position: float
    height: float | None
    z_rotation: float

    def __init__(
        self, z_position: float, z_rotation: float, height: float | None = None
    ):
        self.z_position = z_position
        self.height = height
        self.z_rotation = z_rotation

        while self.z_rotation < 0:
            self.z_rotation += 2 * np.pi

        while self.z_rotation >= 2 * np.pi:
            self.z_rotation -= 2 * np.pi


class State:
    geometry: o3d.geometry.TriangleMesh
    triangle_labels: np.ndarray
    is_intersection: bool
    road_presence: list[int]
    main_road: int
    rot_indices: list[int]
    pole_base_radius: float
    pole_top_radius: float
    pole_base_height: float
    pole_scale: float
    pole_scaled_height: float
    traffic_light_heights: list[float]
    pedestrian_signal_heights: list[float]
    lamp_height: float
    z_rotation: float
    side_signs: List[Placement]
    normal_signs: List[Placement]

    def __init__(self):
        self.geometry = o3d.geometry.TriangleMesh()
        self.triangle_labels = np.array([], dtype=object)
        self.is_intersection = False
        self.road_presence = [0, 0]
        self.main_road = 0
        self.rot_indices = [0, 0]
        self.pole_base_radius = 0.143
        self.pole_top_radius = 0.0895
        self.pole_base_height = 8.45
        self.pole_scale = 1.0
        self.pole_scaled_height = 8.45
        self.traffic_light_heights = [0, 0]
        self.pedestrian_signal_heights = [0, 0]
        self.lamp_height = 0.0
        self.z_rotation = 0.0
        self.side_signs = []
        self.normal_signs = []

    def pole_radius_at(self, height: float) -> float:
        return np.interp(
            height,
            [0, self.pole_scaled_height],
            [self.pole_base_radius, self.pole_top_radius],
        )

    def add_geometry(self, mesh: o3d.geometry.TriangleMesh, label: UtilityPoleLabel):
        self.geometry += mesh
        new_labels = np.full(len(mesh.triangles), label, dtype=object)
        self.triangle_labels = np.concatenate((self.triangle_labels, new_labels))
