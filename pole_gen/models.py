from enum import Enum
from typing import Dict

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
    geometry: Dict[o3d.geometry.TriangleMesh, UtilityPoleLabel]
    intersection: bool
    road_presence: list[int]
    main_road: int
    rot_indices: list[int]
    pole_base_height: float
    pole_scale: float
    pole_scaled_height: float
    traffic_light_heights: list[float]
    lamp_height: float

    def __init__(self):
        self.geometry = {}
        self.intersection = False
        self.road_presence = [0, 0]
        self.main_road = 0
        self.rot_indices = [0, 0]
        self.pole_base_height = 8.45
        self.pole_scale = 1.0
        self.pole_scaled_height = 8.45
        self.traffic_light_heights = [0, 0]
        self.lamp_height = 0.0
