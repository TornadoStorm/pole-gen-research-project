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
    geometry: Dict[o3d.geometry.TriangleMesh, UtilityPoleLabel] = {}
    intersection: bool = False
    road_presence: list[int] = [0, 0]
    main_road: int = 0
    rot_indices: list[int] = [0, 0]
    pole_base_height: float = 8.45
    pole_scale: float = 1.0
    pole_scaled_height: float = 8.45
    traffic_light_heights: list[float] = [0, 0]
    lamp_height: float = 0.0
