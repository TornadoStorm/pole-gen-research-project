class State:
    intersection: bool = False
    road_presence: list[int] = [0, 0]
    main_road: int = 0
    rot_indices: list[int] = [0, 0]
    pole_base_height: float = 8.45
    pole_scale: float = 1.0
    pole_scaled_height: float = 8.45
    traffic_light_heights: list[float] = [0, 0]
    lamp_height: float = 0.0
