import numpy as np


def signed_angle_difference(angle1: float, angle2: float) -> float:
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi
