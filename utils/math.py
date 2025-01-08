import numpy as np


def signed_angle_difference(angle1: float, angle2: float) -> float:
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi


def ranges_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)
