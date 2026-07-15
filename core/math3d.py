"""Small Qt-free math helpers shared by core logic."""

from __future__ import annotations

import math

import numpy as np


def normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quaternion / norm


def quaternion_slerp(start, end, fraction):
    start = normalize_quaternion(start)
    end = normalize_quaternion(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(start + fraction * (end - start))
    theta = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * theta) * start
        + math.sin(fraction * theta) * end
    ) / math.sin(theta)
