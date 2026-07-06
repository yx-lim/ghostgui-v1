"""Shared quaternion helpers used by core solvers and GUI interaction."""

from __future__ import annotations

import math

import numpy as np


def normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quaternion / norm


def quaternion_multiply(left, right):
    w1, x1, y1, z1 = normalize_quaternion(left)
    w2, x2, y2, z2 = normalize_quaternion(right)
    return normalize_quaternion(np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]))


def axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= max(1e-12, float(np.linalg.norm(axis)))
    half = 0.5 * angle
    return np.array([math.cos(half), *(axis * math.sin(half))])


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
