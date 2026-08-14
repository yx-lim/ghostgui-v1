"""MuJoCo configuration-space trajectory helpers."""

from __future__ import annotations

import mujoco
import numpy as np


def interpolate_qpos_manifold(mj_model, start, end, fraction):
    """Interpolate two qpos vectors through the MuJoCo position manifold.

    Ball and free joints store unit quaternions in ``qpos``.  Blending their
    four entries independently can leave the unit sphere and take the long
    rotation path.  MuJoCo instead differentiates the endpoints into the
    model's tangent space and integrates the requested fraction back onto the
    position manifold.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    expected_shape = (int(mj_model.nq),)
    if start.shape != expected_shape or end.shape != expected_shape:
        raise ValueError(
            "qpos interpolation endpoints must both have shape "
            f"{expected_shape}"
        )
    fraction = float(fraction)
    if not np.isfinite(fraction):
        raise ValueError("qpos interpolation fraction must be finite")

    displacement = np.zeros(int(mj_model.nv), dtype=float)
    mujoco.mj_differentiatePos(
        mj_model, displacement, 1.0, start, end
    )
    result = start.copy()
    mujoco.mj_integratePos(
        mj_model, result, displacement, fraction
    )
    return result
