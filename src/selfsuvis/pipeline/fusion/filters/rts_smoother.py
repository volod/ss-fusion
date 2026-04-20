"""Rauch-Tung-Striebel (RTS) fixed-interval smoother.

Operates on *frame-sampled* posterior states from the forward Kalman pass.
The predicted covariance between consecutive frames is approximated using the
constant-velocity motion model and the frame timestamps — this is exact when
no measurements arrive between frames and a very good approximation otherwise
(measurements shrink P, so the smoothed gain is conservative).

Usage::

    from selfsuvis.pipeline.fusion.filters.rts_smoother import rts_smooth, FilteredStep

    steps = [FilteredStep(t_sec=t, x=x, P=P) for t, x, P in forward_pass]
    smoothed = rts_smooth(steps, process_pos_std, process_vel_std)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class FilteredStep:
    """Posterior state record from the forward Kalman pass at one frame timestamp."""
    t_sec: float
    x: np.ndarray   # (6,) [px, py, pz, vx, vy, vz]
    P: np.ndarray   # (6, 6)


@dataclass
class SmoothedStep:
    t_sec: float
    x: np.ndarray   # (6,) smoothed state
    P: np.ndarray   # (6, 6) smoothed covariance
    cov_trace: float


def _build_cv_matrices(
    dt: float,
    process_pos_std: float,
    process_vel_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (F, Q) for a constant-velocity model over interval dt."""
    F = np.eye(6, dtype=np.float64)
    F[:3, 3:] = np.eye(3, dtype=np.float64) * dt
    dt_eff = max(dt, 1e-3)
    q_pos = (process_pos_std * dt_eff) ** 2
    q_vel = (process_vel_std * dt_eff) ** 2
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)
    return F, Q


def rts_smooth(
    history: Sequence[FilteredStep],
    process_pos_std: float,
    process_vel_std: float,
) -> List[SmoothedStep]:
    """Run the RTS backward pass over forward-filtered frame states.

    Args:
        history: Filtered states at frame timestamps, ordered ascending by t_sec.
        process_pos_std: Process noise for position (m), same as forward filter.
        process_vel_std: Process noise for velocity (m/s), same as forward filter.

    Returns:
        List of SmoothedStep at the same timestamps (same length as history).
    """
    n = len(history)
    if n == 0:
        return []

    xs = [h.x.copy() for h in history]
    Ps = [h.P.copy() for h in history]
    ts = [h.t_sec for h in history]

    # Backward pass
    xs_smooth = [None] * n
    Ps_smooth = [None] * n
    xs_smooth[-1] = xs[-1].copy()
    Ps_smooth[-1] = Ps[-1].copy()

    for k in range(n - 2, -1, -1):
        dt = max(0.0, ts[k + 1] - ts[k])
        F, Q = _build_cv_matrices(dt, process_pos_std, process_vel_std)

        # Predicted state/covariance from k to k+1 (using posterior at k)
        x_pred = F @ xs[k]
        P_pred = F @ Ps[k] @ F.T + Q

        # Smoother gain
        G = Ps[k] @ F.T @ np.linalg.pinv(P_pred)

        xs_smooth[k] = xs[k] + G @ (xs_smooth[k + 1] - x_pred)
        P_s = Ps[k] + G @ (Ps_smooth[k + 1] - P_pred) @ G.T
        Ps_smooth[k] = 0.5 * (P_s + P_s.T)  # symmetry

    return [
        SmoothedStep(
            t_sec=ts[i],
            x=xs_smooth[i],
            P=Ps_smooth[i],
            cov_trace=float(np.trace(Ps_smooth[i])),
        )
        for i in range(n)
    ]
