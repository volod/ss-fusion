"""Per-object probabilistic Kalman filter for image-plane tracking.

State vector: [cx, cy, w, h, vcx, vcy]
  cx, cy  — bounding-box centre, normalised image coords [0, 1]
  w,  h   — bounding-box size, normalised [0, 1]
  vcx,vcy — centre velocity (normalised coords / frame-equivalent time unit)

Observation: [cx, cy, w, h]  (H = [I4 | 0_{4×2}])

Gating: Mahalanobis distance²  with chi-squared threshold (4 DOF, p=0.99 → 13.28).

RTS smoother history is accumulated so that the caller can retrieve smoothed
estimates in a backward pass after all frames are processed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Chi-squared threshold for 4 DOF at p = 0.99
_MAHA_GATE_4DOF = 13.277

# Minimum observation noise to avoid division issues (normalised coords)
_MIN_OBS_NOISE = 1e-6


def _bbox_to_state(bbox_norm: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_norm
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(x2 - x1, 1e-4)
    h = max(y2 - y1, 1e-4)
    return np.array([cx, cy, w, h], dtype=np.float64)


def _state_to_bbox(x: np.ndarray) -> List[float]:
    cx, cy, w, h = x[0], x[1], x[2], x[3]
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


@dataclass
class ObjectFilterHistory:
    """Forward-pass record at one time step, used by RTS smoother."""
    t_sec: float
    x: np.ndarray   # (6,) filtered
    P: np.ndarray   # (6, 6) filtered
    x_pred: np.ndarray   # (6,) before update
    P_pred: np.ndarray   # (6, 6) before update


class ObjectKalmanFilter:
    """Single-object constant-velocity Kalman filter in image coordinates.

    Attributes
    ----------
    track_id : persistent ID assigned at birth
    label    : category string (e.g. "car")
    hits     : number of successful measurement updates
    misses   : consecutive frames without a matching detection
    state    : "tentative" | "confirmed" | "deleted"
    """

    # Process noise per frame (normalised coords)
    _Q_SCALE = 0.01

    def __init__(
        self,
        track_id: int,
        label: str,
        initial_bbox: List[float],
        t_sec: float,
        obs_noise: float = 0.005,
    ) -> None:
        self.track_id = track_id
        self.label = label
        self.hits = 1
        self.misses = 0
        self.state = "tentative"
        self._obs_noise = max(obs_noise, _MIN_OBS_NOISE)
        self._history: List[ObjectFilterHistory] = []

        z = _bbox_to_state(initial_bbox)
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:4] = z
        # Initial velocity uncertainty is proportional to bbox size
        vel_var = (max(z[2], z[3]) * 0.5) ** 2
        self.P = np.diag([
            self._obs_noise ** 2,
            self._obs_noise ** 2,
            self._obs_noise ** 2,
            self._obs_noise ** 2,
            vel_var,
            vel_var,
        ]).astype(np.float64)
        self.t_sec = t_sec

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, t_sec: float) -> None:
        """Advance state to t_sec with constant-velocity model."""
        dt = max(0.0, t_sec - self.t_sec)
        F = np.eye(6, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        q = (self._Q_SCALE * max(dt, 1e-3)) ** 2
        Q = np.eye(6, dtype=np.float64) * q
        # Size dimensions have lower process noise than position
        Q[2, 2] = q * 0.25
        Q[3, 3] = q * 0.25

        self._x_pred = F @ self.x
        self._P_pred = F @ self.P @ F.T + Q
        self.x = self._x_pred.copy()
        self.P = self._P_pred.copy()
        self.t_sec = t_sec

    def update(self, bbox_norm: List[float], t_sec: float) -> None:
        """Apply a matched detection as a measurement update."""
        z = _bbox_to_state(bbox_norm)
        H = np.zeros((4, 6), dtype=np.float64)
        H[:4, :4] = np.eye(4, dtype=np.float64)
        R = np.eye(4, dtype=np.float64) * (self._obs_noise ** 2)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)

        x_pred_record = self.x.copy() if hasattr(self, "_x_pred") else self.x.copy()
        P_pred_record = self.P.copy() if hasattr(self, "_P_pred") else self.P.copy()

        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        # Joseph form for numerical stability
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self._history.append(ObjectFilterHistory(
            t_sec=t_sec,
            x=self.x.copy(),
            P=self.P.copy(),
            x_pred=x_pred_record,
            P_pred=P_pred_record,
        ))
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1

    def mahalanobis_distance_sq(self, bbox_norm: List[float]) -> float:
        """Return gated Mahalanobis distance² to a candidate detection."""
        z = _bbox_to_state(bbox_norm)
        H = np.zeros((4, 6), dtype=np.float64)
        H[:4, :4] = np.eye(4, dtype=np.float64)
        R = np.eye(4, dtype=np.float64) * (self._obs_noise ** 2)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            return float(y @ np.linalg.pinv(S) @ y)
        except np.linalg.LinAlgError:
            return float("inf")

    def is_gated(self, bbox_norm: List[float]) -> bool:
        return self.mahalanobis_distance_sq(bbox_norm) <= _MAHA_GATE_4DOF

    def predicted_bbox(self) -> List[float]:
        return _state_to_bbox(self.x)

    @property
    def history(self) -> List[ObjectFilterHistory]:
        return self._history
