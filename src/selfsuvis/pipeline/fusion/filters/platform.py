from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from selfsuvis.pipeline.fusion.measurements import PlatformMeasurement


@dataclass
class PlatformStateFilter:
    """Constant-velocity Kalman filter with optional acceleration control."""

    process_pos_std_m: float
    process_vel_std_mps: float
    init_vel_std_mps: float
    x: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    t_sec: Optional[float] = None
    current_accel_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    innovation_norms: List[float] = field(default_factory=list)
    measurement_counts: Dict[str, int] = field(default_factory=dict)

    def is_initialized(self) -> bool:
        return self.x is not None and self.P is not None and self.t_sec is not None

    def initialize_from_position(self, t_sec: float, position_enu_m: np.ndarray, position_cov: np.ndarray) -> None:
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = position_enu_m
        vel_var = self.init_vel_std_mps ** 2
        self.P = np.zeros((6, 6), dtype=np.float64)
        self.P[:3, :3] = position_cov
        self.P[3:, 3:] = np.eye(3, dtype=np.float64) * vel_var
        self.t_sec = t_sec

    def predict(self, to_t_sec: float) -> None:
        if not self.is_initialized():
            return
        assert self.x is not None and self.P is not None and self.t_sec is not None
        dt = max(0.0, float(to_t_sec - self.t_sec))
        if dt <= 0.0:
            self.t_sec = to_t_sec
            return

        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        B = np.zeros((6, 3), dtype=np.float64)
        B[:3, :] = 0.5 * np.eye(3, dtype=np.float64) * dt * dt
        B[3:, :] = np.eye(3, dtype=np.float64) * dt

        q_pos = (self.process_pos_std_m * max(dt, 1e-3)) ** 2
        q_vel = (self.process_vel_std_mps * max(dt, 1e-3)) ** 2
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)

        self.x = F @ self.x + B @ self.current_accel_enu
        self.P = F @ self.P @ F.T + Q
        self.t_sec = to_t_sec

    def set_acceleration(self, measurement: PlatformMeasurement) -> None:
        self.current_accel_enu = np.array(measurement.values, dtype=np.float64)
        self.measurement_counts[measurement.kind] = self.measurement_counts.get(measurement.kind, 0) + 1

    def update_position(self, measurement: PlatformMeasurement) -> None:
        if not self.is_initialized():
            self.initialize_from_position(
                measurement.t_sec,
                np.array(measurement.values, dtype=np.float64),
                measurement.covariance_matrix(),
            )
            self.measurement_counts[measurement.kind] = self.measurement_counts.get(measurement.kind, 0) + 1
            return

        assert self.x is not None and self.P is not None
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, :3] = np.eye(3, dtype=np.float64)
        z = np.array(measurement.values, dtype=np.float64)
        R = measurement.covariance_matrix()
        self._update(H, z, R, measurement.kind)

    def update_altitude(self, measurement: PlatformMeasurement) -> None:
        if not self.is_initialized():
            return
        assert self.x is not None and self.P is not None
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 2] = 1.0
        z = np.array([measurement.values[0]], dtype=np.float64)
        R = measurement.covariance_matrix()
        self._update(H, z, R, measurement.kind)

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray, kind: str) -> None:
        assert self.x is not None and self.P is not None
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + (K @ y)
        I = np.eye(self.P.shape[0], dtype=np.float64)
        self.P = (I - K @ H) @ self.P
        self.innovation_norms.append(float(np.linalg.norm(y)))
        self.measurement_counts[kind] = self.measurement_counts.get(kind, 0) + 1
