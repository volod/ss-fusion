
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PlatformMeasurement:
    """Typed measurement used by the platform-state fusion subsystem."""

    kind: str
    t_sec: float
    values: Tuple[float, ...]
    covariance: Tuple[Tuple[float, ...], ...]
    source: str = ""
    frame: str = "enu"
    quality: str = "nominal"

    def covariance_matrix(self) -> np.ndarray:
        return np.array(self.covariance, dtype=np.float64)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "t_sec": self.t_sec,
            "values": list(self.values),
            "covariance": [list(row) for row in self.covariance],
            "source": self.source,
            "frame": self.frame,
            "quality": self.quality,
        }
