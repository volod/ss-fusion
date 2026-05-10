"""Shared Docker image reference catalog."""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class DockerImageRef:
    env_var: str
    default: str = ""

    def describe(self) -> dict[str, str]:
        return {
            "env_image_var": self.env_var,
            "default_image": self.default,
        }


REALTIME_ENGINE_IMAGES: Final[dict[str, DockerImageRef]] = {
    "vins_fusion": DockerImageRef("REALTIME_VINS_FUSION_IMAGE"),
    "orbslam3": DockerImageRef("REALTIME_ORBSLAM3_IMAGE"),
    "liosam": DockerImageRef("REALTIME_LIOSAM_IMAGE"),
    "nvblox": DockerImageRef("REALTIME_NVBLOX_IMAGE"),
    "voxblox": DockerImageRef("REALTIME_VOXBLOX_IMAGE"),
}
