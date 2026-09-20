"""Sensor adapter framework for site state API v1."""

from .base import SensorAdapter
from .registry import registry

__all__ = ["SensorAdapter", "registry"]
