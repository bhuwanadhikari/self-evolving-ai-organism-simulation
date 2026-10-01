"""Closed artificial-life simulator. Everything here is virtual: no network,
no filesystem access for organisms, no real-world side effects."""

from .config import SimulationConfig, load_config

__all__ = ["SimulationConfig", "load_config"]
