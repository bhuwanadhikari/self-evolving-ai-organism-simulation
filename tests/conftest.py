import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SimulationConfig  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = SimulationConfig()
    c.ticks = 50
    c.population.initial_size = 10
    c.logging.output_dir = str(tmp_path)
    c.logging.print_every = 0
    c.logging.checkpoint_every = 0
    return c
