import random

from simulation.config import AdaptiveConfig, HazardConfig, ResourceConfig, WorldConfig
from simulation.world import World


def make(**rc):
    return World(WorldConfig(5, 4), ResourceConfig(**rc), HazardConfig(), AdaptiveConfig(), random.Random(0))


def test_initial_resources_and_cap():
    w = make(initial_count=30, max_per_cell=2)
    assert w.total_resources() == 30
    assert max(max(r) for r in w.resources) <= 2


def test_regeneration_respects_cap():
    w = make(initial_count=0, regeneration_rate=1.0, max_per_cell=3)
    for _ in range(10):
        w.update({})
    assert w.total_resources() == 5 * 4 * 3


def test_take():
    w = make(initial_count=0)
    w.resources[1][1] = 3
    assert w.take(1, 1, 2) == 2 and w.resources[1][1] == 1
    assert w.take(1, 1, 5) == 1 and w.take(1, 1, 5) == 0


def test_edges_and_wrap():
    w = make()
    assert w.step((0, 0), "W") is None
    assert w.step((0, 0), "E") == (1, 0)
    ww = World(WorldConfig(5, 4, wrap=True), ResourceConfig(), HazardConfig(), AdaptiveConfig(), random.Random(0))
    assert ww.step((0, 0), "W") == (4, 0)
    assert ww.distance((0, 0), (4, 0)) == 1


def test_hazard_zones():
    w = World(WorldConfig(10, 10), ResourceConfig(), HazardConfig(zone_count=1, max_level=1.0), AdaptiveConfig(),
              random.Random(0))
    hx, hy = w.hazard_zones[0]
    assert w.hazard(hx, hy) == 1.0
