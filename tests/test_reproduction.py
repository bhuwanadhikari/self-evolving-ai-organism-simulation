import random

from simulation.environment import Environment
from simulation.genome import Genome
from simulation.reproduction import Reproducer, founder
from simulation.world import World


def build(cfg):
    w = World(cfg.world, cfg.resources, cfg.hazards, cfg.adaptive, random.Random(0))
    env = Environment(cfg, w, random.Random(1))
    env.tick = 100
    n = [100]

    def new_id():
        n[0] += 1
        return f"O{n[0]}"
    return env, Reproducer(cfg, env, random.Random(2), new_id)


def adult(cfg, env, oid, pos, energy=150):
    o = founder(cfg, oid, "L" + oid, pos, Genome())
    o.age, o.energy = 50, energy
    env.add(o)
    return o


def test_asexual_reproduction(cfg):
    env, rep = build(cfg)
    p = adult(cfg, env, "O1", (5, 5))
    p.wants_reproduction = {}
    births = rep.phase([p])
    assert len(births) == 1
    child = births[0][0]
    assert child.parent_ids == ["O1"] and child.generation == 1 and child.lineage_id == "LO1"
    assert p.energy == 150 - cfg.reproduction.cost and p.offspring_count == 1
    assert child.energy == cfg.reproduction.offspring_energy


def test_reproduction_rejected_without_energy(cfg):
    env, rep = build(cfg)
    p = adult(cfg, env, "O1", (5, 5), energy=30)
    p.wants_reproduction = {}
    assert rep.phase([p]) == []
    assert p.last_result["success"] is False and "energy" in p.last_result["message"]


def test_sexual_needs_mutual_consent(cfg):
    cfg.reproduction.mode = "sexual"
    env, rep = build(cfg)
    a, b = adult(cfg, env, "O1", (5, 5)), adult(cfg, env, "O2", (5, 6))
    a.wants_reproduction = {"partner": "O2"}
    assert rep.phase([a, b]) == []
    a.wants_reproduction, b.wants_reproduction = {"partner": "O2"}, {}
    births = rep.phase([a, b])
    assert len(births) == 1 and sorted(births[0][0].parent_ids) == ["O1", "O2"]


def test_memory_inheritance_modes(cfg):
    for mode, expect in [("none", 0), ("genome_only", 0), ("selected", 3), ("compressed", 1)]:
        cfg.memory.inheritance_mode, cfg.memory.inherit_count = mode, 3
        env, rep = build(cfg)
        p = adult(cfg, env, "O1", (5, 5))
        for t in range(5):
            p.memory.record(t, "consumed", "ate", {"pos": [1, 1]})
        p.wants_reproduction = {}
        child = rep.phase([p])[0][0]
        inherited = [m for m in child.memory.long_term if m.kind == "inherited"]
        assert (len(inherited) >= 1) if expect else not inherited, mode
        if mode == "selected":
            assert len(inherited) == 3


def test_population_cap(cfg):
    cfg.population.max_population = 1
    env, rep = build(cfg)
    p = adult(cfg, env, "O1", (5, 5))
    p.wants_reproduction = {}
    assert rep.phase([p]) == [] and "no room" in p.last_result["message"]
