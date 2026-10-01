import random

from simulation.actions import available_actions, parse_action
from simulation.environment import Environment
from simulation.genome import Genome
from simulation.reproduction import founder
from simulation.world import World


def setup_env(cfg):
    cfg.resources.initial_count = 0
    w = World(cfg.world, cfg.resources, cfg.hazards, cfg.adaptive, random.Random(0))
    env = Environment(cfg, w, random.Random(1))
    return env


def org(cfg, env, oid, pos):
    o = founder(cfg, oid, "L" + oid, pos, Genome())
    env.add(o)
    return o


def test_parse_rejects_unknown_and_bad_args(cfg):
    allowed = available_actions(cfg)
    assert parse_action({"action": "hack_server"}, allowed).name == "invalid"
    assert parse_action({"action": "move", "args": {"direction": "up"}}, allowed).name == "invalid"
    assert parse_action({"action": "move", "args": {"direction": "north"}}, allowed).args == {"direction": "N"}
    assert parse_action("rest", allowed).name == "invalid"
    assert "attack" not in allowed and "share" not in allowed


def test_move_edge_and_cost(cfg):
    env = setup_env(cfg)
    o = org(cfg, env, "O1", (0, 0))
    r = env.execute(o, parse_action({"action": "move", "args": {"direction": "W"}}, available_actions(cfg)))
    assert not r.success and "edge" in r.message
    r = env.execute(o, parse_action({"action": "move", "args": {"direction": "E"}}, available_actions(cfg)))
    assert r.success and o.position == (1, 0)
    assert o.energy == cfg.organism.initial_energy - 2 * cfg.organism.move_cost


def test_consume(cfg):
    env = setup_env(cfg)
    o = org(cfg, env, "O1", (2, 2))
    a = parse_action({"action": "consume"}, available_actions(cfg))
    assert not env.execute(o, a).success
    env.world.resources[2][2] = 5
    e0 = o.energy
    assert env.execute(o, a).success
    assert env.world.resources[2][2] == 5 - cfg.organism.consume_amount
    assert o.energy > e0


def test_communication_range_and_log(cfg):
    env = setup_env(cfg)
    a, b = org(cfg, env, "O1", (0, 0)), org(cfg, env, "O2", (10, 10))
    act = parse_action({"action": "communicate", "args": {"target": "O2", "message": "hi"}}, available_actions(cfg))
    assert not env.execute(a, act).success
    b.position = (1, 1)
    assert env.execute(a, act).success
    assert b.inbox[0]["message"] == "hi" and env.comm_log[-1]["to"] == "O2"


def test_share_and_attack(cfg):
    cfg.actions.share_enabled = cfg.actions.attack_enabled = True
    env = setup_env(cfg)
    a, b = org(cfg, env, "O1", (3, 3)), org(cfg, env, "O2", (3, 4))
    allowed = available_actions(cfg)
    env.execute(a, parse_action({"action": "share", "args": {"target": "O2", "amount": 20}}, allowed))
    assert b.energy == cfg.organism.initial_energy + 20
    env.execute(a, parse_action({"action": "attack", "args": {"target": "O2"}}, allowed))
    assert b.health == cfg.organism.initial_health - cfg.actions.attack_damage
    assert any(m.kind == "attacked_by" for m in b.memory.short_term)


def test_metabolism_and_death(cfg):
    env = setup_env(cfg)
    o = org(cfg, env, "O1", (0, 0))
    o.energy = 0.5
    env.update_state(o)
    assert env.death_cause(o) == "starvation"
    o.energy, o.health = 50, 0
    assert env.death_cause(o) == "injury"
    o.health, o.age = 50, cfg.organism.max_age
    assert env.death_cause(o) == "old_age"
