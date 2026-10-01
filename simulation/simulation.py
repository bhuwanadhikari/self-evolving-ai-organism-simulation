"""Main simulation loop.

Each tick:
  1. environment.update()              resources regenerate, hotspots/hazards move
  2. observe every living organism     (shuffled order, seeded)
  3. brain.decide_batch()              only for organisms that need a new decision
  4. environment.execute()             validate + apply actions sequentially
  5. memory updates                    (done inside execute / physiology)
  6. physiology                        metabolism, hazard damage, healing, ageing
  7. reproduction phase                incl. crossover + mutation (variation)
  8. death phase
  9. metrics, snapshots, checkpoint

Decisions are taken simultaneously from the start-of-tick observation and
resolved sequentially in a random (seeded) order, so no organism always
moves first.

Determinism: all randomness flows from named random.Random streams derived
from the seed. With a non-LLM brain, the same seed and config give the same
run bit-for-bit. LLM outputs are the only non-deterministic component.
"""

from __future__ import annotations

import json
import pickle
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.metrics import MetricsCollector
from brains import Brain, DecisionContext, make_brain

from .actions import REPEATABLE, Action, available_actions, parse_action
from .config import ENVIRONMENT_VERSION, SIMULATION_VERSION, SimulationConfig
from .environment import Environment
from .genome import Genome
from .logging_utils import RunLogger, truncate_after
from .organism import Organism
from .reproduction import Reproducer, birth_record, founder
from .world import World

RNG_STREAMS = ("world", "env", "order", "brain", "evolution", "init")


def needs_decision(org: Organism, obs: dict[str, Any], cfg: SimulationConfig, tick: int) -> bool:
    """LLM cost control: re-use the previous action unless something changed."""
    dp = cfg.brain.decision
    if dp.every_tick or org.last_decision is None:
        return True
    last = org.last_decision
    if last["action"] not in REPEATABLE:
        return True
    if org.last_result and not org.last_result.get("success", True):
        return True
    if tick - org.last_decision_tick >= dp.max_interval:
        return True
    if abs(org.energy - org.last_decision_energy) >= dp.energy_change_threshold:
        return True
    if abs(org.health - org.last_decision_health) >= dp.health_change_threshold:
        return True
    if obs["messages"] or obs["location"]["hazard_level"] > 0:
        return True
    if last["action"] == "consume" and obs["location"]["resources"] == 0:
        return True
    if tuple(e["id"] for e in obs["nearby_entities"]) != org.last_neighbors:
        return True
    return False


class Simulation:
    def __init__(self, cfg: SimulationConfig, run_dir: str | Path | None = None, brain: Brain | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir or Path(cfg.logging.output_dir) / f"{cfg.name}_seed{cfg.seed}")
        self.rngs = {s: random.Random(f"{cfg.seed}-{s}") for s in RNG_STREAMS}
        self.tick = 0
        self.next_id = 0
        self.world = World(cfg.world, cfg.resources, cfg.hazards, cfg.adaptive, self.rngs["world"])
        self.env = Environment(cfg, self.world, self.rngs["env"])
        self.env.tick = 0
        self.reproducer = Reproducer(cfg, self.env, self.rngs["evolution"], self._new_id)
        self.metrics = MetricsCollector()
        self.allowed = available_actions(cfg)
        self.finished = False
        self.brain = brain or make_brain(cfg.brain)
        self.logger = RunLogger(self.run_dir)
        self._write_metadata()
        self._spawn_founders()

    # ---- pickling (checkpoints) ---------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("brain", None)
        state.pop("logger", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.brain = None  # type: ignore[assignment]
        self.logger = None  # type: ignore[assignment]

    def _new_id(self) -> str:
        self.next_id += 1
        return f"O{self.next_id}"

    # ---- setup --------------------------------------------------------------------------
    def _write_metadata(self) -> None:
        brain_desc = self.brain.describe()
        meta = {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "description": self.cfg.description,
            "hypothesis": self.cfg.hypothesis,
            "simulation_version": SIMULATION_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "brain": brain_desc,
            "model_version": brain_desc.get("model", brain_desc.get("name")),
            "prompt_version": brain_desc.get("prompt_version"),
            "allowed_actions": self.allowed,
            "python": platform.python_version(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": self.cfg.to_dict(),
        }
        self.logger.write_json("simulation.json", meta)

    def _spawn_founders(self) -> None:
        pc, rng = self.cfg.population, self.rngs["init"]
        for i in range(pc.initial_size):
            genome = Genome.random(rng) if pc.initial_genome == "random" else Genome.from_dict(pc.fixed_genome)
            pos = (rng.randrange(self.world.w), rng.randrange(self.world.h))
            org = founder(self.cfg, self._new_id(), f"L{i + 1}", pos, genome)
            self.env.add(org)
            self.logger.write("genomes", birth_record(org, [], [], 0))
        self._snapshot()

    # ---- main loop -------------------------------------------------------------------------
    def run(self, ticks: int | None = None) -> dict[str, Any]:
        target = ticks if ticks is not None else self.cfg.ticks
        t0 = time.time()
        try:
            while self.tick < target and not self.finished:
                self.step()
        finally:
            self.save_checkpoint()
            summary = self.summary(time.time() - t0)
            self.logger.write_json("summary.json", summary)
            self.logger.write_json("organisms.json", [o.record_summary() for o in self.env.organisms.values()])
            self.logger.flush()
        return summary

    def step(self) -> None:
        cfg, env = self.cfg, self.env
        self.tick += 1
        env.tick = self.tick
        env.comm_log, env.event_log = [], []

        # 1. Environment update.
        occ = {pos: len(ids) for pos, ids in env.occupancy.items()}
        self.world.update(occ)
        for ev in self.world.events:
            self.logger.write("events", {"tick": self.tick, "type": "environment", **ev})

        # 2. Order + observations.
        order = sorted(env.alive(), key=lambda o: int(o.id[1:]))
        self.rngs["order"].shuffle(order)
        observations = {o.id: env.observe(o) for o in order}

        # 3. Decisions (batched, only where needed).
        to_decide = [o for o in order if needs_decision(o, observations[o.id], cfg, self.tick)]
        ctxs = [DecisionContext(o.id, observations[o.id],
                                [m.to_dict() for m in o.memory.retrieve(cfg.memory.retrieval_count)],
                                o.genome.to_dict(), self.allowed) for o in to_decide]
        proposals = self.brain.decide_batch(ctxs, self.rngs["brain"]) if ctxs else []
        decisions: dict[str, Action] = {}
        for o, prop in zip(to_decide, proposals):
            if prop.get("_raw") is not None:
                o.stats["llm_calls"] += 1
                self.logger.write("brain_calls", {"tick": self.tick, "organism": o.id,
                                                  **{k.lstrip("_"): v for k, v in prop.items()}})
            clean = {k: v for k, v in prop.items() if not k.startswith("_")}
            action = parse_action(clean, self.allowed)
            if prop.get("_parse_error") and not action.parse_error:
                action.parse_error = prop["_parse_error"]
            decisions[o.id] = action
            o.last_decision = action.to_dict()
            o.last_decision_tick = self.tick
            o.last_decision_energy, o.last_decision_health = o.energy, o.health
            o.last_neighbors = tuple(e["id"] for e in observations[o.id]["nearby_entities"])
        cached = 0
        for o in order:
            if o.id not in decisions:
                a = parse_action(o.last_decision, self.allowed)
                a.from_cache = True
                decisions[o.id] = a
                cached += 1

        # 4. Execute sequentially.
        action_counts: Counter = Counter()
        failed = 0
        for o in order:
            if o.energy <= 0 or o.health <= 0:
                continue  # died earlier this tick (e.g. attacked); resolved in death phase
            a = decisions[o.id]
            e_before, h_before = o.energy, o.health
            res = env.execute(o, a)
            action_counts[a.name] += 1
            failed += 0 if res.success else 1
            if cfg.logging.log_actions:
                self.logger.write("events", {
                    "tick": self.tick, "type": "action", "organism": o.id, "action": a.name, "args": a.args,
                    "success": res.success, "result": res.message, "energy_before": round(e_before, 2),
                    "energy_after": round(o.energy, 2), "health_before": round(h_before, 2),
                    "position": list(o.position), "reason": a.reason, "cached": a.from_cache,
                    **({"parse_error": a.parse_error} if a.parse_error else {})})
        for rec in env.comm_log:
            self.logger.write("communications", rec)
        for rec in env.event_log:
            self.logger.write("events", rec)

        # 6. Physiology.
        for o in order:
            if o.alive:
                env.update_state(o)

        # 7. Reproduction + mutation.
        births = self.reproducer.phase(order)
        mutations = 0
        for child, parents, mutated, inherited in births:
            mutations += len(mutated)
            rec = birth_record(child, parents, mutated, inherited)
            self.logger.write("genomes", rec)
            self.logger.write("events", {"tick": self.tick, "type": "birth", "organism": child.id,
                                         "parents": child.parent_ids, "generation": child.generation,
                                         "lineage_id": child.lineage_id, "mutated_traits": mutated})

        # 8. Death.
        deaths = []
        for o in sorted(env.alive(), key=lambda o: int(o.id[1:])):
            cause = env.death_cause(o)
            if cause:
                env.kill(o, cause)
                deaths.append(o)
                self.logger.write("events", {"tick": self.tick, "type": "death", "organism": o.id,
                                             "cause": cause, "age": o.age, "generation": o.generation,
                                             "lineage_id": o.lineage_id, "offspring": o.offspring_count})

        # 9. Metrics, snapshots, checkpoints.
        alive = env.alive()
        row = self.metrics.row(self.tick, alive, [b[0] for b in births], deaths, action_counts, failed,
                               self.world.total_resources(), mutations, len(to_decide) if ctxs else 0, cached)
        self.logger.metrics(row)
        lc = cfg.logging
        if lc.snapshot_every and self.tick % lc.snapshot_every == 0:
            self._snapshot()
        if lc.checkpoint_every and self.tick % lc.checkpoint_every == 0:
            self.save_checkpoint()
        if lc.print_every and self.tick % lc.print_every == 0:
            print(f"[{cfg.name}] tick {self.tick:5d}  pop {row['population']:4d}  births {self.metrics.total_births:5d}"
                  f"  deaths {self.metrics.total_deaths:5d}  gen {row['max_generation']:3d}"
                  f"  energy {row['avg_energy']:6.1f}  diversity {row['genome_diversity']:.3f}", flush=True)
        if not alive:
            self.finished = True
            self.logger.write("events", {"tick": self.tick, "type": "extinction"})
            self._snapshot()
            print(f"[{cfg.name}] extinction at tick {self.tick}")

    def _snapshot(self) -> None:
        snap = {"tick": self.tick, **self.world.snapshot(),
                "organisms": [{"id": o.id, "x": o.position[0], "y": o.position[1], "lineage": o.lineage_id,
                               "energy": round(o.energy, 1), "health": round(o.health, 1),
                               "generation": o.generation} for o in self.env.alive()]}
        self.logger.write("snapshots", snap)

    # ---- checkpoints -----------------------------------------------------------------------
    def save_checkpoint(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(exist_ok=True)
        path = d / f"tick_{self.tick:07d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        (d / "latest.txt").write_text(path.name)
        self.logger.flush()
        return path

    @classmethod
    def resume(cls, run_dir: str | Path, checkpoint: str | None = None, brain: Brain | None = None) -> "Simulation":
        run_dir = Path(run_dir)
        d = run_dir / "checkpoints"
        name = checkpoint or (d / "latest.txt").read_text().strip()
        with open(d / name, "rb") as f:
            sim: Simulation = pickle.load(f)
        sim.run_dir = run_dir
        truncate_after(run_dir, sim.tick)
        sim.brain = brain or make_brain(sim.cfg.brain)
        sim.logger = RunLogger(run_dir, append=True)
        return sim

    def summary(self, wall_time: float) -> dict[str, Any]:
        alive = self.env.alive()
        all_orgs = list(self.env.organisms.values())
        lineages = Counter(o.lineage_id for o in alive)
        return {
            "ticks": self.tick,
            "extinct": not alive,
            "final_population": len(alive),
            "organisms_ever": len(all_orgs),
            "total_births": self.metrics.total_births,
            "total_deaths": self.metrics.total_deaths,
            "max_generation": self.metrics.max_generation,
            "surviving_lineages": dict(lineages.most_common()),
            "death_causes": dict(Counter(o.death_cause for o in all_orgs if not o.alive)),
            "brain_calls": sum(o.stats["llm_calls"] for o in all_orgs),
            "wall_time_s": round(wall_time, 2),
        }


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "summary.json").read_text())
