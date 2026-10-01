"""Reproduction, variation and memory inheritance.

Asexual:  parent genome -> copy -> mutation -> child genome
Sexual:   parent A + parent B -> crossover -> mutation -> child genome

Both parents must have chosen `reproduce` in the same tick (mutual consent)
in sexual mode. There is no fitness function: whoever manages to meet the
energetic requirements reproduces.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .config import SimulationConfig
from .environment import Environment
from .genome import Genome
from .memory import EpisodicMemory, MemoryItem
from .organism import Organism


def make_memory(cfg: SimulationConfig, genome: Genome) -> EpisodicMemory:
    mc = cfg.memory
    cap = mc.long_term_capacity
    if cfg.organism.genome_physical_effects:
        cap = int(round(cap * genome.memory_capacity_multiplier()))
    return EpisodicMemory(mc.short_term_capacity, cap, mc.promote_threshold)


def inherit_memory(cfg: SimulationConfig, child: Organism, parents: list[Organism], tick: int) -> int:
    mode, k = cfg.memory.inheritance_mode, cfg.memory.inherit_count
    n = 0
    if mode == "selected":
        for p in parents:
            for m in p.memory.select_for_offspring(k // len(parents) or 1):
                child.memory.store_long(MemoryItem(tick, "inherited", m.text, m.importance,
                                                   {**m.data, "origin": p.id, "origin_tick": m.tick}))
                n += 1
    elif mode == "compressed":
        for p in parents:
            for fact in p.memory.compress(k // len(parents) or 1):
                child.memory.store_long(MemoryItem(tick, "inherited", fact, 0.8, {"origin": p.id}))
                n += 1
    # "none" and "genome_only": no memory is transmitted.
    return n


class Reproducer:
    def __init__(self, cfg: SimulationConfig, env: Environment, rng: random.Random,
                 new_id: Callable[[], str]):
        self.cfg, self.env, self.rng, self.new_id = cfg, env, rng, new_id

    def _eligible(self, org: Organism, share: float) -> str | None:
        rc = self.cfg.reproduction
        if not rc.enabled:
            return "reproduction is not possible"
        if org.age < rc.min_age:
            return f"too young (age {org.age} < {rc.min_age})"
        if self.env.tick - org.last_reproduction_tick < rc.cooldown:
            return "not ready again yet"
        need = max(rc.min_energy * share, rc.cost * share)
        if org.energy < need:
            return f"not enough energy (need {need:.0f})"
        return None

    def phase(self, order: list[Organism]) -> list[tuple[Organism, list[Organism], list[str], int]]:
        """Resolve all reproduction intents registered this tick.

        Returns a list of (child, parents, mutated_traits, inherited_memories).
        """
        rc = self.cfg.reproduction
        births = []
        requesters = [o for o in order if o.alive and o.wants_reproduction is not None]
        done: set[str] = set()
        for org in requesters:
            if org.id in done:
                continue
            req = org.wants_reproduction or {}
            if len(self.env.alive()) >= self.cfg.population.max_population:
                self._fail(org, "no room for offspring")
                continue
            if rc.mode == "asexual":
                err = self._eligible(org, 1.0)
                if err:
                    self._fail(org, err)
                    continue
                org.energy -= rc.cost
                parents = [org]
            else:
                err = self._eligible(org, 0.5)
                if err:
                    self._fail(org, err)
                    continue
                partner = self._find_partner(org, req.get("partner"), requesters, done)
                if partner is None:
                    self._fail(org, "no willing partner nearby")
                    continue
                org.energy -= rc.cost / 2
                partner.energy -= rc.cost / 2
                parents = [org, partner]
            child, mutated, inherited = self._make_child(parents)
            for p in parents:
                done.add(p.id)
                p.offspring_count += 1
                p.last_reproduction_tick = self.env.tick
                p.memory.record(self.env.tick, "reproduced", f"produced offspring {child.id}", {"other": child.id})
                p.last_result = {"action": "reproduce", "success": True,
                                 "message": f"offspring {child.id} was born", "energy_delta": 0.0}
            births.append((child, parents, mutated, inherited))
        for o in requesters:
            o.wants_reproduction = None
        return births

    def _find_partner(self, org: Organism, wanted: str | None, requesters: list[Organism],
                      done: set[str]) -> Organism | None:
        rc = self.cfg.reproduction
        cands = []
        for o in requesters:
            if o.id == org.id or o.id in done or not o.alive:
                continue
            if self.env.world.distance(org.position, o.position) > rc.mate_radius:
                continue
            if self._eligible(o, 0.5):
                continue
            cands.append(o)
        if wanted:
            for o in cands:
                if o.id == wanted:
                    return o
        cands.sort(key=lambda o: (self.env.world.distance(org.position, o.position), o.id))
        return cands[0] if cands else None

    def _fail(self, org: Organism, msg: str) -> None:
        org.stats["failed_actions"] += 1
        org.last_result = {"action": "reproduce", "success": False, "message": msg, "energy_delta": 0.0}
        org.memory.record(self.env.tick, "action_failed", f"reproduce failed: {msg}")

    def _make_child(self, parents: list[Organism]) -> tuple[Organism, list[str], int]:
        cfg, ev = self.cfg, self.cfg.evolution
        if len(parents) == 1:
            genome = parents[0].genome.copy()
        else:
            genome = Genome.crossover(parents[0].genome, parents[1].genome, self.rng, ev.crossover)
        mutated: list[str] = []
        if ev.mutation_enabled and ev.mutation_rate > 0:
            genome, mutated = genome.mutate(self.rng, ev.mutation_rate, ev.mutation_strength)
        # Place in the parent's cell or an adjacent one.
        p0 = parents[0]
        options = [p0.position] + [c for c in (self.env.world.step(p0.position, d) for d in "NSEW") if c]
        pos = self.rng.choice(options)
        child = Organism(
            id=self.new_id(),
            parent_ids=[p.id for p in parents],
            generation=max(p.generation for p in parents) + 1,
            lineage_id=p0.lineage_id,
            birth_tick=self.env.tick,
            position=pos,
            energy=cfg.reproduction.offspring_energy,
            health=cfg.organism.initial_health,
            genome=genome,
            memory=make_memory(cfg, genome),
            mutations=mutated,
            last_decision_energy=cfg.reproduction.offspring_energy,
        )
        inherited = inherit_memory(cfg, child, parents, self.env.tick)
        child.memory.record(self.env.tick, "born", f"came into being at {list(pos)} next to {p0.id}",
                            {"other": p0.id, "pos": list(pos)})
        self.env.add(child)
        return child, mutated, inherited


def founder(cfg: SimulationConfig, oid: str, lineage: str, pos: tuple[int, int], genome: Genome) -> Organism:
    oc = cfg.organism
    return Organism(id=oid, parent_ids=[], generation=0, lineage_id=lineage, birth_tick=0,
                    position=pos, energy=oc.initial_energy, health=oc.initial_health,
                    genome=genome, memory=make_memory(cfg, genome),
                    last_decision_energy=oc.initial_energy, last_decision_health=oc.initial_health)


def birth_record(child: Organism, parents: list[Organism], mutated: list[str], inherited: int) -> dict[str, Any]:
    return {"tick": child.birth_tick, "id": child.id, "parent_ids": child.parent_ids,
            "generation": child.generation, "lineage_id": child.lineage_id,
            "genome": child.genome.to_dict(), "mutated_traits": mutated,
            "inherited_memories": inherited}
