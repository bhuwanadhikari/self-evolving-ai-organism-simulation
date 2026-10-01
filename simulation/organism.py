"""Organism state. Organisms never touch the world directly; the Environment
validates and applies every action on their behalf."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .genome import Genome
from .memory import EpisodicMemory


@dataclass
class Organism:
    id: str
    parent_ids: list[str]
    generation: int
    lineage_id: str
    birth_tick: int
    position: tuple[int, int]
    energy: float
    health: float
    genome: Genome
    memory: EpisodicMemory
    age: int = 0
    alive: bool = True
    offspring_count: int = 0
    death_tick: int | None = None
    death_cause: str | None = None
    last_reproduction_tick: int = -10**9
    mutations: list[str] = field(default_factory=list)
    # Messages delivered this tick, shown in the next observation.
    inbox: list[dict[str, Any]] = field(default_factory=list)
    last_result: dict[str, Any] | None = None
    resting: bool = False
    extra_radius: int = 0  # set by observe(), applies to the next observation
    wants_reproduction: dict[str, Any] | None = None
    # Decision caching (LLM cost control).
    last_decision: dict[str, Any] | None = None
    last_decision_tick: int = -10**9
    last_decision_energy: float = 0.0
    last_decision_health: float = 0.0
    last_neighbors: tuple[str, ...] = ()
    # Lifetime statistics for analysis.
    stats: dict[str, float] = field(default_factory=lambda: {
        "resources_consumed": 0.0,
        "energy_gained": 0.0,
        "distance_moved": 0.0,
        "messages_sent": 0.0,
        "shares_given": 0.0,
        "attacks_made": 0.0,
        "failed_actions": 0.0,
        "llm_calls": 0.0,
    })
    visited: set[tuple[int, int]] = field(default_factory=set)
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=5))

    def record_summary(self) -> dict[str, Any]:
        """Lifetime record for organisms.json / lineage analysis."""
        return {
            "id": self.id,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "lineage_id": self.lineage_id,
            "birth_tick": self.birth_tick,
            "death_tick": self.death_tick,
            "death_cause": self.death_cause,
            "age": self.age,
            "alive": self.alive,
            "offspring_count": self.offspring_count,
            "energy": round(self.energy, 3),
            "health": round(self.health, 3),
            "position": list(self.position),
            "genome": self.genome.to_dict(),
            "mutations": self.mutations,
            "stats": {k: round(v, 3) for k, v in self.stats.items()},
            "cells_visited": len(self.visited),
            "memory_items": len(self.memory),
        }
