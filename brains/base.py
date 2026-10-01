"""Common brain interface.

A brain maps (observation, memory, genome) -> action proposal. It has no
access to the world, the simulator, the filesystem or the network (the Qwen
brain talks only to a local inference server on the experimenter's side; the
organism never sees that channel).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionContext:
    organism_id: str
    observation: dict[str, Any]
    memories: list[dict[str, Any]]
    genome: dict[str, float]
    allowed_actions: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


class Brain(ABC):
    name = "base"
    version = "0"

    @abstractmethod
    def decide(self, ctx: DecisionContext, rng: random.Random) -> dict[str, Any]:
        """Return {"action": str, "args": dict, "reason": str}."""

    def decide_batch(self, ctxs: list[DecisionContext], rng: random.Random) -> list[dict[str, Any]]:
        return [self.decide(c, rng) for c in ctxs]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

    def close(self) -> None:
        pass
