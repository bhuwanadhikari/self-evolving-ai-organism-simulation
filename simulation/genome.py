"""Heritable genome: normalized behavioural tendencies in [0, 1].

Most traits are *dispositions*: they are shown to the brain as part of the
organism's internal state and have no direct mechanical effect. Two traits
also have a physical effect when `organism.genome_physical_effects` is on,
each with a built-in trade-off so no value is free:

* resource_efficiency: more energy per resource unit, but higher metabolism.
* memory_retention: larger long-term memory, which costs upkeep per item.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields

TRAITS: tuple[str, ...] = (
    "survival_priority",
    "reproduction_priority",
    "exploration_rate",
    "risk_tolerance",
    "resource_efficiency",
    "cooperation_tendency",
    "aggression_tendency",
    "curiosity",
    "memory_retention",
)


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class Genome:
    survival_priority: float = 0.5
    reproduction_priority: float = 0.5
    exploration_rate: float = 0.5
    risk_tolerance: float = 0.5
    resource_efficiency: float = 0.5
    cooperation_tendency: float = 0.5
    aggression_tendency: float = 0.5
    curiosity: float = 0.5
    memory_retention: float = 0.5

    # ---- construction -------------------------------------------------
    @classmethod
    def random(cls, rng: random.Random) -> "Genome":
        return cls(**{t: rng.random() for t in TRAITS})

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Genome":
        unknown = set(data) - set(TRAITS)
        if unknown:
            raise KeyError(f"Unknown genome traits: {sorted(unknown)}")
        return cls(**{t: _clip(float(data.get(t, 0.5))) for t in TRAITS})

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in asdict(self).items()}

    def vector(self) -> list[float]:
        return [getattr(self, t) for t in TRAITS]

    def copy(self) -> "Genome":
        return Genome(**asdict(self))

    # ---- variation ----------------------------------------------------
    def mutate(self, rng: random.Random, rate: float, strength: float) -> tuple["Genome", list[str]]:
        """Return (mutated copy, list of mutated trait names)."""
        child = self.copy()
        mutated: list[str] = []
        for t in TRAITS:
            if rng.random() < rate:
                setattr(child, t, _clip(getattr(child, t) + rng.gauss(0.0, strength)))
                mutated.append(t)
        return child, mutated

    @staticmethod
    def crossover(a: "Genome", b: "Genome", rng: random.Random, method: str = "uniform") -> "Genome":
        if method == "uniform":
            return Genome(**{t: getattr(a if rng.random() < 0.5 else b, t) for t in TRAITS})
        if method == "blend":
            out = {}
            for t in TRAITS:
                w = rng.random()
                out[t] = _clip(w * getattr(a, t) + (1 - w) * getattr(b, t))
            return Genome(**out)
        raise ValueError(f"Unknown crossover method: {method}")

    # ---- physical effects -----------------------------------------------
    def energy_gain_multiplier(self) -> float:
        return 0.8 + 0.4 * self.resource_efficiency

    def metabolic_multiplier(self) -> float:
        return 0.9 + 0.2 * self.resource_efficiency

    def memory_capacity_multiplier(self) -> float:
        return 0.5 + self.memory_retention


def distance(a: Genome, b: Genome) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.vector(), b.vector())))


def diversity(genomes: list[Genome], rng: random.Random | None = None, max_pairs: int = 2000) -> float:
    """Mean pairwise Euclidean distance (sampled when the population is large).

    Uses a private RNG seeded from the population so that measuring never
    perturbs the simulation's random streams.
    """
    n = len(genomes)
    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        s = sum(distance(genomes[i], genomes[j]) for i in range(n) for j in range(i + 1, n))
        return s / total_pairs
    r = rng or random.Random(n)
    s = 0.0
    for _ in range(max_pairs):
        i, j = r.sample(range(n), 2)
        s += distance(genomes[i], genomes[j])
    return s / max_pairs


assert tuple(f.name for f in fields(Genome)) == TRAITS
