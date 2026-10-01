"""Per-tick population, evolutionary and behavioural metrics.

These are measurements only. Nothing in the simulator reads them back to
select organisms; selection is purely a consequence of the environment.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from simulation.genome import TRAITS, diversity

ACTION_NAMES = ("observe", "move", "consume", "rest", "communicate", "store_memory", "reproduce",
                "share", "attack", "invalid")


class MetricsCollector:
    def __init__(self) -> None:
        self.total_births = 0
        self.total_deaths = 0
        self.lifespans: list[int] = []
        self.max_generation = 0

    def row(self, tick: int, alive: list[Any], births: list[Any], deaths: list[Any],
            action_counts: Counter, failed: int, world_resources: int, mutations: int,
            brain_calls: int, cached_decisions: int) -> dict[str, Any]:
        self.total_births += len(births)
        self.total_deaths += len(deaths)
        self.lifespans += [d.age for d in deaths]
        n = len(alive)
        if alive:
            self.max_generation = max(self.max_generation, max(o.generation for o in alive))
        death_causes = Counter(d.death_cause for d in deaths)
        lineages = Counter(o.lineage_id for o in alive)
        dominant, dom_n = (lineages.most_common(1)[0] if lineages else ("", 0))

        def mean(xs: list[float]) -> float:
            return round(statistics.fmean(xs), 4) if xs else 0.0

        row: dict[str, Any] = {
            "tick": tick,
            "population": n,
            "births": len(births),
            "deaths": len(deaths),
            "deaths_starvation": death_causes.get("starvation", 0),
            "deaths_injury": death_causes.get("injury", 0),
            "deaths_old_age": death_causes.get("old_age", 0),
            "birth_rate": round(len(births) / n, 4) if n else 0.0,
            "death_rate": round(len(deaths) / (n + len(deaths)), 4) if (n + len(deaths)) else 0.0,
            "avg_energy": mean([o.energy for o in alive]),
            "avg_health": mean([o.health for o in alive]),
            "avg_age": mean([o.age for o in alive]),
            "avg_lifespan_dead": mean(self.lifespans),
            "mean_generation": mean([o.generation for o in alive]),
            "max_generation": self.max_generation,
            "lineages_alive": len(lineages),
            "dominant_lineage": dominant,
            "dominant_lineage_share": round(dom_n / n, 4) if n else 0.0,
            "genome_diversity": round(diversity([o.genome for o in alive]), 4),
            "mutations": mutations,
            "world_resources": world_resources,
            "failed_actions": failed,
            "brain_calls": brain_calls,
            "cached_decisions": cached_decisions,
            "avg_cells_visited": mean([len(o.visited) for o in alive]),
            "avg_memory_items": mean([len(o.memory) for o in alive]),
        }
        for a in ACTION_NAMES:
            row[f"act_{a}"] = action_counts.get(a, 0)
        for t in TRAITS:
            vals = [getattr(o.genome, t) for o in alive]
            row[f"trait_{t}"] = mean(vals)
            row[f"trait_{t}_std"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
        return row
