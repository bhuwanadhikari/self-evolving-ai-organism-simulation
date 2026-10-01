"""The physical world: a grid of locations with resources and hazards.

Environments 1-7 from the design are all expressed through configuration of
this one class (see experiments/ for examples):

1 abundant resources     -> high initial_count / regeneration_rate
2 limited resources      -> low initial_count / regeneration_rate
3 moving resources       -> distribution=hotspots, drift_every>0
4 hazards                -> hazards.zone_count>0
5 environmental change   -> relocate_every>0 (resources and/or hazards)
6 population competition -> scarce resources + many founder lineages
7 adaptive environment   -> adaptive.enabled (overharvest depletion, crowding)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .config import AdaptiveConfig, HazardConfig, ResourceConfig, WorldConfig

DIRECTIONS: dict[str, tuple[int, int]] = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


@dataclass
class Location:
    x: int
    y: int
    resources: int
    hazard_level: float
    environmental_conditions: dict[str, float] = field(default_factory=dict)
    organisms: list[str] = field(default_factory=list)


class World:
    def __init__(self, wcfg: WorldConfig, rcfg: ResourceConfig, hcfg: HazardConfig,
                 acfg: AdaptiveConfig, rng: random.Random):
        self.w, self.h, self.wrap = wcfg.width, wcfg.height, wcfg.wrap
        self.rcfg, self.hcfg, self.acfg = rcfg, hcfg, acfg
        self.rng = rng
        self.tick = 0
        self.resources = [[0] * self.w for _ in range(self.h)]
        self.fertility = [[1.0] * self.w for _ in range(self.h)]
        self.base_hazard = [[0.0] * self.w for _ in range(self.h)]
        self.crowd_hazard = [[0.0] * self.w for _ in range(self.h)]
        self.hotspots: list[list[int]] = []
        self.hazard_zones: list[list[int]] = []
        self.events: list[dict[str, Any]] = []  # environment events this tick
        self._init()

    # ---- geometry ------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def step(self, pos: tuple[int, int], direction: str) -> tuple[int, int] | None:
        dx, dy = DIRECTIONS[direction]
        x, y = pos[0] + dx, pos[1] + dy
        if self.wrap:
            return x % self.w, y % self.h
        return (x, y) if self.in_bounds(x, y) else None

    def delta(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        """Vector from a to b (shortest, respecting wrap)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        if self.wrap:
            if abs(dx) > self.w / 2:
                dx -= int(math.copysign(self.w, dx))
            if abs(dy) > self.h / 2:
                dy -= int(math.copysign(self.h, dy))
        return dx, dy

    def distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        dx, dy = self.delta(a, b)
        return max(abs(dx), abs(dy))  # Chebyshev

    def cells_within(self, pos: tuple[int, int], radius: int):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = pos[0] + dx, pos[1] + dy
                if self.wrap:
                    yield (x % self.w, y % self.h), dx, dy
                elif self.in_bounds(x, y):
                    yield (x, y), dx, dy

    # ---- state -----------------------------------------------------------------
    def hazard(self, x: int, y: int) -> float:
        return min(1.0, self.base_hazard[y][x] + self.crowd_hazard[y][x])

    def location(self, x: int, y: int, organisms: list[str] | None = None) -> Location:
        return Location(x, y, self.resources[y][x], round(self.hazard(x, y), 3),
                        {"fertility": round(self.fertility[y][x], 3)}, organisms or [])

    def total_resources(self) -> int:
        return sum(map(sum, self.resources))

    def take(self, x: int, y: int, amount: int) -> int:
        got = min(amount, self.resources[y][x])
        self.resources[y][x] -= got
        if got and self.acfg.enabled:
            f = self.fertility[y][x] - self.acfg.depletion_per_harvest * got
            self.fertility[y][x] = max(self.acfg.min_fertility, f)
        return got

    # ---- initialisation --------------------------------------------------------
    def _random_cell(self) -> list[int]:
        return [self.rng.randrange(self.w), self.rng.randrange(self.h)]

    def _init(self) -> None:
        if self.rcfg.distribution == "hotspots":
            self.hotspots = [self._random_cell() for _ in range(self.rcfg.hotspot_count)]
        self.hazard_zones = [self._random_cell() for _ in range(self.hcfg.zone_count)]
        self._recompute_hazard()
        # Scatter initial resources weighted by regeneration weight.
        cells = [(x, y) for y in range(self.h) for x in range(self.w)]
        weights = [self._regen_weight(x, y) for x, y in cells]
        placed = 0
        attempts = 0
        cap = self.rcfg.max_per_cell * len(cells)
        target = min(self.rcfg.initial_count, cap)
        while placed < target and attempts < target * 50:
            attempts += 1
            x, y = self.rng.choices(cells, weights=weights)[0]
            if self.resources[y][x] < self.rcfg.max_per_cell:
                self.resources[y][x] += 1
                placed += 1

    def _regen_weight(self, x: int, y: int) -> float:
        if self.rcfg.distribution != "hotspots" or not self.hotspots:
            return 1.0
        r2 = 2 * self.rcfg.hotspot_radius ** 2
        best = 0.0
        for hx, hy in self.hotspots:
            dx, dy = self.delta((hx, hy), (x, y))
            best = max(best, math.exp(-(dx * dx + dy * dy) / r2))
        return max(self.rcfg.hotspot_floor, best)

    def _recompute_hazard(self) -> None:
        r2 = 2 * self.hcfg.zone_radius ** 2
        for y in range(self.h):
            for x in range(self.w):
                level = 0.0
                for hx, hy in self.hazard_zones:
                    dx, dy = self.delta((hx, hy), (x, y))
                    level = max(level, self.hcfg.max_level * math.exp(-(dx * dx + dy * dy) / r2))
                self.base_hazard[y][x] = round(level, 3) if level >= 0.05 else 0.0
        self._weights = [[self._regen_weight(x, y) for x in range(self.w)] for y in range(self.h)]

    def _drift(self, points: list[list[int]]) -> None:
        for p in points:
            d = self.rng.choice(list(DIRECTIONS.values()))
            p[0] = (p[0] + d[0]) % self.w if self.wrap else min(self.w - 1, max(0, p[0] + d[0]))
            p[1] = (p[1] + d[1]) % self.h if self.wrap else min(self.h - 1, max(0, p[1] + d[1]))

    # ---- dynamics ---------------------------------------------------------------
    def update(self, occupancy: dict[tuple[int, int], int]) -> None:
        """Advance environmental state by one tick."""
        self.tick += 1
        self.events = []
        t = self.tick
        changed = False
        rc, hc = self.rcfg, self.hcfg
        if self.hotspots:
            if rc.relocate_every and t % rc.relocate_every == 0:
                self.hotspots = [self._random_cell() for _ in self.hotspots]
                self.events.append({"type": "resource_relocation", "hotspots": [list(h) for h in self.hotspots]})
                changed = True
            elif rc.drift_every and t % rc.drift_every == 0:
                self._drift(self.hotspots)
                changed = True
        if self.hazard_zones:
            if hc.relocate_every and t % hc.relocate_every == 0:
                self.hazard_zones = [self._random_cell() for _ in self.hazard_zones]
                self.events.append({"type": "hazard_relocation", "zones": [list(z) for z in self.hazard_zones]})
                changed = True
            elif hc.drift_every and t % hc.drift_every == 0:
                self._drift(self.hazard_zones)
                changed = True
        if changed:
            self._recompute_hazard()

        # Regeneration. One RNG draw per cell every tick keeps streams aligned.
        for y in range(self.h):
            row, frow, wrow = self.resources[y], self.fertility[y], self._weights[y]
            for x in range(self.w):
                p = rc.regeneration_rate * wrow[x] * frow[x]
                if self.rng.random() < p and row[x] < rc.max_per_cell:
                    row[x] += 1

        if self.acfg.enabled:
            for y in range(self.h):
                frow = self.fertility[y]
                for x in range(self.w):
                    if frow[x] < 1.0:
                        frow[x] = min(1.0, frow[x] + self.acfg.fertility_recovery)
            if self.acfg.crowding_threshold > 0:
                for y in range(self.h):
                    for x in range(self.w):
                        self.crowd_hazard[y][x] = 0.0
                for (x, y), n in occupancy.items():
                    if n > self.acfg.crowding_threshold:
                        self.crowd_hazard[y][x] = self.acfg.crowding_hazard

    def snapshot(self) -> dict[str, Any]:
        return {
            "resources": [row[:] for row in self.resources],
            "hazard": [[round(self.hazard(x, y), 2) for x in range(self.w)] for y in range(self.h)],
            "hotspots": [list(h) for h in self.hotspots],
            "hazard_zones": [list(z) for z in self.hazard_zones],
        }
