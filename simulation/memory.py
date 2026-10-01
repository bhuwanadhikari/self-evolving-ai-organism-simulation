"""Episodic memory with bounded capacity.

Short-term memory is a FIFO of recent events. When an item falls out of
short-term memory it is promoted to long-term memory if its importance is
high enough. Long-term memory is capacity-limited (scaled by the genome's
memory_retention when physical effects are on) and costs energy upkeep per
item, so remembering is never free.

Importance values below are *salience* defaults for the memory system, not
behavioural rules: they decide what is retained, not what the organism does.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any

# Default salience by event kind.
IMPORTANCE: dict[str, float] = {
    "moved": 0.05,
    "rested": 0.05,
    "observed": 0.2,
    "consumed": 0.5,
    "found_resource": 0.5,
    "action_failed": 0.3,
    "damage": 0.7,
    "attacked_by": 0.9,
    "attacked": 0.6,
    "received_share": 0.8,
    "shared": 0.6,
    "message_received": 0.5,
    "message_sent": 0.2,
    "reproduced": 0.7,
    "born": 0.6,
    "noted": 1.0,  # the organism explicitly chose to store this
    "inherited": 0.8,
    "saw_death": 0.7,
}


@dataclass
class MemoryItem:
    tick: int
    kind: str
    text: str
    importance: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryItem":
        return cls(**d)


class EpisodicMemory:
    def __init__(self, short_capacity: int, long_capacity: int, promote_threshold: float = 0.5):
        self.short_capacity = max(1, short_capacity)
        self.long_capacity = max(0, long_capacity)
        self.promote_threshold = promote_threshold
        self.short_term: deque[MemoryItem] = deque()
        self.long_term: list[MemoryItem] = []

    # ---- writing ----------------------------------------------------------
    def record(self, tick: int, kind: str, text: str, data: dict[str, Any] | None = None,
               importance: float | None = None) -> MemoryItem:
        item = MemoryItem(tick, kind, text, IMPORTANCE.get(kind, 0.3) if importance is None else importance,
                          data or {})
        self.short_term.append(item)
        while len(self.short_term) > self.short_capacity:
            old = self.short_term.popleft()
            if old.importance >= self.promote_threshold:
                self._add_long(old)
        return item

    def store_long(self, item: MemoryItem) -> None:
        self._add_long(item)

    def _add_long(self, item: MemoryItem) -> None:
        if self.long_capacity == 0:
            return
        self.long_term.append(item)
        if len(self.long_term) > self.long_capacity:
            # Evict the least valuable: low importance and old.
            victim = min(self.long_term, key=lambda m: (m.importance, m.tick))
            self.long_term.remove(victim)

    # ---- reading ----------------------------------------------------------
    def retrieve(self, n: int) -> list[MemoryItem]:
        """Most important long-term memories plus the most recent short-term ones."""
        n_long = min(len(self.long_term), n // 2)
        long_sel = sorted(self.long_term, key=lambda m: (-m.importance, -m.tick))[:n_long]
        short_sel = list(self.short_term)[-(n - n_long):] if n - n_long > 0 else []
        return sorted(long_sel, key=lambda m: m.tick) + short_sel

    def __len__(self) -> int:
        return len(self.short_term) + len(self.long_term)

    # ---- inheritance --------------------------------------------------------
    def select_for_offspring(self, k: int) -> list[MemoryItem]:
        pool = self.long_term + list(self.short_term)
        return sorted(pool, key=lambda m: (-m.importance, -m.tick))[:k]

    def compress(self, k: int) -> list[str]:
        """Deterministic summary of experience (Mode C: 'compressed knowledge').

        Produces short factual statements (no advice): places where food was
        obtained, individuals associated with harm or help, and event counts.
        """
        pool = self.long_term + list(self.short_term)
        facts: list[str] = []
        food = Counter(tuple(m.data["pos"]) for m in pool if m.kind in ("consumed", "found_resource") and "pos" in m.data)
        for pos, c in food.most_common(2):
            facts.append(f"food was obtained at {list(pos)} ({c} times)")
        harm = Counter(m.data.get("other") for m in pool if m.kind == "attacked_by" and m.data.get("other"))
        for oid, c in harm.most_common(1):
            facts.append(f"{oid} caused harm ({c} times)")
        help_ = Counter(m.data.get("other") for m in pool if m.kind == "received_share" and m.data.get("other"))
        for oid, c in help_.most_common(1):
            facts.append(f"{oid} gave food ({c} times)")
        dmg = Counter(tuple(m.data["pos"]) for m in pool if m.kind == "damage" and "pos" in m.data)
        for pos, c in dmg.most_common(1):
            facts.append(f"injury happened at {list(pos)} ({c} times)")
        kinds = Counter(m.kind for m in pool)
        if kinds:
            facts.append("experience: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        return facts[:k]

    # ---- serialization ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "short_capacity": self.short_capacity,
            "long_capacity": self.long_capacity,
            "promote_threshold": self.promote_threshold,
            "short_term": [m.to_dict() for m in self.short_term],
            "long_term": [m.to_dict() for m in self.long_term],
        }
