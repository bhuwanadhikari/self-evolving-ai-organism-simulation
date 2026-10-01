"""Genome-parameterised heuristic brain (Stage 1-3 baseline and cheap control).

IMPORTANT for interpretation: this brain's genome->behaviour mapping is
designed by hand. Evolution can still act on it (selection changes trait
frequencies, and therefore behaviour), but behaviours seen with this brain are
*not* emergent cognition. It exists to validate the evolutionary machinery
cheaply and to serve as a control for LLM-driven runs.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Brain, DecisionContext

DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def _toward(dx: int, dy: int, rng: random.Random) -> str:
    if dx == 0 and dy == 0:
        return rng.choice("NSEW")
    if abs(dx) > abs(dy) or (abs(dx) == abs(dy) and rng.random() < 0.5):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


class RuleBrain(Brain):
    name = "rule"
    version = "1"

    def decide(self, ctx: DecisionContext, rng: random.Random) -> dict[str, Any]:
        g, obs, allowed = ctx.genome, ctx.observation, set(ctx.allowed_actions)
        me, loc = obs["self"], obs["location"]
        e = me["energy"] / me["max_energy"]
        blocked = set(obs.get("blocked_directions", []))
        hunger_line = 0.3 + 0.5 * g["survival_priority"]
        hungry = e < hunger_line
        others = obs.get("nearby_entities", [])
        adjacent = [o for o in others if o["distance"] <= 1]
        cells = obs.get("nearby_cells", [])

        def act(a: str, reason: str, **args: Any) -> dict[str, Any]:
            return {"action": a, "args": args, "reason": reason}

        def free_dirs() -> list[str]:
            return [d for d in "NSEW" if d not in blocked]

        # 1. Leave dangerous ground unless risk-tolerant.
        if loc["hazard_level"] > 0 and rng.random() > g["risk_tolerance"]:
            hz = {(c["dx"], c["dy"]): c["hazard_level"] for c in cells}
            options = sorted(free_dirs(), key=lambda d: (hz.get(DIRS[d], 0.0), rng.random()))
            if options:
                return act("move", "danger here", direction=options[0])

        # 2. Aggression when hungry and a weaker neighbour is adjacent.
        if "attack" in allowed and hungry and adjacent and rng.random() < g["aggression_tendency"]:
            weakest = min(adjacent, key=lambda o: (o["visible_energy"], o["id"]))
            if weakest["visible_energy"] < me["energy"] or rng.random() < g["risk_tolerance"]:
                return act("attack", "hungry and a weaker one is near", target=weakest["id"])

        # 3. Eat.
        if loc["resources"] > 0 and (hungry or e < 0.95):
            return act("consume", "food here")

        # 4. Reproduce when well fed.
        if "reproduce" in allowed and e > 0.9 - 0.4 * g["reproduction_priority"] \
                and rng.random() < g["reproduction_priority"]:
            if adjacent:
                return act("reproduce", "well fed", partner=adjacent[0]["id"])
            return act("reproduce", "well fed")

        # 5. Share with a weak neighbour.
        if "share" in allowed and adjacent and e > 0.5 and rng.random() < g["cooperation_tendency"] * 0.5:
            needy = min(adjacent, key=lambda o: (o["visible_energy"], o["id"]))
            if needy["visible_energy"] < 40:
                return act("share", "neighbour looks weak", target=needy["id"], amount=10)

        # 6. Tell a neighbour about food.
        if "communicate" in allowed and others and cells and rng.random() < g["cooperation_tendency"] * 0.1:
            food = max(cells, key=lambda c: c["resources"])
            if food["resources"] > 0:
                return act("communicate", "sharing information", target=others[0]["id"],
                           message=f"food at {food['x']},{food['y']}")

        # 7. Go to visible food (avoiding hazards beyond tolerance).
        food_cells = [c for c in cells if c["resources"] > 0 and c["hazard_level"] <= g["risk_tolerance"]]
        if food_cells and (hungry or e < 0.9):
            target = min(food_cells, key=lambda c: (max(abs(c["dx"]), abs(c["dy"])), -c["resources"], c["y"], c["x"]))
            d = _toward(target["dx"], target["dy"], rng)
            if d not in blocked:
                return act("move", "food nearby", direction=d)

        # 8. Go to remembered food or follow a message when hungry.
        if hungry:
            for m in reversed(ctx.memories):
                pos = (m.get("data") or {}).get("pos")
                if m["kind"] in ("consumed", "found_resource", "inherited", "noted") and pos:
                    dx, dy = pos[0] - loc["x"], pos[1] - loc["y"]
                    if (dx or dy) and rng.random() < 0.5 + 0.5 * g["memory_retention"]:
                        d = _toward(dx, dy, rng)
                        if d not in blocked:
                            return act("move", "remembered food", direction=d)
                    break

        # 9. Look around occasionally.
        if rng.random() < 0.1 * g["curiosity"]:
            return act("observe", "curious")

        # 10. Explore or rest.
        if rng.random() < max(g["exploration_rate"], 0.6 if hungry else 0.0):
            fd = free_dirs()
            if fd:
                return act("move", "exploring", direction=rng.choice(fd))
        return act("rest", "nothing to do")
