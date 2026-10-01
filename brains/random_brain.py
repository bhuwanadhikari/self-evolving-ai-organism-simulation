"""Uniformly random actions: a null model for comparison."""

from __future__ import annotations

import random
from typing import Any

from .base import Brain, DecisionContext


class RandomBrain(Brain):
    name = "random"
    version = "1"

    def decide(self, ctx: DecisionContext, rng: random.Random) -> dict[str, Any]:
        action = rng.choice(ctx.allowed_actions)
        args: dict[str, Any] = {}
        others = [e["id"] for e in ctx.observation.get("nearby_entities", [])]
        if action == "move":
            args = {"direction": rng.choice("NSEW")}
        elif action in ("communicate", "share", "attack"):
            if not others:
                action, args = "rest", {}
            else:
                args = {"target": rng.choice(others)}
                if action == "communicate":
                    args["message"] = rng.choice(["hello", "food", "danger", "here"])
                if action == "share":
                    args["amount"] = 10
        elif action == "store_memory":
            loc = ctx.observation["location"]
            args = {"note": f"was at {loc['x']},{loc['y']}"}
        return {"action": action, "args": args, "reason": "random"}
