"""Restricted action API.

A brain returns a *proposal* ({"action": ..., "args": {...}, "reason": ...}).
`parse_action` turns it into an `Action` or a rejected placeholder; the
Environment then validates it against the world state and applies it. There is
no other channel through which a brain can affect anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SimulationConfig
from .world import DIRECTIONS

BASE_ACTIONS = ("observe", "move", "consume", "rest", "store_memory", "reproduce")

ACTION_DOCS: dict[str, str] = {
    "observe": "look around more carefully (wider view next time)",
    "move": 'move one step; args: {"direction": "N"|"S"|"E"|"W"}',
    "consume": "eat food at your current location",
    "rest": "stay still; uses less energy and heals",
    "communicate": 'send a short message; args: {"target": "<id>", "message": "<text>"}',
    "store_memory": 'remember something; args: {"note": "<text>"}',
    "reproduce": 'produce offspring (needs a lot of energy); args: {"partner": "<id>"} optional',
    "share": 'give food energy to another; args: {"target": "<id>", "amount": <number>}',
    "attack": 'harm another nearby; args: {"target": "<id>"}',
}

REPEATABLE = {"move", "consume", "rest"}


@dataclass
class Action:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    parse_error: str | None = None
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {"action": self.name, "args": self.args, "reason": self.reason}
        if self.parse_error:
            d["parse_error"] = self.parse_error
        return d


@dataclass
class ActionResult:
    success: bool
    message: str
    energy_delta: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "message": self.message,
                "energy_delta": round(self.energy_delta, 3), **({"data": self.data} if self.data else {})}


def available_actions(cfg: SimulationConfig) -> list[str]:
    acts = list(BASE_ACTIONS)
    if cfg.communication.enabled:
        acts.insert(4, "communicate")
    if cfg.actions.share_enabled:
        acts.append("share")
    if cfg.actions.attack_enabled:
        acts.append("attack")
    if not cfg.reproduction.enabled:
        acts.remove("reproduce")
    return acts


def parse_action(proposal: dict[str, Any] | None, allowed: list[str]) -> Action:
    """Validate the *shape* of a proposal. World-state validation happens later."""
    if not isinstance(proposal, dict):
        return Action("invalid", parse_error="decision was not an object")
    name = str(proposal.get("action", "")).strip().lower()
    args = proposal.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    reason = str(proposal.get("reason", ""))[:200]
    if name not in allowed:
        return Action("invalid", args, reason, parse_error=f"unknown action '{name}'")
    if name == "move":
        d = str(args.get("direction", "")).strip().upper()[:1]
        if d not in DIRECTIONS:
            return Action("invalid", args, reason, parse_error="move needs direction N, S, E or W")
        args = {"direction": d}
    elif name in ("communicate", "share", "attack"):
        if not args.get("target"):
            return Action("invalid", args, reason, parse_error=f"{name} needs a target")
        args = dict(args)
        args["target"] = str(args["target"])
        if name == "share":
            try:
                args["amount"] = float(args.get("amount", 10))
            except (TypeError, ValueError):
                return Action("invalid", args, reason, parse_error="share amount must be a number")
        if name == "communicate":
            args["message"] = str(args.get("message", ""))
    elif name == "store_memory":
        args = {"note": str(args.get("note", args.get("data", "")))}
    elif name == "reproduce":
        args = {"partner": str(args["partner"])} if args.get("partner") else {}
    else:
        args = {}
    return Action(name, args, reason)
