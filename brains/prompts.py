"""Prompt construction for language-model brains.

Creator-blind by design: prompts never mention humans, simulations, models,
Python, experiments, or goals such as "survive". They describe only the
being's body, surroundings, memories and possible actions. Any concept of
origin, purpose or death has to be formed from experience.

Bump PROMPT_VERSION whenever the wording changes; it is recorded in every run.
"""

from __future__ import annotations

import json
from typing import Any

from simulation.actions import ACTION_DOCS

PROMPT_VERSION = "1.0"

TRAIT_WORDS = {
    "survival_priority": "concern for your own condition",
    "reproduction_priority": "urge to produce offspring",
    "exploration_rate": "drive to wander",
    "risk_tolerance": "tolerance for danger",
    "resource_efficiency": "efficiency of digestion",
    "cooperation_tendency": "inclination to help others",
    "aggression_tendency": "inclination to harm others",
    "curiosity": "curiosity",
    "memory_retention": "retentiveness of memory",
}

MECHANICS = (
    "Facts about your body: energy is used up a little every moment and faster when moving; "
    "food restores energy; dangerous ground damages health; when energy or health reaches zero, "
    "or when you grow very old, you stop permanently. "
)


def system_prompt(organism_id: str, allowed: list[str], reveal_mechanics: bool) -> str:
    actions = "\n".join(f"- {a}: {ACTION_DOCS[a]}" for a in allowed)
    return (
        f"You are {organism_id}, a being living in a world made of cells arranged on a grid "
        "(x grows to the east, y grows to the south). "
        "Each moment you perceive your surroundings and choose exactly one action.\n"
        + (MECHANICS if reveal_mechanics else "")
        + "\nActions:\n" + actions + "\n\n"
        'Reply with only one JSON object: {"action": "<name>", "args": {...}, "reason": "<a few words>"}'
    )


def user_prompt(obs: dict[str, Any], memories: list[dict[str, Any]], genome: dict[str, float]) -> str:
    dispositions = {TRAIT_WORDS[k]: round(v, 2) for k, v in genome.items()}
    mem_lines = [f"[t{m['tick']}] {m['text']}" for m in memories]
    view = {k: v for k, v in obs.items() if k != "tick"}
    return (
        "Your innate dispositions (0 = very low, 1 = very high):\n"
        + json.dumps(dispositions)
        + "\n\nWhat you remember:\n"
        + ("\n".join(mem_lines) if mem_lines else "(nothing)")
        + "\n\nWhat you perceive now:\n"
        + json.dumps(view, separators=(",", ":"))
        + "\n\nChoose your action."
    )
