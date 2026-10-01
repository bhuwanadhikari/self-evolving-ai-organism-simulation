from __future__ import annotations

from simulation.config import BrainConfig

from .base import Brain, DecisionContext
from .random_brain import RandomBrain
from .rule_brain import RuleBrain


def make_brain(cfg: BrainConfig) -> Brain:
    if cfg.type == "random":
        return RandomBrain()
    if cfg.type == "rule":
        return RuleBrain()
    if cfg.type == "qwen":
        from .qwen_brain import QwenBrain

        return QwenBrain(cfg)
    raise ValueError(f"Unknown brain type: {cfg.type}")


__all__ = ["Brain", "DecisionContext", "RandomBrain", "RuleBrain", "make_brain"]
