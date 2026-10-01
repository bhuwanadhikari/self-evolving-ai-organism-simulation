"""Experiment configuration.

Every tunable number in the simulator lives here. Configs are loaded from
YAML/JSON and merged over these defaults; unknown keys raise an error so that
typos in an experiment file never silently change an experiment.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SIMULATION_VERSION = "0.1.0"
ENVIRONMENT_VERSION = "0.1.0"


@dataclass
class PopulationConfig:
    initial_size: int = 20
    # "random": each founder gets a uniformly random genome.
    # "fixed": every founder gets `fixed_genome` (missing traits default to 0.5).
    initial_genome: str = "random"
    fixed_genome: dict[str, float] = field(default_factory=dict)
    # Hard compute guard, not a fitness rule. When reached, reproduction fails
    # with the feedback "no room for offspring".
    max_population: int = 200


@dataclass
class WorldConfig:
    width: int = 20
    height: int = 20
    wrap: bool = False  # toroidal world if True


@dataclass
class ResourceConfig:
    initial_count: int = 150  # total resource units scattered at t=0
    max_per_cell: int = 5
    energy_per_unit: float = 10.0
    # Probability per tick per cell (scaled by cell weight) of +1 unit.
    regeneration_rate: float = 0.02
    # "uniform" or "hotspots" (regeneration concentrated around moving centres)
    distribution: str = "uniform"
    hotspot_count: int = 3
    hotspot_radius: float = 3.0
    hotspot_floor: float = 0.05  # min regeneration weight far from hotspots
    drift_every: int = 0  # Env 3: hotspots move one step every N ticks (0=off)
    relocate_every: int = 0  # Env 5: hotspots jump to new places every N ticks


@dataclass
class HazardConfig:
    zone_count: int = 0
    zone_radius: float = 2.0
    max_level: float = 1.0  # hazard_level in [0, 1]
    damage: float = 10.0  # health lost per tick at hazard_level 1.0
    drift_every: int = 0
    relocate_every: int = 0


@dataclass
class AdaptiveConfig:
    """Env 7: the environment responds to population behaviour."""

    enabled: bool = False
    # Each unit harvested lowers the cell's fertility (regeneration multiplier).
    depletion_per_harvest: float = 0.05
    fertility_recovery: float = 0.002
    min_fertility: float = 0.1
    # Crowding raises local hazard (e.g. disease) above this many occupants.
    crowding_threshold: int = 0  # 0 = off
    crowding_hazard: float = 0.3


@dataclass
class OrganismConfig:
    initial_energy: float = 100.0
    max_energy: float = 200.0
    initial_health: float = 100.0
    max_health: float = 100.0
    max_age: int = 400
    metabolic_cost: float = 1.0
    move_cost: float = 1.0
    communicate_cost: float = 0.5
    store_memory_cost: float = 0.5
    observe_cost: float = 0.2
    rest_metabolic_multiplier: float = 0.5
    rest_heal: float = 2.0
    health_regen: float = 0.5  # passive per tick when energy is above threshold
    health_regen_energy_threshold: float = 0.5  # fraction of max_energy
    consume_amount: int = 2  # max units taken per consume action
    perception_radius: int = 2
    observe_radius_bonus: int = 2  # extra radius for the observe() action
    # Genome traits with direct physical effects (see Genome docs). If False,
    # the genome only influences the brain.
    genome_physical_effects: bool = True


@dataclass
class ReproductionConfig:
    enabled: bool = True
    mode: str = "asexual"  # "asexual" or "sexual"
    cost: float = 60.0  # total energy paid by the parent(s)
    offspring_energy: float = 40.0  # energy the child starts with
    min_energy: float = 80.0  # parent must have at least this much
    min_age: int = 10
    cooldown: int = 10
    mate_radius: int = 1  # sexual mode: partner must be this close


@dataclass
class EvolutionConfig:
    mutation_enabled: bool = True
    mutation_rate: float = 0.02  # per-trait probability
    mutation_strength: float = 0.1  # std-dev of Gaussian perturbation
    crossover: str = "uniform"  # sexual mode only


@dataclass
class MemoryConfig:
    short_term_capacity: int = 10
    long_term_capacity: int = 20
    promote_threshold: float = 0.5  # importance needed to enter long-term memory
    upkeep_cost_per_item: float = 0.01  # energy per long-term item per tick
    # "none" | "selected" | "compressed" | "genome_only"
    inheritance_mode: str = "none"
    inherit_count: int = 5
    retrieval_count: int = 8  # items shown to the brain per decision


@dataclass
class CommunicationConfig:
    enabled: bool = True
    max_length: int = 80
    range: int = 3  # Chebyshev distance; 0 = unlimited
    noise: float = 0.0  # per-character corruption probability


@dataclass
class ActionsConfig:
    share_enabled: bool = False
    attack_enabled: bool = False
    attack_cost: float = 3.0
    attack_damage: float = 15.0
    attack_steal_fraction: float = 0.0  # fraction of target energy transferred
    kin_recognition: bool = False  # show "related" flag for same-lineage neighbours


@dataclass
class DecisionPolicyConfig:
    """When the brain is actually consulted (LLM cost control).

    If `every_tick` is False, the previous action is repeated until something
    significant changes (see simulation.decision_policy).
    """

    every_tick: bool = True
    max_interval: int = 5
    energy_change_threshold: float = 15.0
    health_change_threshold: float = 10.0


@dataclass
class BrainConfig:
    type: str = "rule"  # "random" | "rule" | "qwen"
    backend: str = "ollama"  # qwen: "ollama" | "openai" | "transformers" | "mock"
    model: str = "qwen2.5:7b-instruct"
    endpoint: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 160
    timeout: float = 120.0
    concurrency: int = 4
    # Inference endpoints are restricted to localhost unless explicitly allowed.
    allow_remote_endpoint: bool = False
    # Name of the environment variable holding the API key (never put the key
    # itself in a config file). Empty = no Authorization header sent.
    api_key_env: str = ""
    # "minimal" | "low" | "medium" | "high"; empty = provider default. Only
    # applies to the openai backend, and only to reasoning models (e.g.
    # gpt-oss) — most providers won't let reasoning be disabled outright, but
    # lowering effort cuts hidden reasoning tokens (cost/latency) a lot.
    reasoning_effort: str = ""
    # If True, the prompt states how energy/health/death work. Default False:
    # organisms must learn the rules of their world from experience.
    reveal_mechanics: bool = False
    log_prompts: bool = False
    retries: int = 2
    decision: DecisionPolicyConfig = field(default_factory=DecisionPolicyConfig)


@dataclass
class LoggingConfig:
    output_dir: str = "runs"
    log_actions: bool = True
    snapshot_every: int = 5
    checkpoint_every: int = 100
    print_every: int = 25


@dataclass
class SimulationConfig:
    name: str = "baseline"
    seed: int = 42
    ticks: int = 500
    description: str = ""
    hypothesis: str = ""
    population: PopulationConfig = field(default_factory=PopulationConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    hazards: HazardConfig = field(default_factory=HazardConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        errors: list[str] = []
        if self.world.width < 1 or self.world.height < 1:
            errors.append("world dimensions must be positive")
        if self.reproduction.mode not in ("asexual", "sexual"):
            errors.append("reproduction.mode must be 'asexual' or 'sexual'")
        if self.memory.inheritance_mode not in ("none", "selected", "compressed", "genome_only"):
            errors.append("memory.inheritance_mode invalid")
        if self.resources.distribution not in ("uniform", "hotspots"):
            errors.append("resources.distribution must be 'uniform' or 'hotspots'")
        if self.brain.type not in ("random", "rule", "qwen"):
            errors.append("brain.type must be random, rule or qwen")
        if self.population.initial_genome not in ("random", "fixed"):
            errors.append("population.initial_genome must be 'random' or 'fixed'")
        if not 0.0 <= self.evolution.mutation_rate <= 1.0:
            errors.append("evolution.mutation_rate must be in [0, 1]")
        if errors:
            raise ValueError("Invalid config: " + "; ".join(errors))


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> Any:
    """Recursively apply `data` onto dataclass instance `obj`."""
    valid = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)
    return obj


def config_from_dict(data: dict[str, Any]) -> SimulationConfig:
    cfg = _merge(SimulationConfig(), data)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SimulationConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    cfg = _merge(SimulationConfig(), data)
    if overrides:
        _merge(cfg, overrides)
    cfg.validate()
    return cfg


def parse_override(expr: str) -> dict[str, Any]:
    """Turn 'brain.type=qwen' into {'brain': {'type': 'qwen'}} (value parsed as YAML)."""
    import yaml

    key, _, raw = expr.partition("=")
    value = yaml.safe_load(raw)
    out: dict[str, Any] = {}
    node = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
