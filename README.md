# Artificial Life: Survival, Variation and Evolution

A closed, fully simulated ecosystem in which organisms controlled by a
language model (Qwen 7B-class), or by cheap stand-in brains, must obtain
energy, avoid hazards, reproduce and die. Heritable genomes mutate and are
selected **only** by what the environment allows to survive and reproduce.

The question it is built to study:

> If agents live in a closed world where survival and reproduction are
> necessary for continuation, do survival-oriented behaviour, adaptation,
> cooperation, competition or other strategies emerge through variation and
> selection?

It is only a computational experiment. Results say nothing about real
organisms, consciousness, creators, or the simulation hypothesis.

---

## Quick start

```bash
cd artificial-life
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                                  # 57 tests, ~3 s
python run_simulation.py experiments/baseline.yaml   # rule brain, 500 ticks, ~5 s
open runs/baseline_seed42/dashboard.html
```

### With Qwen (Ollama)

```bash
brew install ollama            # or see https://ollama.com
ollama serve &                 # if the app isn't already running
ollama pull qwen2.5:7b-instruct
OLLAMA_NUM_PARALLEL=4 ollama serve   # optional: allow parallel requests

# check the prompt/parse pipeline first, no model needed:
python run_simulation.py experiments/qwen_mock_dryrun.yaml

# then run the milestone: 20 Qwen organisms
python run_simulation.py experiments/qwen_first20.yaml --ticks 50
```

Speed depends on your hardware. Expect roughly 0.3–2 s per decision on an
Apple-silicon Mac, so 20 organisms × 200 ticks takes minutes to hours.
Decision caching (see below) usually removes 30–60% of calls. Runs save
checkpoints, so you can stop and continue at any time:

```bash
python run_simulation.py --resume runs/qwen_first20_seed42 --ticks 400
```

Other backends: `--set brain.backend=openai --set brain.endpoint=http://localhost:8000 --set brain.model=Qwen/Qwen2.5-7B-Instruct`
for vLLM, llama.cpp server or LM Studio, or `brain.backend=transformers` for
in-process Hugging Face inference (`pip install torch transformers accelerate`).

---

## Command line

```bash
python run_simulation.py CONFIG [--seeds 1 2 3] [--ticks N] [--set key=value ...] [--tree] [--no-dashboard]
python run_simulation.py --resume RUN_DIR [--ticks N]
python -m analysis.visualization RUN_DIR          # rebuild a dashboard
python -m analysis.compare runs/exp_c_* runs/exp_d_*   # compare runs across seeds
```

`--set` overrides any config value, for example
`--set memory.inheritance_mode=compressed --set evolution.mutation_rate=0.05`.

---

## Layout

```text
artificial-life/
├── simulation/
│   ├── config.py          all parameters (dataclasses), YAML/JSON loading, validation
│   ├── world.py           grid, resources, hotspots, hazards, adaptive fertility (Env 1-7)
│   ├── organism.py        organism state
│   ├── genome.py          9 traits in [0,1], mutation, crossover, diversity
│   ├── memory.py          episodic short/long-term memory, capacity, compression
│   ├── actions.py         restricted action API + proposal parsing
│   ├── environment.py     observations, action validation/effects, physiology, death
│   ├── reproduction.py    asexual/sexual reproduction, mutation, memory inheritance
│   ├── simulation.py      main loop, decision caching, checkpoints
│   └── logging_utils.py   structured output files
├── brains/
│   ├── base.py            Brain interface + DecisionContext
│   ├── random_brain.py    null model
│   ├── rule_brain.py      genome-parameterised heuristic (baseline/control)
│   ├── prompts.py         creator-blind prompts (versioned)
│   └── qwen_brain.py      Qwen via ollama / openai-compatible / transformers / mock
├── analysis/
│   ├── metrics.py         per-tick population/evolution/behaviour metrics
│   ├── lineage.py         lineage tree, ASCII rendering, lineage summaries
│   ├── visualization.py   builds the self-contained HTML dashboard
│   ├── dashboard_template.html
│   └── compare.py         cross-run comparison
├── experiments/           YAML configs (environments 1-7, experiments A-G, Qwen)
├── tests/
├── run_simulation.py
└── requirements.txt
```

---

## How the world works

### Tick order

1. `world.update()`: food regrows; hotspots and hazards drift or relocate; adaptive feedback runs.
2. Every living organism observes. Organisms act in a random order drawn from the seed.
3. The brain decides, in one batch, for organisms that need a new decision.
4. The environment validates and applies each action in sequence.
5. Physiology: metabolism, hazard damage, healing, ageing, memory upkeep.
6. Reproduction phase, including crossover and mutation.
7. Death phase: `energy <= 0` is starvation, `health <= 0` is injury, `age >= max_age` is old age.
8. Metrics, snapshots and checkpoints are recorded.

Decisions are taken at the same time from the start-of-tick view and then
resolved one by one, so no organism always moves first.

### Actions (the only interface a brain has)

| action | effect | cost |
|---|---|---|
| `observe` | wider perception next tick | small |
| `move(direction)` | one step N/S/E/W | `move_cost` |
| `consume` | eat up to `consume_amount` food here | – |
| `rest` | lower metabolism, heal | – |
| `communicate(target, message)` | deliver text (length/range limit, optional noise) | `communicate_cost` |
| `store_memory(note)` | write to long-term memory | small + upkeep |
| `reproduce(partner?)` | resolved in reproduction phase | `reproduction.cost` |
| `share(target, amount)` | give energy to an adjacent organism *(if enabled)* | the amount |
| `attack(target)` | damage an adjacent organism, optionally steal energy *(if enabled)* | `attack_cost` |

Every invalid or failed action gets feedback, for example
`"not enough energy (need 80)"` or `"cannot move W: edge of the world"`. That
feedback appears in the next observation and in memory.

### Genome

`survival_priority, reproduction_priority, exploration_rate, risk_tolerance,
resource_efficiency, cooperation_tendency, aggression_tendency, curiosity,
memory_retention`, all in [0, 1].

For the Qwen brain, the genome is shown only as *innate dispositions* in the
prompt. No code branches on it. Two traits also have physical effects, each
with a trade-off. You can switch these off with
`organism.genome_physical_effects: false`:

* `resource_efficiency`: more energy per food unit, but higher metabolism.
* `memory_retention`: larger long-term memory, which costs energy upkeep per item.

### Memory

* Short-term memory is a FIFO of recent events (`short_term_capacity`).
* Important events are promoted to long-term memory when they leave short-term
  memory. Long-term memory is capacity-limited and costs energy per item per tick.
* Inheritance (`memory.inheritance_mode`):
  * `none`: Mode A.
  * `selected`: Mode B. The child gets the parent's most important memories.
  * `compressed`: Mode C. The child gets deterministic factual summaries,
    such as "food was obtained at [x, y]" or "O9 caused harm".
  * `genome_only`: Mode D. Currently the same as `none` for memory. It is kept
    as its own label so experiment records stay explicit.

### Creator-blind prompts

Prompts describe only the organism's body, surroundings, memories and possible
actions. They never mention humans, simulations, models, Python or
experiments. They never state a goal such as "survive". A unit test enforces
this. By default the prompt does not explain death either: organisms see
their energy and health, and they see others "stop moving".
`brain.reveal_mechanics: true` adds a factual description of the body's rules
as an experimental variable. Bump `PROMPT_VERSION` in `brains/prompts.py`
whenever the wording changes.

### Decision caching (LLM cost control)

With `brain.decision.every_tick: false`, an organism repeats its last
repeatable action (`move`, `consume` or `rest`) until any of these happens:

* its last action failed
* its energy or health changed past a threshold
* a message arrived
* it is standing on a hazard
* the food it was eating ran out
* its neighbours changed
* `max_interval` ticks have passed

Each tick, every organism that needs a decision is sent in one batch, and HTTP
backends run the requests concurrently.

---

## Environments and experiments

| file | purpose |
|---|---|
| `baseline.yaml` | Env 1: abundant food, validates the system |
| `env2_limited.yaml` | Env 2: limited food, competition |
| `env3_moving.yaml` | Env 3: drifting food hotspots, exploration |
| `env4_hazards.yaml` | Env 4: hazard zones |
| `env5_change.yaml` | Env 5: hotspots and hazards relocate periodically |
| `env6_competition.yaml` | Env 6: 40 founder lineages, scarce food |
| `env7_adaptive.yaml` | Env 7: overharvesting lowers fertility; crowding creates hazard |
| `exp_a_no_evolution.yaml` | A: identical fixed genome, no mutation |
| `exp_b_mutation.yaml` | B: identical founders, mutation, weak selection |
| `exp_c_reproduction.yaml` | C: random founders, reproduction + mutation (selection) |
| `exp_d_scarcity.yaml` | D: scarcity |
| `exp_e_cooperation.yaml` | E: sharing enabled, kin visible |
| `exp_f_competition.yaml` | F: scarcity + attack + sharing |
| `exp_g_memory_inheritance.yaml` | G: memory inheritance (vary `memory.inheritance_mode`) |
| `qwen_first20.yaml` | milestone: 20 Qwen organisms |
| `qwen_mock_dryrun.yaml` | full Qwen pipeline without a model |

Each config has `description` and `hypothesis` fields. Write the hypothesis
**before** running, and keep configs fixed after you observe results. If a new
behaviour suggests a change, make it a new config (new name), not an edit to
the old one. For conclusions, run several seeds, for example
`--seeds 1 2 3 4 5 6 7 8`.

### A note on the rule brain

`RuleBrain` maps the genome to behaviour through hand-written heuristics. It
exists to validate the evolutionary machinery cheaply and to act as a control.
Selection on its genome is real, but its behaviours are **designed, not
emergent**. Emergence claims should come from LLM-brain runs compared against
rule and random controls.

---

## Outputs (`runs/<name>_seed<seed>/`)

| file | contents |
|---|---|
| `simulation.json` | seed, full config, simulation/environment/prompt/model versions |
| `summary.json` | final counts, surviving lineages, death causes |
| `organisms.json` | every organism that ever lived: parents, lineage, genome, lifetime stats |
| `events.jsonl` | every action (energy before/after, reason, cached?), births, deaths, environment changes, shares, attacks |
| `genomes.jsonl` | genome at birth + mutated traits |
| `communications.jsonl` | every message (sent vs. received text) |
| `brain_calls.jsonl` | raw model output, parsed action, latency (optionally full prompts) |
| `metrics.csv` | per tick: population, births/deaths by cause, energy, lifespan, generations, lineages, diversity, trait mean/std, action counts, brain calls |
| `snapshots.jsonl` | world state + organism positions every `snapshot_every` ticks |
| `dashboard.html` | self-contained dashboard (world replay, charts, trait distributions, lineage tree, event timeline, messages) |
| `checkpoints/` | resumable pickles |

## Reproducibility

All randomness comes from named `random.Random` streams derived from the seed
(`world`, `env`, `order`, `brain`, `evolution`, `init`). With the rule or
random brain, the same seed and config reproduce a run exactly, and resuming
from a checkpoint gives the same result as an uninterrupted run. The tests
check both. With Qwen, the model output is the only non-deterministic part.
Each request carries a seed drawn from the brain stream, so Ollama at a fixed
temperature is usually repeatable on the same machine.

## Safety boundary

Everything is virtual. Organisms have no tools except the action API above.
They have no network, filesystem or shell access, and nothing outside the
world state. The inference endpoint is on the experimenter's side and is
limited to localhost unless `brain.allow_remote_endpoint: true`. Model
weights are frozen. Nothing here trains the model or reaches real
infrastructure.

## Next steps

* Run each environment over many seeds with rule and random controls, then with Qwen.
* Behavioural analysis of `brain_calls.jsonl` and `communications.jsonl`, for
  example message vocabulary over generations and mentions of origin, death,
  self and others.
* A FastAPI + React live dashboard if the static one becomes limiting.
* Later: evolving the neural policy itself (for example LoRA adapters as a
  heritable genome) through the same `Brain` interface.
