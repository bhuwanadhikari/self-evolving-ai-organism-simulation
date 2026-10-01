#!/usr/bin/env python3
"""Run an artificial-life experiment.

Examples
--------
  python run_simulation.py experiments/baseline.yaml
  python run_simulation.py experiments/scarcity.yaml --seeds 1 2 3 4 5
  python run_simulation.py experiments/qwen_first20.yaml --ticks 50
  python run_simulation.py experiments/baseline.yaml --set brain.type=random --set ticks=200
  python run_simulation.py --resume runs/baseline_seed42 --ticks 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from analysis.lineage import ascii_tree, lineage_summary, load_organisms
from analysis.transcript import build_transcript
from analysis.visualization import build_dashboard
from simulation.config import SimulationConfig, deep_update, load_config, parse_override
from simulation.simulation import Simulation


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Populate os.environ from a local .env file (KEY=VALUE per line). Never
    overwrites a variable already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="YAML/JSON experiment file")
    ap.add_argument("--seeds", type=int, nargs="+", help="run once per seed (overrides config seed)")
    ap.add_argument("--ticks", type=int, help="number of ticks (overrides config)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config value, e.g. --set evolution.mutation_rate=0.05")
    ap.add_argument("--out", help="output root directory (default: config logging.output_dir)")
    ap.add_argument("--resume", metavar="RUN_DIR", help="resume a run from its latest checkpoint")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--tree", action="store_true", help="print the lineage tree of surviving lineages")
    ap.add_argument("--no-transcript", action="store_true",
                    help="skip writing the timestamped prompt/response transcript HTML file")
    args = ap.parse_args(argv)

    if args.resume:
        sim = Simulation.resume(args.resume)
        summary = sim.run(args.ticks)
        _report(sim.run_dir, summary, args)
        return 0

    if not args.config:
        ap.error("a config file is required (or --resume)")
    overrides: dict = {}
    for expr in args.set:
        deep_update(overrides, parse_override(expr))
    if args.ticks is not None:
        overrides["ticks"] = args.ticks
    if args.out:
        deep_update(overrides, {"logging": {"output_dir": args.out}})
    base: SimulationConfig = load_config(args.config, overrides)
    for seed in args.seeds or [base.seed]:
        cfg = load_config(args.config, {**overrides, "seed": seed})
        print(f"=== {cfg.name} | seed {seed} | brain {cfg.brain.type} | {cfg.ticks} ticks ===")
        sim = Simulation(cfg)
        summary = sim.run()
        _report(sim.run_dir, summary, args)
    return 0


def _report(run_dir: Path, summary: dict, args: argparse.Namespace) -> None:
    print("summary:", summary)
    orgs = load_organisms(run_dir)
    print("top lineages:")
    for r in lineage_summary(orgs)[:5]:
        print(f"  {r['lineage_id']:>5}: total {r['total']:4d}  alive {r['alive']:3d}  max gen {r['max_generation']}")
    if args.tree:
        print(ascii_tree(orgs, only_surviving=True))
    if not args.no_dashboard:
        print("dashboard:", build_dashboard(run_dir))
    if not args.no_transcript:
        print("transcript:", build_transcript(run_dir))
    print("output:", run_dir)


if __name__ == "__main__":
    sys.exit(main())
