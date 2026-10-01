"""Compare finished runs (e.g. several seeds of two experiments).

    python -m analysis.compare runs/scarcity_seed* runs/baseline_seed*

Prints one row per run plus a per-experiment mean, using summary.json and the
final row of metrics.csv. Descriptive only - apply your own statistics.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from simulation.genome import TRAITS

COLS = ("ticks", "final_population", "total_births", "total_deaths", "max_generation")


def final_metrics(run: Path) -> dict[str, str]:
    with open(run / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def main(dirs: list[str]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    header = ["run", *COLS, "diversity", *[t[:10] for t in TRAITS]]
    print("\t".join(header))
    for d in dirs:
        run = Path(d)
        if not (run / "summary.json").exists():
            continue
        s = json.loads((run / "summary.json").read_text())
        m = final_metrics(run)
        name = json.loads((run / "simulation.json").read_text())["name"]
        row = {**{c: s[c] for c in COLS}, "diversity": float(m.get("genome_diversity", 0) or 0),
               **{t: float(m.get(f"trait_{t}", 0) or 0) for t in TRAITS}}
        groups[name].append(row)
        print("\t".join([run.name] + [f"{row[c]:.3g}" if isinstance(row[c], float) else str(row[c])
                                      for c in (*COLS, "diversity", *TRAITS)]))
    print("\nmeans by experiment:")
    for name, rows in groups.items():
        vals = [f"{statistics.fmean(r[c] for r in rows):.3g}" for c in (*COLS, "diversity", *TRAITS)]
        print("\t".join([f"{name} (n={len(rows)})", *vals]))


if __name__ == "__main__":
    main(sys.argv[1:])
