"""Build a self-contained HTML dashboard from a run directory.

    python -m analysis.visualization runs/baseline_seed42

Writes runs/.../dashboard.html (open it in any browser; no server needed).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from simulation.genome import TRAITS

from .lineage import load_organisms, tree_json

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def _read_jsonl(path: Path, keep=lambda r: True, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if keep(r):
                out.append(r)
                if limit and len(out) > limit * 2:
                    out = out[-limit:]
    return out[-limit:] if limit else out


def _read_metrics(path: Path, max_points: int = 1500) -> dict[str, list]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    step = max(1, len(rows) // max_points)
    rows = rows[::step]
    cols: dict[str, list] = {}
    for k in rows[0].keys() if rows else []:
        vals = []
        for r in rows:
            try:
                v = float(r[k])
                vals.append(int(v) if v.is_integer() else round(v, 4))
            except ValueError:
                vals.append(r[k])
        cols[k] = vals
    return cols


def build_dashboard(run_dir: str | Path, max_frames: int = 400, max_events: int = 5000) -> Path:
    run = Path(run_dir)
    meta = json.loads((run / "simulation.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    organisms = load_organisms(run)
    snaps = _read_jsonl(run / "snapshots.jsonl")
    if len(snaps) > max_frames:
        step = len(snaps) / max_frames
        snaps = [snaps[int(i * step)] for i in range(max_frames)] + [snaps[-1]]
    events = _read_jsonl(run / "events.jsonl", keep=lambda r: r.get("type") != "action", limit=max_events)
    comms = _read_jsonl(run / "communications.jsonl", limit=400)
    lineage_sizes = Counter(o["lineage_id"] for o in organisms)
    tree = tree_json(organisms)
    cfg = meta["config"]
    data = {
        "meta": meta,
        "summary": summary,
        "world": {"width": cfg["world"]["width"], "height": cfg["world"]["height"],
                  "max_per_cell": cfg["resources"]["max_per_cell"]},
        "snapshots": snaps,
        "metrics": _read_metrics(run / "metrics.csv"),
        "traits": list(TRAITS),
        "tree": tree,
        "tree_truncated": len(tree) < len(organisms),
        "top_lineages": [l for l, _ in lineage_sizes.most_common(7)],
        "n_lineages": len(lineage_sizes),
        "events": events,
        "events_truncated": len(events) >= max_events,
        "comms": comms,
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = run / "dashboard.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(build_dashboard(d))
