"""Structured run output.

Files written to runs/<name>_seed<seed>/:
  simulation.json        metadata: config, seed, versions, brain description
  organisms.json         every organism that ever lived (lifetime summary)
  events.jsonl           actions, births, deaths, environment events
  genomes.jsonl          genome of every organism at birth
  communications.jsonl   every message sent
  brain_calls.jsonl      raw language-model outputs (LLM brains only)
  metrics.csv            one row per tick
  snapshots.jsonl        world + organism positions every N ticks (for the dashboard)
  checkpoints/           resumable pickles
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, IO

JSONL_FILES = ("events", "genomes", "communications", "brain_calls", "snapshots")


class RunLogger:
    def __init__(self, run_dir: Path, append: bool = False):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self.files: dict[str, IO[str]] = {n: open(self.dir / f"{n}.jsonl", mode) for n in JSONL_FILES}
        self.metrics_path = self.dir / "metrics.csv"
        self._metrics_file: IO[str] | None = None
        self._writer: csv.DictWriter | None = None
        if append and self.metrics_path.exists():
            with open(self.metrics_path) as f:
                header = f.readline().strip().split(",")
            self._metrics_file = open(self.metrics_path, "a", newline="")
            self._writer = csv.DictWriter(self._metrics_file, fieldnames=header)
        elif not append and self.metrics_path.exists():
            self.metrics_path.unlink()

    def write(self, stream: str, record: dict[str, Any]) -> None:
        self.files[stream].write(json.dumps(record, separators=(",", ":"), default=str) + "\n")

    def metrics(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._metrics_file = open(self.metrics_path, "w", newline="")
            self._writer = csv.DictWriter(self._metrics_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)

    def write_json(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=str))

    def flush(self) -> None:
        for f in self.files.values():
            f.flush()
        if self._metrics_file:
            self._metrics_file.flush()

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        if self._metrics_file:
            self._metrics_file.close()


def truncate_after(run_dir: Path, tick: int) -> None:
    """Drop log lines newer than `tick` (used when resuming from a checkpoint)."""
    for n in JSONL_FILES:
        p = run_dir / f"{n}.jsonl"
        if not p.exists():
            continue
        keep = []
        for line in p.read_text().splitlines():
            try:
                if json.loads(line).get("tick", 0) <= tick:
                    keep.append(line)
            except json.JSONDecodeError:
                continue
        p.write_text("".join(k + "\n" for k in keep))
    m = run_dir / "metrics.csv"
    if m.exists():
        lines = m.read_text().splitlines()
        if lines:
            out = [lines[0]] + [ln for ln in lines[1:] if int(ln.split(",", 1)[0]) <= tick]
            m.write_text("\n".join(out) + "\n")
