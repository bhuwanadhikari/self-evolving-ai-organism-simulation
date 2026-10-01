from pathlib import Path

import pytest

from simulation.config import load_config
from simulation.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_with_same_seed(cfg, tmp_path):
    Simulation(cfg, tmp_path / "a").run()
    Simulation(cfg, tmp_path / "b").run()
    assert (tmp_path / "a/metrics.csv").read_text() == (tmp_path / "b/metrics.csv").read_text()
    assert (tmp_path / "a/events.jsonl").read_text() == (tmp_path / "b/events.jsonl").read_text()


def test_different_seed_differs(cfg, tmp_path):
    Simulation(cfg, tmp_path / "a").run()
    cfg.seed = 7
    Simulation(cfg, tmp_path / "b").run()
    assert (tmp_path / "a/metrics.csv").read_text() != (tmp_path / "b/metrics.csv").read_text()


def test_resume_matches_uninterrupted_run(cfg, tmp_path):
    cfg.ticks = 60
    Simulation(cfg, tmp_path / "full").run()
    Simulation(cfg, tmp_path / "part").run(30)
    Simulation.resume(tmp_path / "part").run(60)
    assert (tmp_path / "full/metrics.csv").read_text() == (tmp_path / "part/metrics.csv").read_text()
    assert (tmp_path / "full/genomes.jsonl").read_text() == (tmp_path / "part/genomes.jsonl").read_text()


def test_outputs_written(cfg, tmp_path):
    Simulation(cfg, tmp_path / "r").run()
    for f in ("simulation.json", "organisms.json", "events.jsonl", "genomes.jsonl", "communications.jsonl",
              "metrics.csv", "snapshots.jsonl", "summary.json"):
        assert (tmp_path / "r" / f).exists(), f


def test_energy_pressure_kills_without_food(cfg, tmp_path):
    cfg.resources.initial_count = 0
    cfg.resources.regeneration_rate = 0
    cfg.ticks = 300
    s = Simulation(cfg, tmp_path / "r")
    summary = s.run()
    assert summary["extinct"] and summary["death_causes"] == {"starvation": 10}


def test_dashboard_builds(cfg, tmp_path):
    from analysis.visualization import build_dashboard
    Simulation(cfg, tmp_path / "r").run()
    html = build_dashboard(tmp_path / "r").read_text()
    assert "/*__DATA__*/null" not in html


@pytest.mark.parametrize("path", sorted((ROOT / "experiments").glob("*.yaml")), ids=lambda p: p.name)
def test_experiment_configs_load(path):
    load_config(path)
