import random

import pytest

from brains.base import DecisionContext
from brains.prompts import system_prompt, user_prompt
from brains.qwen_brain import QwenBrain, extract_json, normalize
from simulation.config import BrainConfig
from simulation.simulation import Simulation

ALLOWED = ["observe", "move", "consume", "rest", "communicate", "store_memory", "reproduce"]


@pytest.mark.parametrize("raw", [
    '{"action": "move", "args": {"direction": "N"}}',
    '```json\n{"action": "move", "args": {"direction": "N"}}\n```',
    'I think I will go north. {"action": "move", "args": {"direction": "N"}, "reason": "x"} ok',
    '{"action": "move", "direction": "N"}',
])
def test_parse_variants(raw):
    p = normalize(extract_json(raw), ALLOWED, raw)
    assert p["action"] == "move" and p["args"]["direction"] == "N"


def test_parse_garbage():
    p = normalize(extract_json("zzz"), ALLOWED, "zzz")
    assert p["action"] == "invalid"


def test_refuses_remote_endpoint():
    with pytest.raises(ValueError):
        QwenBrain(BrainConfig(type="qwen", backend="ollama", endpoint="http://example.com:11434"))


def test_prompts_are_creator_blind():
    obs = {"tick": 1, "self": {"id": "O1"}, "location": {}, "messages": []}
    text = (system_prompt("O1", ALLOWED, True) + user_prompt(obs, [], {"curiosity": 0.5})).lower()
    for word in ("simulation", "human", "qwen", "python", "model", "experiment", "created", "survive",
                 "program", "computer", "artificial"):
        assert word not in text, word


def test_mock_backend_full_loop(cfg, tmp_path):
    cfg.brain.type, cfg.brain.backend = "qwen", "mock"
    cfg.brain.decision.every_tick = False
    s = Simulation(cfg, tmp_path / "q")
    summary = s.run()
    assert summary["brain_calls"] > 0
    calls = (tmp_path / "q/brain_calls.jsonl").read_text().splitlines()
    assert len(calls) == summary["brain_calls"]
    # decision caching: fewer calls than organism-ticks
    assert summary["brain_calls"] < sum(1 for line in (tmp_path / "q/events.jsonl").read_text().splitlines()
                                        if '"type":"action"' in line)


def test_mock_decide_is_valid():
    b = QwenBrain(BrainConfig(type="qwen", backend="mock"))
    obs = {"self": {"energy": 50, "max_energy": 200}, "location": {"x": 0, "y": 0, "resources": 1, "hazard_level": 0},
           "nearby_cells": [], "nearby_entities": [], "blocked_directions": [], "messages": []}
    genome = {t: 0.5 for t in ("survival_priority", "reproduction_priority", "exploration_rate", "risk_tolerance",
                               "resource_efficiency", "cooperation_tendency", "aggression_tendency", "curiosity",
                               "memory_retention")}
    out = b.decide(DecisionContext("O1", obs, [], genome, ALLOWED), random.Random(0))
    assert out["action"] in ALLOWED


def test_ollama_backend_against_fake_server(cfg, tmp_path):
    """Exercise the real HTTP code path with a stand-in local server."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "qwen2.5:7b-instruct"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            self._send({"message": {"content": '{"action": "rest", "args": {}, "reason": "tired"}'}})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg.brain.type, cfg.brain.backend = "qwen", "ollama"
        cfg.brain.endpoint = f"http://127.0.0.1:{srv.server_address[1]}"
        cfg.ticks = 3
        summary = Simulation(cfg, tmp_path / "o").run()
    finally:
        srv.shutdown()
    assert summary["brain_calls"] == 30
    assert seen[0]["format"] == "json" and seen[0]["messages"][0]["role"] == "system"
