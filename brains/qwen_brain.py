"""Qwen (7B-class) language-model brain.

Backends (all run on the experimenter's machine):

* ollama        - Ollama server, default http://localhost:11434 (recommended)
* openai        - any OpenAI-compatible local server (vLLM, llama.cpp, LM Studio)
* transformers  - load weights in-process with Hugging Face transformers
* mock          - no model; returns RuleBrain decisions through the full
                  prompt/parse pipeline (for tests and dry runs)

Safety boundary: the endpoint must be localhost unless
`brain.allow_remote_endpoint` is set. The organism never gets any tool; the
model's text output is only ever parsed into one of the simulator's actions.
Model weights are frozen; nothing here trains the model.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from simulation.config import BrainConfig

from .base import Brain, DecisionContext
from .prompts import PROMPT_VERSION, system_prompt, user_prompt
from .rule_brain import RuleBrain

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first well-formed JSON object out of model text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = None
    return None


def normalize(obj: dict[str, Any] | None, allowed: list[str], raw: str) -> dict[str, Any]:
    """Coerce common model output variations into the proposal schema."""
    if obj is None:
        m = re.search(r"\b(" + "|".join(map(re.escape, allowed)) + r")\b", raw.lower())
        if not m:
            return {"action": "invalid", "args": {}, "reason": "", "_parse_error": "no JSON in output"}
        obj = {"action": m.group(1)}
    action = str(obj.get("action", "")).strip().lower().replace("()", "")
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    for k in ("direction", "target", "message", "note", "amount", "partner"):
        if k in obj and k not in args:
            args[k] = obj[k]
    return {"action": action, "args": args, "reason": str(obj.get("reason", ""))[:200]}


class _HTTPBackend:
    def __init__(self, cfg: BrainConfig):
        host = urlparse(cfg.endpoint).hostname or ""
        if host not in LOCAL_HOSTS and not cfg.allow_remote_endpoint:
            raise ValueError(f"Refusing non-local inference endpoint '{cfg.endpoint}'. "
                             "Set brain.allow_remote_endpoint: true if this is intended.")
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        if cfg.api_key_env and not self.api_key:
            raise ValueError(f"Environment variable '{cfg.api_key_env}' is not set (needed for {cfg.endpoint}).")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.cfg.endpoint.rstrip("/") + path, data=json.dumps(payload).encode(),
                                     headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            # Surface the provider's error body (e.g. OpenRouter's reason for a 403).
            body = exc.read().decode(errors="replace")[:500]
            raise urllib.error.HTTPError(exc.url, exc.code, f"{exc.reason}: {body}", exc.headers, None) from None


class OllamaBackend(_HTTPBackend):
    def ping(self) -> None:
        with urllib.request.urlopen(self.cfg.endpoint.rstrip("/") + "/api/tags", timeout=10) as r:
            models = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        if not any(m == self.cfg.model or m.startswith(self.cfg.model + ":") for m in models):
            raise RuntimeError(f"Model '{self.cfg.model}' not found in Ollama. Run: ollama pull {self.cfg.model}")

    def generate(self, system: str, user: str, seed: int) -> str:
        out = self._post("/api/chat", {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens, "seed": seed},
        })
        return out["message"]["content"] or ""


class OpenAIBackend(_HTTPBackend):
    def ping(self) -> None:
        req = urllib.request.Request(self.cfg.endpoint.rstrip("/") + "/v1/models", headers=self._headers())
        urllib.request.urlopen(req, timeout=10).close()

    def generate(self, system: str, user: str, seed: int) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "seed": seed,
        }
        if self.cfg.reasoning_effort:
            payload["reasoning"] = {"effort": self.cfg.reasoning_effort}
        out = self._post("/v1/chat/completions", payload)
        return out["choices"][0]["message"]["content"] or ""


class TransformersBackend:
    """In-process batched generation. Requires `pip install torch transformers accelerate`."""

    def __init__(self, cfg: BrainConfig):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        name = cfg.model if "/" in cfg.model else "Qwen/Qwen2.5-7B-Instruct"
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
        self.model.eval()

    def ping(self) -> None:
        pass

    def generate_batch(self, pairs: list[tuple[str, str]], seed: int) -> list[str]:
        import torch

        torch.manual_seed(seed)
        texts = [self.tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                              tokenize=False, add_generation_prompt=True) for s, u in pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.cfg.max_tokens,
                                      do_sample=self.cfg.temperature > 0,
                                      temperature=max(self.cfg.temperature, 1e-5),
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class MockBackend:
    """Runs the full prompt/parse path but answers with RuleBrain decisions."""

    def __init__(self, cfg: BrainConfig):
        self.rule = RuleBrain()

    def ping(self) -> None:
        pass


class QwenBrain(Brain):
    name = "qwen"

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.version = f"{cfg.backend}:{cfg.model}"
        backends = {"ollama": OllamaBackend, "openai": OpenAIBackend,
                    "transformers": TransformersBackend, "mock": MockBackend}
        if cfg.backend not in backends:
            raise ValueError(f"Unknown qwen backend: {cfg.backend}")
        self.backend = backends[cfg.backend](cfg)
        self.backend.ping()
        self.pool = ThreadPoolExecutor(max_workers=max(1, cfg.concurrency))
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "backend": self.cfg.backend, "model": self.cfg.model,
                "prompt_version": PROMPT_VERSION, "temperature": self.cfg.temperature}

    def _prompts(self, ctx: DecisionContext) -> tuple[str, str]:
        return (system_prompt(ctx.organism_id, ctx.allowed_actions, self.cfg.reveal_mechanics),
                user_prompt(ctx.observation, ctx.memories, ctx.genome))

    def _finish(self, ctx: DecisionContext, raw: str, latency: float, prompts: tuple[str, str]) -> dict[str, Any]:
        prop = normalize(extract_json(raw), ctx.allowed_actions, raw)
        prop["_raw"] = raw[:1000]
        prop["_latency"] = round(latency, 3)
        if self.cfg.log_prompts:
            prop["_prompt"] = {"system": prompts[0], "user": prompts[1]}
        return prop

    def _one(self, ctx: DecisionContext, seed: int, mock_rng: random.Random | None) -> dict[str, Any]:
        prompts = self._prompts(ctx)
        t0 = time.time()
        if isinstance(self.backend, MockBackend):
            raw = json.dumps(self.backend.rule.decide(ctx, mock_rng or random.Random(seed)))
        else:
            last: Exception | None = None
            for attempt in range(self.cfg.retries + 1):
                try:
                    raw = self.backend.generate(*prompts, seed=seed)
                    break
                except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                    last = exc
                    time.sleep(1.0 * (attempt + 1))
            else:
                raise RuntimeError(f"inference failed for {ctx.organism_id}: {last}")
        return self._finish(ctx, raw, time.time() - t0, prompts)

    def decide(self, ctx: DecisionContext, rng: random.Random) -> dict[str, Any]:
        return self.decide_batch([ctx], rng)[0]

    def decide_batch(self, ctxs: list[DecisionContext], rng: random.Random) -> list[dict[str, Any]]:
        # Seeds are drawn from the simulation RNG in a fixed order, so the
        # non-LLM parts stay deterministic regardless of model output timing.
        seeds = [rng.randrange(2**31) for _ in ctxs]
        self.calls += len(ctxs)
        if isinstance(self.backend, MockBackend):
            return [self._one(c, s, random.Random(s)) for c, s in zip(ctxs, seeds)]
        if isinstance(self.backend, TransformersBackend):
            out = []
            bs = max(1, self.cfg.concurrency)
            for i in range(0, len(ctxs), bs):
                chunk = ctxs[i:i + bs]
                prompts = [self._prompts(c) for c in chunk]
                t0 = time.time()
                raws = self.backend.generate_batch(prompts, seeds[i])
                lat = (time.time() - t0) / len(chunk)
                out += [self._finish(c, r, lat, p) for c, r, p in zip(chunk, raws, prompts)]
            return out
        return list(self.pool.map(lambda cs: self._one(cs[0], cs[1], None), zip(ctxs, seeds)))

    def close(self) -> None:
        self.pool.shutdown(wait=False)
