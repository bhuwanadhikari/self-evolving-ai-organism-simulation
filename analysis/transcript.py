"""Build a self-contained HTML transcript of every prompt sent to an LLM
brain and the model's raw response.

    python -m analysis.transcript runs/openrouter_gpt_oss_20b_seed42

Full system/user prompts are only present if the run used
`brain.log_prompts: true` (default False); otherwise the transcript still
shows each call's raw model response and parsed action.

Writes runs/.../transcript_<timestamp>.html (a fresh, timestamped file every
time it's built, so re-running the same experiment never overwrites an
earlier transcript). Open it in any browser; no server needed.
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _call_html(c: dict[str, Any]) -> str:
    prompt = c.get("prompt") or {}
    system, user = prompt.get("system", ""), prompt.get("user", "")
    parsed = json.dumps({"action": c.get("action", ""), "args": c.get("args", {}),
                         "reason": c.get("reason", "")}, indent=2)
    parts = [f'<summary>t{_esc(c.get("tick"))} &middot; {_esc(c.get("organism", ""))} '
             f'&middot; <b>{_esc(c.get("action", ""))}</b>'
             + (f' <span class="err">{_esc(c["parse_error"])}</span>' if c.get("parse_error") else "")
             + f' <span class="reason">{_esc(c.get("reason", ""))}</span>'
             + f' <span class="lat">{_esc(c.get("latency", ""))}s</span></summary>']
    if system:
        parts.append(f"<h4>System prompt</h4><pre>{_esc(system)}</pre>")
    if user:
        parts.append(f"<h4>User prompt</h4><pre>{_esc(user)}</pre>")
    parts.append(f"<h4>Model response (raw)</h4><pre>{_esc(c.get('raw', ''))}</pre>")
    parts.append(f"<h4>Parsed</h4><pre>{_esc(parsed)}</pre>")
    return "<details class=\"call\">" + "".join(parts) + "</details>"


STYLE = """
body { font: 14px/1.4 -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px;
       margin: 2rem auto; padding: 0 1rem; color: #222; }
h1 { font-size: 1.2rem; }
.note { background: #fff3cd; padding: .5rem 1rem; border-radius: 4px; }
details.call { border: 1px solid #ddd; border-radius: 6px; margin: .5rem 0; padding: .4rem .8rem; }
summary { cursor: pointer; font-family: ui-monospace, monospace; }
summary .reason { color: #555; margin-left: .5rem; }
summary .lat { color: #999; float: right; }
summary .err { color: #b00; margin-left: .5rem; }
h4 { margin: .6rem 0 .2rem; font-size: .85rem; color: #555; }
pre { white-space: pre-wrap; word-break: break-word; background: #f7f7f7; padding: .5rem; border-radius: 4px; margin: 0; }
"""


def build_transcript(run_dir: str | Path, timestamp: str | None = None) -> Path:
    run_dir = Path(run_dir)
    calls = _read_jsonl(run_dir / "brain_calls.jsonl")
    has_prompts = any(c.get("prompt") for c in calls)
    note = "" if has_prompts else (
        '<p class="note">This run used <code>brain.log_prompts: false</code> (the default), '
        "so only the model's raw responses were recorded, not the system/user prompts. "
        "Set <code>brain.log_prompts: true</code> in the config and re-run to capture full prompts.</p>")
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    doc = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f"<title>Prompt transcript — {_esc(run_dir.name)} — {_esc(ts)}</title>"
        f"<style>{STYLE}</style></head><body>"
        f"<h1>Prompt transcript — {_esc(run_dir.name)}</h1>"
        f"<p>Generated {_esc(ts)} UTC &middot; {len(calls)} model calls.</p>{note}"
        + "".join(_call_html(c) for c in calls)
        + "</body></html>"
    )
    out = run_dir / f"transcript_{ts}.html"
    out.write_text(doc)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(build_transcript(d))
