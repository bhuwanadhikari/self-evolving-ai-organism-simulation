"""Lineage tree construction and rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_organisms(run_dir: str | Path) -> list[dict[str, Any]]:
    return json.loads((Path(run_dir) / "organisms.json").read_text())


def build_tree(organisms: list[dict[str, Any]]) -> tuple[dict[str, list[str]], list[str], dict[str, dict]]:
    """Return (children map keyed by first parent, founder ids, id->record)."""
    by_id = {o["id"]: o for o in organisms}
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for o in sorted(organisms, key=lambda o: int(o["id"][1:])):
        if o["parent_ids"]:
            children[o["parent_ids"][0]].append(o["id"])
        else:
            roots.append(o["id"])
    return children, roots, by_id


def descendants(oid: str, children: dict[str, list[str]]) -> int:
    stack, n = list(children.get(oid, [])), 0
    while stack:
        c = stack.pop()
        n += 1
        stack.extend(children.get(c, []))
    return n


def lineage_summary(organisms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for o in organisms:
        groups[o["lineage_id"]].append(o)
    out = []
    for lid, members in groups.items():
        out.append({
            "lineage_id": lid,
            "total": len(members),
            "alive": sum(1 for m in members if m["alive"]),
            "max_generation": max(m["generation"] for m in members),
            "extinct_at": None if any(m["alive"] for m in members)
            else max(m["death_tick"] or 0 for m in members),
        })
    return sorted(out, key=lambda r: (-r["alive"], -r["total"], r["lineage_id"]))


def ascii_tree(organisms: list[dict[str, Any]], max_depth: int = 6, max_children: int = 6,
               only_surviving: bool = False) -> str:
    children, roots, by_id = build_tree(organisms)
    alive_desc: dict[str, bool] = {}

    def has_alive(oid: str) -> bool:
        if oid not in alive_desc:
            alive_desc[oid] = by_id[oid]["alive"] or any(has_alive(c) for c in children.get(oid, []))
        return alive_desc[oid]

    lines: list[str] = []

    def label(oid: str) -> str:
        o = by_id[oid]
        status = "alive" if o["alive"] else f"died t{o['death_tick']} ({o['death_cause']})"
        return f"{oid} [{o['lineage_id']} g{o['generation']}] {status}, {o['offspring_count']} offspring"

    def walk(oid: str, prefix: str, last: bool, depth: int) -> None:
        lines.append(prefix + ("└── " if last else "├── ") + label(oid))
        kids = [c for c in children.get(oid, []) if not only_surviving or has_alive(c)]
        if depth >= max_depth:
            if kids:
                lines.append(prefix + ("    " if last else "│   ") + f"└── … {descendants(oid, children)} descendants")
            return
        shown = kids[:max_children]
        for i, c in enumerate(shown):
            walk(c, prefix + ("    " if last else "│   "), i == len(shown) - 1 and len(kids) <= max_children, depth + 1)
        if len(kids) > max_children:
            lines.append(prefix + ("    " if last else "│   ") + f"└── … {len(kids) - max_children} more children")

    rs = [r for r in roots if not only_surviving or has_alive(r)]
    lines.append("Generation 0")
    for i, r in enumerate(rs):
        walk(r, "", i == len(rs) - 1, 0)
    return "\n".join(lines)


def tree_json(organisms: list[dict[str, Any]], max_nodes: int = 3000) -> list[dict[str, Any]]:
    """Compact node list for the dashboard (id, parent, lineage, gen, birth, death, alive)."""
    nodes = sorted(organisms, key=lambda o: int(o["id"][1:]))[:max_nodes]
    return [{"id": o["id"], "p": o["parent_ids"][0] if o["parent_ids"] else None, "l": o["lineage_id"],
             "g": o["generation"], "b": o["birth_tick"], "d": o["death_tick"], "a": o["alive"],
             "c": o["death_cause"], "o": o["offspring_count"]} for o in nodes]
