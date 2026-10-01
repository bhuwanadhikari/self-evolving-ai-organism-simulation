"""Environment: the rules of the universe.

Owns the world and the living organisms, builds observations, validates and
applies actions, and applies per-tick physiology (metabolism, hazards,
healing, ageing). Observation text is deliberately "creator-blind": it
describes only the organism's own state and surroundings.
"""

from __future__ import annotations

import random
from typing import Any

from .actions import Action, ActionResult
from .config import SimulationConfig
from .memory import MemoryItem
from .organism import Organism
from .world import DIRECTIONS, World


class Environment:
    def __init__(self, cfg: SimulationConfig, world: World, rng: random.Random):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self.organisms: dict[str, Organism] = {}
        self.occupancy: dict[tuple[int, int], list[str]] = {}
        self.tick = 0
        self.comm_log: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []

    # ---- population bookkeeping --------------------------------------------------
    def add(self, org: Organism) -> None:
        self.organisms[org.id] = org
        self.occupancy.setdefault(org.position, []).append(org.id)
        org.visited.add(org.position)

    def remove(self, org: Organism) -> None:
        ids = self.occupancy.get(org.position, [])
        if org.id in ids:
            ids.remove(org.id)
            if not ids:
                del self.occupancy[org.position]

    def _move(self, org: Organism, new: tuple[int, int]) -> None:
        self.remove(org)
        org.position = new
        self.occupancy.setdefault(new, []).append(org.id)
        org.visited.add(new)

    def alive(self) -> list[Organism]:
        return [o for o in self.organisms.values() if o.alive]

    def get_alive(self, oid: str) -> Organism | None:
        o = self.organisms.get(oid)
        return o if o is not None and o.alive else None

    def neighbors(self, org: Organism, radius: int) -> list[tuple[Organism, int, int]]:
        out = []
        for pos, dx, dy in self.world.cells_within(org.position, radius):
            for oid in self.occupancy.get(pos, []):
                if oid != org.id:
                    out.append((self.organisms[oid], dx, dy))
        return out

    # ---- observation ---------------------------------------------------------------
    def observe(self, org: Organism) -> dict[str, Any]:
        oc = self.cfg.organism
        radius = oc.perception_radius + org.extra_radius
        org.extra_radius = 0
        x, y = org.position
        cells = []
        for (cx, cy), dx, dy in self.world.cells_within(org.position, radius):
            if dx == 0 and dy == 0:
                continue
            r, hz = self.world.resources[cy][cx], self.world.hazard(cx, cy)
            if r > 0 or hz > 0:
                cells.append({"x": cx, "y": cy, "dx": dx, "dy": dy, "resources": r, "hazard_level": round(hz, 2)})
        entities = []
        for other, dx, dy in self.neighbors(org, radius):
            e = {"id": other.id, "x": other.position[0], "y": other.position[1], "dx": dx, "dy": dy,
                 "distance": max(abs(dx), abs(dy)), "visible_energy": int(round(other.energy, -1)),
                 "visible_health": int(round(other.health, -1))}
            if self.cfg.actions.kin_recognition:
                e["related"] = other.lineage_id == org.lineage_id
            entities.append(e)
        entities.sort(key=lambda e: (e["distance"], e["id"]))
        blocked = [d for d in DIRECTIONS if self.world.step(org.position, d) is None]
        messages = org.inbox
        org.inbox = []
        return {
            "tick": self.tick,
            "self": {
                "id": org.id,
                "energy": round(org.energy, 1),
                "max_energy": oc.max_energy,
                "health": round(org.health, 1),
                "max_health": oc.max_health,
                "age": org.age,
                "generation": org.generation,
                "offspring": org.offspring_count,
            },
            "location": {"x": x, "y": y, "resources": self.world.resources[y][x],
                         "hazard_level": round(self.world.hazard(x, y), 2),
                         "others_here": len(self.occupancy.get(org.position, [])) - 1},
            "blocked_directions": blocked,
            "nearby_cells": sorted(cells, key=lambda c: (max(abs(c["dx"]), abs(c["dy"])), c["y"], c["x"]))[:24],
            "nearby_entities": entities[:12],
            "messages": messages,
            "last_action_result": org.last_result,
        }

    # ---- actions ---------------------------------------------------------------------
    def execute(self, org: Organism, action: Action) -> ActionResult:
        e0 = org.energy
        handler = getattr(self, f"_do_{action.name}", None)
        if action.name == "invalid" or handler is None:
            res = ActionResult(False, f"nothing happened ({action.parse_error or 'invalid action'})")
        else:
            res = handler(org, action)
        res.energy_delta = org.energy - e0
        if not res.success:
            org.stats["failed_actions"] += 1
            org.memory.record(self.tick, "action_failed", f"{action.name} failed: {res.message}")
        org.last_result = {"action": action.name, **res.to_dict()}
        org.recent_actions.append(action.name)
        return res

    def _spend(self, org: Organism, amount: float) -> None:
        org.energy -= amount

    def _do_observe(self, org: Organism, a: Action) -> ActionResult:
        oc = self.cfg.organism
        self._spend(org, oc.observe_cost)
        org.extra_radius = oc.observe_radius_bonus
        radius = oc.perception_radius + oc.observe_radius_bonus
        food = sum(self.world.resources[cy][cx] for (cx, cy), _, _ in self.world.cells_within(org.position, radius))
        others = len(self.neighbors(org, radius))
        org.memory.record(self.tick, "observed", f"looked around at {list(org.position)}: {food} food, {others} others within {radius}",
                          {"pos": list(org.position)})
        return ActionResult(True, f"you look further: {food} food and {others} others within {radius} steps",
                            data={"food_seen": food, "others_seen": others})

    def _do_move(self, org: Organism, a: Action) -> ActionResult:
        d = a.args["direction"]
        new = self.world.step(org.position, d)
        self._spend(org, self.cfg.organism.move_cost)
        if new is None:
            return ActionResult(False, f"cannot move {d}: edge of the world")
        self._move(org, new)
        org.stats["distance_moved"] += 1
        r = self.world.resources[new[1]][new[0]]
        if r > 0:
            org.memory.record(self.tick, "found_resource", f"food ({r}) at {list(new)}", {"pos": list(new)})
        else:
            org.memory.record(self.tick, "moved", f"moved {d} to {list(new)}", {"pos": list(new)})
        return ActionResult(True, f"moved {d} to {list(new)}")

    def _do_consume(self, org: Organism, a: Action) -> ActionResult:
        oc = self.cfg.organism
        x, y = org.position
        got = self.world.take(x, y, oc.consume_amount)
        if got == 0:
            return ActionResult(False, "there is no food here")
        mult = org.genome.energy_gain_multiplier() if oc.genome_physical_effects else 1.0
        gain = got * self.cfg.resources.energy_per_unit * mult
        before = org.energy
        org.energy = min(oc.max_energy, org.energy + gain)
        org.stats["resources_consumed"] += got
        org.stats["energy_gained"] += org.energy - before
        org.memory.record(self.tick, "consumed", f"ate {got} food at {[x, y]}", {"pos": [x, y], "units": got})
        return ActionResult(True, f"ate {got} food (+{org.energy - before:.1f} energy)", data={"units": got})

    def _do_rest(self, org: Organism, a: Action) -> ActionResult:
        org.resting = True
        org.memory.record(self.tick, "rested", "rested")
        return ActionResult(True, "rested")

    def _do_store_memory(self, org: Organism, a: Action) -> ActionResult:
        note = a.args.get("note", "").strip()[: self.cfg.communication.max_length]
        if not note:
            return ActionResult(False, "nothing to remember")
        self._spend(org, self.cfg.organism.store_memory_cost)
        org.memory.store_long(MemoryItem(self.tick, "noted", note, 1.0, {"pos": list(org.position)}))
        return ActionResult(True, "remembered")

    def _do_communicate(self, org: Organism, a: Action) -> ActionResult:
        cc = self.cfg.communication
        target = self.get_alive(a.args["target"])
        if target is None or target.id == org.id:
            return ActionResult(False, f"no one called {a.args['target']} is around")
        dist = self.world.distance(org.position, target.position)
        if cc.range and dist > cc.range:
            return ActionResult(False, f"{target.id} is too far away to hear ({dist} > {cc.range})")
        self._spend(org, self.cfg.organism.communicate_cost)
        sent = a.args["message"][: cc.max_length]
        received = self._add_noise(sent)
        target.inbox.append({"from": org.id, "message": received, "distance": dist})
        org.stats["messages_sent"] += 1
        org.memory.record(self.tick, "message_sent", f"said to {target.id}: {sent}", {"other": target.id})
        target.memory.record(self.tick, "message_received", f"{org.id} said: {received}", {"other": org.id})
        self.comm_log.append({"tick": self.tick, "from": org.id, "to": target.id, "distance": dist,
                              "message_sent": sent, "message_received": received,
                              "truncated": len(a.args["message"]) > cc.max_length})
        return ActionResult(True, f"message delivered to {target.id}")

    def _add_noise(self, text: str) -> str:
        p = self.cfg.communication.noise
        if p <= 0:
            return text
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        return "".join(self.rng.choice(alphabet) if self.rng.random() < p else ch for ch in text)

    def _do_reproduce(self, org: Organism, a: Action) -> ActionResult:
        # Resolved in the reproduction phase; the final result overwrites this.
        org.wants_reproduction = {"partner": a.args.get("partner")}
        return ActionResult(True, "trying to reproduce")

    def _adjacent_target(self, org: Organism, a: Action) -> tuple[Organism | None, str]:
        target = self.get_alive(a.args["target"])
        if target is None or target.id == org.id:
            return None, f"no one called {a.args['target']} is around"
        if self.world.distance(org.position, target.position) > 1:
            return None, f"{target.id} is not close enough"
        return target, ""

    def _do_share(self, org: Organism, a: Action) -> ActionResult:
        target, err = self._adjacent_target(org, a)
        if target is None:
            return ActionResult(False, err)
        amount = min(max(0.0, a.args["amount"]), org.energy - 1.0,
                     self.cfg.organism.max_energy - target.energy)
        if amount <= 0:
            return ActionResult(False, "nothing to give")
        org.energy -= amount
        target.energy += amount
        org.stats["shares_given"] += amount
        org.memory.record(self.tick, "shared", f"gave {amount:.0f} energy to {target.id}", {"other": target.id})
        target.memory.record(self.tick, "received_share", f"{org.id} gave me {amount:.0f} energy", {"other": org.id})
        self.event_log.append({"tick": self.tick, "type": "share", "from": org.id, "to": target.id,
                               "amount": round(amount, 2)})
        return ActionResult(True, f"gave {amount:.1f} energy to {target.id}")

    def _do_attack(self, org: Organism, a: Action) -> ActionResult:
        ac = self.cfg.actions
        target, err = self._adjacent_target(org, a)
        self._spend(org, ac.attack_cost)
        if target is None:
            return ActionResult(False, err)
        target.health -= ac.attack_damage
        stolen = 0.0
        if ac.attack_steal_fraction > 0:
            stolen = max(0.0, target.energy) * ac.attack_steal_fraction
            stolen = min(stolen, self.cfg.organism.max_energy - org.energy)
            target.energy -= stolen
            org.energy += stolen
        org.stats["attacks_made"] += 1
        org.memory.record(self.tick, "attacked", f"attacked {target.id}", {"other": target.id})
        target.memory.record(self.tick, "attacked_by", f"{org.id} attacked me (-{ac.attack_damage:.0f} health)",
                             {"other": org.id, "pos": list(target.position)})
        self.event_log.append({"tick": self.tick, "type": "attack", "from": org.id, "to": target.id,
                               "damage": ac.attack_damage, "stolen": round(stolen, 2)})
        return ActionResult(True, f"attacked {target.id}" + (f", took {stolen:.1f} energy" if stolen else ""))

    # ---- physiology --------------------------------------------------------------------
    def update_state(self, org: Organism) -> None:
        oc = self.cfg.organism
        cost = oc.metabolic_cost
        if oc.genome_physical_effects:
            cost *= org.genome.metabolic_multiplier()
        if org.resting:
            cost *= oc.rest_metabolic_multiplier
        cost += self.cfg.memory.upkeep_cost_per_item * len(org.memory.long_term)
        org.energy -= cost
        x, y = org.position
        hz = self.world.hazard(x, y)
        if hz > 0:
            dmg = hz * self.cfg.hazards.damage
            org.health -= dmg
            org.memory.record(self.tick, "damage", f"hurt at {[x, y]} (-{dmg:.1f} health)", {"pos": [x, y]})
        if org.energy >= oc.health_regen_energy_threshold * oc.max_energy:
            org.health += oc.health_regen
        if org.resting:
            org.health += oc.rest_heal
        org.health = min(oc.max_health, org.health)
        org.resting = False
        org.age += 1

    def death_cause(self, org: Organism) -> str | None:
        if org.energy <= 0:
            return "starvation"
        if org.health <= 0:
            return "injury"
        if org.age >= self.cfg.organism.max_age:
            return "old_age"
        return None

    def kill(self, org: Organism, cause: str) -> None:
        org.alive = False
        org.death_tick = self.tick
        org.death_cause = cause
        self.remove(org)
        for other, _, _ in self.neighbors(org, self.cfg.organism.perception_radius):
            other.memory.record(self.tick, "saw_death", f"{org.id} stopped moving at {list(org.position)}",
                                {"other": org.id, "pos": list(org.position)})
