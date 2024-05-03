import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Self

from ..materials import Material

BELT_SPEED = Fraction("15")
BELT = "transport-belt"
UNDERGROUND_BELT = "underground-belt"
SPLITTER = "splitter"

Entities = dict[tuple[float, float], tuple[str, dict[str, int | str]]]


@dataclass
class MainBus:
    needle: tuple[int, int]  # top left corner position
    lines: list[tuple[Material, Fraction]]

    @classmethod
    def from_materials(cls, materials: list[Material]) -> Self:
        lines = [(material, 2 * BELT_SPEED) for material in materials]
        return cls((0, 0), lines)

    def split_outputs(self, lines: list[tuple[Material, Fraction]]) -> Entities:
        assert all(amount <= BELT_SPEED for _, amount in lines)
        x0, y0 = self.needle
        entities: Entities = {}
        for y, (target_material, target_amount) in enumerate(lines):
            y = y0 + 2 * y
            for x, (material, amount) in enumerate(self.lines):
                if target_material == material:
                    break
                for dx, dy in itertools.product((1.5, 2.5), (0.5, 1.5)):
                    entities[x0 + 4 * x + dx, y + dy] = (BELT, {"direction": 0})
            entities[x0 + 4 * x + 1.5, y + 0.5] = (BELT, {"direction": 0})
            splitter = (SPLITTER, {"direction": 0})
            if target_amount > BELT_SPEED / 2:
                splitter[1]["output_priority"] = "right"
            entities[x0 + 4 * x + 3.0, y + 0.5] = splitter
            entities[x0+4*x+2.0, y + 1.5] = (SPLITTER, {"direction": 0, "output_priority": "right"})
            entities[x0+4*x+3.5, y + 1.5] = (BELT, {"direction": 2})
            assert amount >= target_amount
            self.lines[x] = (material, amount - target_amount)
            for x, (material, amount) in enumerate(self.lines[x + 1:], x + 1):
                for dx, dy in itertools.product((1.5, 2.5), (0.5, 1.5)):
                    entities[x0 + 4 * x + dx, y + dy] = (BELT, {"direction": 0})
                inp = (UNDERGROUND_BELT, {"direction": 2, "type": "input"})
                out = (UNDERGROUND_BELT, {"direction": 2, "type": "output"})
                entities[x0 + 4 * x + 0.5, y + 1.5] = inp
                entities[x0 + 4 * x + 3.5, y + 1.5] = out
        self.needle = (x0, y0 + len(lines) * 2)
        return entities

    def split_lines(
        self, lines: list[tuple[Material, Fraction, Literal["input", "output"]]]
    ) -> Entities:
        assert all(amount <= BELT_SPEED for _, amount, typ in lines if typ == "input")
        x0, y0 = self.needle
        entities: Entities = {}
        for y, (target_material, target_amount, direction) in enumerate(lines):
            if direction == "input" and all(m != target_material for m, _ in self.lines):
                self.lines = [(target_material, Fraction("0"))] + self.lines
                x0 -= 4
            for x, (material, amount) in enumerate(self.lines):
                if target_material == material:
                    break
                for dx, dy in itertools.product((1.5, 2.5), (0.5, 1.5)):
                    if (x0 + 4 * x + dx + 0.5, y0 + dy) not in entities:
                        entities[x0 + 4 * x + dx, y0 + dy] = (BELT, {"direction": 0})
            if direction == "output":
                entities[x0 + 4 * x + 1.5, y0 + 0.5] = (BELT, {"direction": 0})
                splitter = (SPLITTER, {"direction": 0})
                if target_amount > BELT_SPEED / 2:
                    splitter[1]["output_priority"] = "right"
                entities[x0 + 4 * x + 3.0, y0 + 0.5] = splitter
                entities[x0+4*x+3.5, y0 + 1.5] = (BELT, {"direction": 2})
                entities[x0+4*x+2.0, y0 + 1.5] = (SPLITTER, {
                    "direction": 0, "output_priority": "right"
                })
                assert amount >= target_amount
                self.lines[x] = (material, amount - target_amount)
            else:
                entities[x0 + 4 * x + 1.5, y0 + 0.5] = (BELT, {"direction": 0})
                entities[x0 + 4 * x + 2.5, y0 + 0.5] = (BELT, {"direction": 0})
                splitter = (SPLITTER, {"direction": 0, "output_priority": "right"})
                entities[x0 + 4 * x + 2.0, y0 + 1.5] = splitter
                entities[x0 + 4 * x + 3.5, y0 + 1.5] = (BELT, {"direction": 0})
                entities[x0 + 4 * x + 3.0, y0 + 2.5] = (SPLITTER, {
                    "direction": 0, "filter": target_material.get_recipe_name(),
                    "input_priority": "right", "output_priority": "left"
                })
                assert amount + target_amount <= 2 * BELT_SPEED
                self.lines[x] = (material, amount + target_amount)
            for x, (material, amount) in enumerate(self.lines[x + 1:], x + 1):
                for dx, dy in itertools.product((1.5, 2.5), (0.5, 1.5)):
                    if (x0 + 4 * x + dx + 0.5, y0 + dy) not in entities:
                        entities[x0 + 4 * x + dx, y0 + dy] = (BELT, {"direction": 0})
                if direction == "input":
                    left_belt = (UNDERGROUND_BELT, {"direction": 6, "type": "output"})
                    right_belt = (UNDERGROUND_BELT, {"direction": 6, "type": "input"})
                else:
                    left_belt = (UNDERGROUND_BELT, {"direction": 2, "type": "input"})
                    right_belt = (UNDERGROUND_BELT, {"direction": 2, "type": "output"})
                entities[x0 + 4 * x + 0.5, y0 + 1.5] = left_belt
                entities[x0 + 4 * x + 3.5, y0 + 1.5] = right_belt
            y0 += 2
        self.needle = x0, y0
        return entities
