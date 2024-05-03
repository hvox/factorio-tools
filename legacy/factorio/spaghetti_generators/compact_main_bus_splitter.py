import itertools
from dataclasses import dataclass
from fractions import Fraction
from typing import Self

from ..entities import (Entities, EntityAttributes, assembler, belt, splitter,
                        underground)
from ..recipes import MATERIALS
from .types import Forks


def get_line_left(lines: list[tuple[str, int]], material: str) -> tuple[int, int]:
    x = 0
    for i, (line, width) in enumerate(lines):
        if line == material:
            return i, x
        x += width + 2
    raise ValueError(f"{material} not found in the main bus")


def main_bus(lines: list[tuple[str, int]], splits: list[list[str | None]]) -> Entities:
    for (prev_line, _), (next_line, _) in zip(lines, lines[1:]):
        if MATERIALS[prev_line] < MATERIALS[next_line]:
            raise ValueError(f"Materials {prev_line} and {next_line} are in wrong order")
    entities: Entities = {}
    y0 = 0
    # x1 = (len(lines) - 1) * 2 + sum(width for _, width in lines)
    y1 = sum(len(split) + 1 for split in splits) - 1
    for y in range(y0 - 4, y1 + 4):
        x = 0
        for material, width in lines:
            for dx in range(width):
                entities[x + dx + 0.5, y + 0.5] = belt(1, 0)
            x += width + 2
    for split in splits:
        if len(split) == 0:
            y0 += 1
            continue
        *inputs, output = split
        for input in inputs:
            if input is None:
                y0 += 1
                continue
            index, x = get_line_left(lines, input)
            _, width = lines[index]
            if (x + width - 0.5, y0 - 0.5) in entities:
                del entities[x + width - 0.5, y0 - 0.5]
                entities[x + width, y0 - 0.5] = splitter(1, 1, 0, 1)
            else:
                entities[x + width, y0 - 0.5] = splitter(1, 1, 1, 1)
            for dx in range(1, width):
                del entities[x + width - dx - 0.5, y0 + dx - 0.5]
                del entities[x + width - dx + 0.5, y0 + dx - 0.5]
                entities[x + width - dx, y0 + dx - 0.5] = splitter(1, 1, 0, 1)
            x += width + 1
            entities[x - 0.5, y0 + 0.5] = belt(1, 2)
            for _, width in lines[index+1:]:
                left, right = underground(1, 2)
                entities[x + 0.5, y0 + 0.5] = left
                x += width + 2
                entities[x - 0.5, y0 + 0.5] = right
            y0 += 1
        if output is None:
            y0 += 2
            continue
        index, x = get_line_left(lines, output)
        _, width = lines[index]
        del entities[x + width - 0.5, y0 + 1.5]
        entities[x + width, y0 + 1.5] = splitter(1, 0, 1, -1, output)
        # TODO: check if top-going splitters work here as good as at inputs
        for dx in range(1, width):
            del entities[x + width - dx - 0.5, y0 - dx + 1.5]
            del entities[x + width - dx + 0.5, y0 - dx + 1.5]
            entities[x + width - dx, y0 - dx + 1.5] = splitter(1, 0, 1, 1)
        x += width + 1
        entities[x - 0.5, y0 + 0.5] = belt(1, 0)
        for _, width in lines[index+1:]:
            down, up = underground(1, -2)
            entities[x + 0.5, y0 + 0.5] = up
            x += width + 2
            entities[x - 0.5, y0 + 0.5] = down
        y0 += 2
    return entities


def tight_main_bus(lines: list[str], forks: Forks) -> Entities:
    lines = list(lines)
    entities: Entities = {}
    for y in range(-len(lines), len(forks)):
        for x in range(len(lines)):
            entities[x + 0.5, y + 0.5] = belt(1, 0)
    for y, current_row_fork in enumerate(forks):
        if current_row_fork is None:
            continue
        fork_material, typ = current_row_fork
        i = lines.index(fork_material)
        entities[len(lines) + 0.5, y + 0.5] = belt(1, 2 if typ == "out" else 0)
        if typ == "out":
            del entities[len(lines) - 0.5, y - 0.5]
            entities[len(lines), y - 0.5] = splitter(1, 0, 0, 1)
        else:
            del entities[len(lines) - 0.5, y + 1.5]
            entities[len(lines), y + 1.5] = splitter(1, 0, 1, -1, fork_material)
        for dy, j in enumerate(reversed(range(i + 1, len(lines))), 1):
            if typ == "in":
                dy -= 2
            del entities[j - 0.5, y - dy - 0.5]
            del entities[j + 0.5, y - dy - 0.5]
            entities[j, y - dy - 0.5] = splitter(1, 0, 0, 1, fork_material)
            if fork_material == lines[j]:
                entities[j, y - dy - 0.5] = splitter(1, 0, 1, 1)
        lines.append(lines.pop(i))
    return entities
