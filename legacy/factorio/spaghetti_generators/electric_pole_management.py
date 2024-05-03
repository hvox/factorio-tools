from math import trunc
from typing import Any

from ..entities import ELECTRIC_POLES, Entities


def get_wire_length(pole: str) -> float:
    return [7.5, 9.0, 30.0, 18.0][ELECTRIC_POLES.index(pole)]


def push(lst: Any, element):
    if element not in lst:
        lst.append(element)


def merge_electric_poles_into_grid(entities: Entities) -> None:
    indexes = {pos: i + 1 for i, pos in enumerate(entities)}
    for (x, y), (pole1, attrs1) in entities.items():
        if pole1 not in ELECTRIC_POLES:
            continue
        wire1 = get_wire_length(pole1)
        for direction in [(0, 1), (1, 0)]:
            for distance in range(1, trunc(wire1) + 1):
                pos2: Any = tuple(x + dir * distance for x, dir in zip((x, y), direction))
                if pos2 not in entities or entities[pos2][0] not in ELECTRIC_POLES:
                    continue
                pole2, attrs2 = entities[pos2]
                if get_wire_length(pole2) < distance:
                    continue
                push(attrs1.setdefault("neighbours", []), indexes[pos2])
                push(attrs2.setdefault("neighbours", []), indexes[x, y])
                break
