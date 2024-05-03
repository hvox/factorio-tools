from fractions import Fraction
from math import ceil
from typing import Any

from ...entities import Entities
from ...libs.dict_arithmetic import dict_max
from ...recipes import MATERIALS, RECIPES
from ..assemly_line_generator import double_assembly_line, max_throughput
from ..compact_main_bus_splitter import tight_main_bus
from ..types import Forks

BELT_SPEED = 15


def get_total_products(*alternative_product_sets: dict[str, Fraction]):
    totals = []
    total: Any
    for total in map(dict, alternative_product_sets):  # type: ignore
        for material in reversed(MATERIALS):
            if material not in total:
                continue
            if material not in RECIPES or RECIPES[material].facility != "assembler":
                continue
            for ingridient, amount in RECIPES[material].ingredients.items():
                total[ingridient] = total.get(ingridient, Fraction(0)) + amount * total[material]
        totals.append(total)
    total = dict_max(*totals)
    return dict(sorted(total.items(), key=lambda p: (MATERIALS[p[0]], p[0])))


def design(*products: dict[str, Fraction]) -> Entities:
    materials = get_total_products(*products)
    main_bus_lines: list[str] = list(reversed(sum((
        [mat] * ceil(x / BELT_SPEED) for (mat, x) in materials.items()
    ), [])))
    x0, y0 = len(main_bus_lines) + 1, 0
    entites: Entities = {}
    forks: Forks = []
    for x, material in enumerate(main_bus_lines):
        y = -len(main_bus_lines)
        chest = {"filters": [{"name": material, "count": 4, "mode": "exactly", "index": 1}]}
        entites[x + 0.5, y - 1] = ("express-loader", {"direction": 0, "type": "output"})
        entites[x + 0.5, y - 2.5] = ("infinity-chest", {"direction": 0, "infinity_settings": chest})
    for product, amount in materials.items():
        if product not in RECIPES or RECIPES[product].facility != "assembler":
            continue
        number_of_splits = ceil(amount / max_throughput(product))
        assembly_line = double_assembly_line(product, amount / number_of_splits)
        forks.extend(assembly_line[0] * number_of_splits)
        for _ in range(number_of_splits):
            for (x, y), entity in assembly_line[1].items():
                entites[x0 + x, y0 + y] = entity
            y0 += len(assembly_line[0])
    for pos, entity in tight_main_bus(main_bus_lines, forks).items():
        entites[pos] = entity
    return entites
