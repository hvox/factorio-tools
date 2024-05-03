from contextlib import suppress
from fractions import Fraction
from itertools import count
from math import ceil

from ...entities import Entities
from ...libs.dict_arithmetic import dict_max
from ...recipes import MATERIALS, RECIPES
from ..assemly_line_generator import assembly_line
from ..compact_main_bus_splitter import main_bus

BELT_SPEED = 15


def get_total_products(*alternative_product_sets: dict[str, Fraction]):
    totals = []
    for total in map(dict, alternative_product_sets):
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
    main_bus_materials = get_total_products(*products)
    assert all(amount <= 4 * BELT_SPEED for amount in main_bus_materials.values())
    main_bus_lines = [
        (mat, ceil(x / BELT_SPEED)) for mat, x in reversed(main_bus_materials.items())
    ]
    splits: list[list[str | None]] = []
    entites: Entities = {}
    x0 = (len(main_bus_lines) - 1) * 2 + sum(width for _, width in main_bus_lines) + 1
    y0 = -1
    for product, amount in main_bus_materials.items():
        if product not in RECIPES or RECIPES[product].facility != "assembler":
            continue
        for number_of_splits in count(1):
            with suppress(ValueError):
                split, line = assembly_line(product, amount / number_of_splits)
                break
        for _ in range(number_of_splits):
            for (x, y), entity in line.items():
                entites[x0 + x, y0 + y] = entity
            y0 += len(split) + 1
        splits.extend([split] * number_of_splits)
    for pos, entity in main_bus(main_bus_lines, splits).items():
        entites[pos] = entity
    return entites
