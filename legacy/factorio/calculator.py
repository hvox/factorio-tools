from fractions import Fraction as Rat

from .libs.dict_arithmetic import dict_max
from .recipes import INTERMEDIATE, MATERIALS, RECIPES

MODULES_PER_FACILITY = {"assembler": 4, "chemical plant": 3, "furnace": 2, "rocket silo": 4}


def dict_diff(dict1: dict, dict2: dict) -> dict:
    result = dict(dict1)
    for key in dict2:
        if key in dict1:
            del result[key]
    return result


def dict_and(dict1: dict, dict2: dict) -> dict:
    result = {}
    for key, value in dict1.items():
        if key in dict2:
            result[key] = value
    return result


def get_total_for_one_product_set(
    target_materials: dict[str, Rat],
    ignored_facilities: set[str] = set(), use_beacons: bool = False
) -> dict[str, Rat]:
    total = dict(target_materials)
    for material in reversed(MATERIALS):
        if material not in total or material not in RECIPES:
            continue
        if RECIPES[material].facility in ignored_facilities:
            continue
        product_amount = total[material]
        if use_beacons and material in INTERMEDIATE:
            product_amount /= 1 + 0.1 * MODULES_PER_FACILITY[RECIPES[material].facility]
        for ingridient, amount in RECIPES[material].ingredients.items():
            total[ingridient] = total.get(ingridient, Rat(0)) + amount * product_amount
    return total


def get_total(
    *alternative_product_sets: dict[str, Rat],
    ignored_facilities: set[str] = set(), use_beacons: bool = False
) -> dict[str, Rat]:
    total = dict_max(*(
        get_total_for_one_product_set(prod, ignored_facilities, use_beacons)
        for prod in alternative_product_sets
    ))
    return dict(sorted(total.items(), key=lambda p: (MATERIALS[p[0]], p[0])))


def get_total_assembly_made(*alternative_product_sets: dict[str, Rat]) -> dict[str, Rat]:
    totals = []
    for total in map(dict, alternative_product_sets):
        for material in reversed(MATERIALS):
            if material not in total:
                continue
            if material not in RECIPES or RECIPES[material].facility != "assembler":
                continue
            for ingridient, amount in RECIPES[material].ingredients.items():
                total[ingridient] = total.get(ingridient, Rat(0)) + amount * total[material]
        totals.append(total)
    total = dict_max(*totals)
    return dict(sorted(total.items(), key=lambda p: (MATERIALS[p[0]], p[0])))


def rationalize_values(dct: dict) -> dict[str, Rat]:
    return dict((((k, Rat(v).limit_denominator(100)) for k, v in dct.items())))
