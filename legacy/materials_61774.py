from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from functools import cached_property
from graphlib import TopologicalSorter
from itertools import chain
from pathlib import Path
from typing import Iterable

from frozendict import frozendict

Real = int | float | Fraction


class Material:
    name: str
    time: Fraction
    ingredients: frozendict[str, Fraction]

    def __init__(self, name: str, time: Real = 0, ingredients: Iterable = {}):
        self.name = name
        self.time = to_fraction(time)
        self.ingredients = frozendict({k: to_fraction(v) for k, v in dict(ingredients).items()})

    def __repr__(self):
        return self.name

    # def __str__(self):
    #     ingredients = ", ".join(f"{k}:{v}" for k, v in self.ingredients.items())
    #     return self.name + "(" + ingredients + ")"

    def to_tsv_row(self) -> str:
        ingredients = ", ".join(f"{k}:{v}" for k, v in self.ingredients.items())
        return self.name + "\t" + str(self.time) + "\t" + ingredients

    @staticmethod
    def from_tsv_row(row: str) -> Material:
        name, time, ingredients_str = row.split("\t")
        ingredients = {k: Fraction(v) for k, v in (s.split(":") for s in ingredients_str.split(", "))}
        return Material(name, Fraction(time), ingredients)

    @property
    def is_liquid(self) -> bool:
        liquids = "water petroleum-gas light-oil heavy-oil sulfuric-acid lubricant".split()
        return self.name in liquids

    @property
    def items_per_lane(self) -> int:
        return 60 * (1000 if self.is_liquid else 15)

    @property
    def buildings(self) -> Fraction:
        time = {
            "space-science-pack": self.time,
            "iron-plate": self.time,
            "coper-plate": self.time,
            "steel-plate": self.time,
            "stone-brick": self.time,
            "battery": self.time,
            "solid-fuel": self.time,
        }
        return time.get(self.name) or (self.time if self.is_liquid else self.time / Fraction(0.75))

    @cached_property
    def color(self) -> object:
        black = 37
        red = 91
        green = 92
        yellow = 93
        blue = 94
        magenta = 95
        cyan = 96
        white = "97;1"
        name = self.name.split("-")[0]
        return {
            "accumulator": blue,
            "advanced": red,
            "automation": red,
            "battery": blue,
            "chemical": cyan,
            "coal": black,
            "copper": red,
            "electric": green,
            "electronic": green,
            "engine": yellow,
            "heavy": yellow,
            "inserter": yellow,
            "iron": cyan,
            "light": red,
            "logistic": green,
            "lubricant": green,
            "military": black,
            "petroleum": black,
            "pipe": cyan,
            "plastic": white,
            "processing": blue,
            "production": magenta,
            "radar": black,
            "rocket": magenta,
            "satellite": magenta,
            "solar": blue,
            "solid": black,
            "space": white,
            "speed": blue,
            "steel": cyan,
            "stone": yellow,
            "sulfur": yellow,
            "transport": yellow,
            "utility": yellow,
            "water": blue,
        }.get(name, 0)

    @staticmethod
    def load_materials() -> Mapping[str, Material]:
        path = Path(__file__).resolve().parent / "materials.tsv"
        recipes = map(Material.from_tsv_row, path.read_text().splitlines()[1:])
        materials = {recipe.name: recipe for recipe in recipes}
        for ingr in chain(*(mat.ingredients for mat in materials.values())):
            materials.setdefault(ingr, Material(ingr, Fraction(0), {}))
        graph = {name: mat.ingredients for name, mat in materials.items()}
        materials = {mat: materials[mat] for mat in TopologicalSorter(graph).static_order()}
        result = frozendict(materials)
        setattr(Material, "materials", result)
        return result

    @staticmethod
    def load_the_boys() -> tuple[Material, ...]:
        materials = getattr(Material, "materials") or Material.load_materials()
        coal = materials["coal"]
        iron = materials["iron-plate"]
        copper = materials["copper-plate"]
        plastic = materials["plastic-bar"]
        green = materials["electronic-circuit"]
        red = materials["advanced-circuit"]
        blue = materials["processing-unit"]
        return coal, iron, copper, plastic, green, red, blue

    def total_cost(self, items_per_minute: Real, ignored=None) -> dict[Material, Fraction]:
        self_amount = to_fraction(items_per_minute)
        materials = getattr(Material, "materials")
        total = {material: Fraction(0) for material in materials}
        total[self.name] = self_amount
        not_ignored = (lambda x: not ignored(materials[x[0]])) if ignored else lambda x: True
        for name, ipm in filter(not_ignored, reversed(total.items())):
            for ingr, ingr_ipm in materials[name].ingredients.items():
                total[ingr] += ipm * ingr_ipm
        return {materials[name]: ipm for name, ipm in total.items() if ipm}

    def __bool__(self):
        return bool(self.ingredients)

    def __iter__(self):
        materials = getattr(Material, "materials")
        for ingr, amount in self.ingredients.items():
            yield materials[ingr], amount


def to_fraction(x: int | float | Fraction | Real) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10**12)
