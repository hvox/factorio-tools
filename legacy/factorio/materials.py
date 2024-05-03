from fractions import Fraction as Rat
from typing import NamedTuple, Self


class Material(NamedTuple):
    lvl: int
    name: str
    creation_time: Rat
    ingredients: dict[Self, Rat]

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def get_recipe_name(self):
        return self.name.lower().replace(" ", "-")

    def as_tree(self) -> str:
        if not self.ingredients:
            return self.name
        children = [child.as_tree() for child in self.ingredients]
        for i, child in enumerate(children[:-1]):
            children[i] = "├── " + ("\n│   ").join(child.split("\n"))
        children[-1] = "└── " + ("\n    ").join(children[-1].split("\n"))
        return f"{self.name}\n" + "\n".join(children)


# I just want deterministic set here. It's a shame, that I have to use dicts!
# TODO: find something on pypi for this specific purpose.
MINED_RESOURCES: dict[Material, int] = {}
MATERIALS: dict[Material, int] = {}


def new(name, creation_time: float = 0, ingredients=None):
    if ingredients is None:
        ingredients = {}
    recipe = dict(reversed(sorted((m, Rat(x).limit_denominator(10)) for m, x in ingredients.items())))
    complexity_level = max((mat.lvl for mat in recipe), default=0) + 1
    time = Rat(creation_time).limit_denominator(10)
    material = Material(complexity_level, name, time, recipe)
    if not material.creation_time:
        MINED_RESOURCES[material] = len(MINED_RESOURCES)
    MATERIALS[material] = len(MATERIALS)
    return material


IRON_PLATE = new("Iron plate")
COPPER_PLATE = new("Copper plate")
STEEL_PLATE = new("Steel plate")

IRON_GEAR_WHEEL = new("Iron gear wheel", 0.5, {IRON_PLATE: 2})
COPPER_CABLE = new("Copper cable", 0.25, {COPPER_PLATE: 0.5})
ELECTRONIC_CIRCUIT = new("Electronic circuit", 0.5, {COPPER_CABLE: 3, IRON_PLATE: 1})
PIPE = new("Pipe", 0.5, {IRON_PLATE: 1})
TRANSPORT_BELT = new("Transport belt", 0.25, {IRON_PLATE: 0.5, IRON_GEAR_WHEEL: 0.5})
SPLITTER = new("Splitter", 1, {ELECTRONIC_CIRCUIT: 5, IRON_PLATE: 5, TRANSPORT_BELT: 4})
UNDERGROUND_BELT = new("Underground belt", 0.5, {IRON_PLATE: 5, TRANSPORT_BELT: 2.5})
INSERTER = new("Inserter", 0.5, {ELECTRONIC_CIRCUIT: 1, IRON_GEAR_WHEEL: 1, IRON_PLATE: 1})
ENGINE_UNIT = new("Engine unit", 10, {IRON_GEAR_WHEEL: 1, PIPE: 2, STEEL_PLATE: 1})
CAR = new("Car", 2, {ENGINE_UNIT: 8, IRON_PLATE: 20, STEEL_PLATE: 5})

AUTOMATION_SCIENCE_PACK = new("Automation science pack", 5, {COPPER_PLATE: 1, IRON_GEAR_WHEEL: 1})
LOGISTIC_SCIENCE_PACK = new("Logistic science pack", 6, {INSERTER: 1, TRANSPORT_BELT: 1})
