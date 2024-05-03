from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import NamedTuple


class Recipe(NamedTuple):
    facility: str
    creation_time: Fraction
    ingredients: dict[str, Fraction]

    def __str__(self) -> str:
        # TODO: implement better string representation for amounts
        ingredients = (f"{ing}:{x}" for ing, x in self.ingredients.items())
        return f"time:{self.creation_time} + " + " + ".join(ingredients)


def read_recipes(item_group: str) -> dict[str, Recipe]:
    recipes = {}
    path = Path(__file__).with_suffix("") / (item_group + ".tsv")
    for row in path.read_text().strip("\n").split("\n")[1:]:
        ingredients = {}
        material, creation_time, facility, rest = row.split("\t")
        for ingridient, amount in (ing.split(":") for ing in rest.split(", ")):
            ingredients[ingridient] = Fraction(amount)
        recipes[material] = Recipe(facility, Fraction(creation_time), ingredients)
    return recipes


LOGISTICS = read_recipes("logistics")
PRODUCTION = read_recipes("production")
INTERMEDIATE = read_recipes("intermediate_products")
COMBAT = read_recipes("combat")
RECIPES = LOGISTICS | PRODUCTION | INTERMEDIATE | COMBAT


def topologically_sorted_materials(recipes: dict[str, Recipe]) -> list[str]:
    @cache
    def get_lvl(material: str):
        if recipe := recipes.get(material):
            return 1 + max(map(get_lvl, recipe.ingredients))
        return 1

    materials: set[tuple[int, str]] = set()
    for material, recipe in recipes.items():
        materials.add((get_lvl(material), material))
        for material in recipe.ingredients:
            materials.add((get_lvl(material), material))
    return [material for _, material in sorted(materials)]


MATERIALS = {m: i for i, m in enumerate(topologically_sorted_materials(RECIPES))}
FLUIDS = {fluid: i for fluid, i in MATERIALS.items() if fluid in sorted([
    "water", "petroleum-gas", "heavy-oil", "sulfuric-acid"
])}
