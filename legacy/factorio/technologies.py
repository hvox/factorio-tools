from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

SCIENCE_PACK_ALIASES = {
    "A": "science-pack-1",
    "L": "science-pack-2",
    "C": "science-pack-3",
    "M": "military-science-pack",
    "P": "production-science-pack",
    "U": "utility-science-pack",
    "S": "space-science-pack",
}


class Technology(NamedTuple):
    cost: int
    creation_time: Fraction
    science_packs: list[str]
    dependencies: list[str]


def read_technologies(table_name: str) -> dict[str, Technology]:
    technologies = {}
    path = Path(__file__).with_suffix("").parent / ("data/" + table_name + ".tsv")
    for row in path.read_text().strip("\n").split("\n")[1:]:
        tech, time, science_packs_and_amount, deps = row.split("\t")
        science_packs_str, amount = science_packs_and_amount.split(":")
        science_packs = [SCIENCE_PACK_ALIASES[symbol] for symbol in science_packs_str]
        creation_time = Fraction(time) * Fraction(amount)
        dependencies = list(deps.split(", ")) if deps else []
        technologies[tech] = Technology(int(amount), creation_time, science_packs, dependencies)
    return technologies


TECHNOLOGIES = read_technologies("technologies")
