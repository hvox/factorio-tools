from fractions import Fraction
from pathlib import Path
from typing import Any

EntityAttributes = dict[str, object]
Entity = tuple[str, EntityAttributes]
Entities = dict[tuple[float, float], Entity]

BELTS = ["transport-belt", "fast-transport-belt", "express-transport-belt"]
UNDERGROUND_BELTS = ["underground-belt", "fast-underground-belt", "express-underground-belt"]
SPLITTERS = ["splitter", "fast-splitter", "express-splitter"]
ASSEMBLY_MACHINES = ["assembling-machine-1", "assembling-machine-2", "assembling-machine-3"]
INSERTERS = ["inserter", "fast-inserter"]
ELECTRIC_POLES = ["small-electric-pole", "medium-electric-pole", "big-electric-pole", "substation"]


def belt(lvl: int, direction: int) -> Entity:
    assert 1 <= lvl <= 3
    name = BELTS[lvl - 1]
    assert direction % 2 == 0
    return (name, {"direction": direction % 8 // 2 * 2})


def inserter(lvl: int, direction: int) -> Entity:
    assert 1 <= lvl <= 2
    name = INSERTERS[lvl - 1]
    assert direction % 2 == 0
    return (name, {"direction": direction % 8 // 2 * 2})


def inserter_with_throughput(throughput: Fraction, direction: int) -> None | Entity:
    if throughput == 0:
        return None
    if throughput <= 2.25:
        lvl = 1
    elif throughput <= 6.00:
        lvl = 2
    else:
        raise ValueError("Throughput is unachievable by inserter")
    return inserter(lvl, direction)


def underground(lvl: int, direction: int) -> tuple[Entity, Entity]:
    assert 1 <= lvl <= 3
    name = UNDERGROUND_BELTS[lvl - 1]
    assert direction % 2 == 0
    inp = {"direction": direction % 8 // 2 * 2, "type": "input"}
    out = {"direction": direction % 8 // 2 * 2, "type": "output"}
    return ((name, inp), (name, out))


def electric_pole(lvl: int) -> Entity:
    assert 1 <= lvl <= 4
    name = ELECTRIC_POLES[lvl - 1]
    return (name, {"direction": 0})


def splitter(
    lvl: int,
    direction: int,
    input_prioriy: int = 0,
    output_priority: int = 0,
    filter_material: str = "",
) -> Entity:
    assert 1 <= lvl <= 3
    name = SPLITTERS[lvl - 1]
    attributes: EntityAttributes = {"direction": direction % 8 // 2 * 2}
    if input_prioriy:
        attributes["input_priority"] = ["", "right", "left"][input_prioriy]
    if output_priority:
        attributes["output_priority"] = ["", "right", "left"][output_priority]
    if filter_material:
        attributes["filter"] = filter_material
    return (name, attributes)


def assembler(lvl: int, produced_material: str = "", direction: int = 0) -> Entity:
    assert 1 <= lvl <= 3
    name = ASSEMBLY_MACHINES[lvl - 1]
    attributes: EntityAttributes = {"direction": direction % 8 // 2 * 2}
    if produced_material:
        attributes["recipe"] = produced_material
    return (name, attributes)


def read_entities(table_name: str) -> dict[str, tuple[int, int]]:
    entities = {}
    path = Path(__file__).with_suffix("") / "../data" / (table_name + ".tsv")
    for row in path.resolve().read_text().strip("\n").split("\n")[1:]:
        entity, width, height = row.split("\t")
        entities[entity] = (int(width), int(height))
    return entities


ENTITIES: dict[str, tuple[int, int]] = read_entities("entities")


def entity_to_python(entity: Entity) -> str:
    attrs: Any
    name, attrs = entity
    if name in BELTS:
        lvl = BELTS.index(name) + 1
        direction = (int(attrs["direction"]) + 2) % 8 - 2
        return f"belt({lvl}, {direction})"
    elif name in UNDERGROUND_BELTS:
        lvl = UNDERGROUND_BELTS.index(name) + 1
        direction = (int(attrs["direction"]) + 2) % 8 - 2
        typ = ["input", "output"].index(attrs["type"])
        return f"underground({lvl}, {direction})[{typ}]"
    elif name in INSERTERS:
        lvl = INSERTERS.index(name) + 1
        direction = (attrs["direction"] + 2) % 8 - 2
        return f"inserter({lvl}, {direction})"
    elif name in SPLITTERS:
        lvl = SPLITTERS.index(name) + 1
        direction = (attrs["direction"] + 2) % 8 - 2
        input_priority = ["left", "", "right"].index(attrs.get("input_priority", "")) - 1
        output_priority = ["left", "", "right"].index(attrs.get("output_priority", "")) - 1
        filter = attrs.get("filter", "")
        return f"splitter({lvl}, {direction}, {input_priority}, {output_priority}, {filter!r})"
    elif name in ASSEMBLY_MACHINES:
        lvl = ASSEMBLY_MACHINES.index(name) + 1
        direction = (attrs["direction"] + 2) % 8 - 2
        product = attrs.get("recipe", "")
        return f"assembler({lvl}, {product!r}, {direction})"
    elif name in ELECTRIC_POLES:
        lvl = ELECTRIC_POLES.index(name) + 1
        return f"electric_pole({lvl})"
    return repr(entity)


def entities_to_python(entities: Entities) -> list[str]:
    code = ["x, y = 0, 0  # bottom left corner", "entities: Entities = {}"]
    for (x, y), entity in sorted(entities.items()):
        code.append(f"entities[x + {x}, y + {y}] = {entity_to_python(entity)}")
    return code
