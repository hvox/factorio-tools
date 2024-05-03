from random import choice

from ..blueprints import rotated
from ..materials import Material


def design_factory_for(material: Material):
    occupied = {(0.5, 0.5)}
    material.get_recipe_name()
    entities, queue = {}, [((0.5, 0.5), material)]
    while queue:
        (x, y), material = queue.pop(0)
        if not material.ingredients:
            continue
        entities[x, y] = ("assembling-machine-1", {"recipe": material.get_recipe_name()})
        for ingridient in material.ingredients:
            dx, dy = 4, 0
            inserter = {(x + 2, y): ("inserter", {"direction": 2})}
            for _ in range(choice(list(range(4)))):
                dx, dy, inserter = -dy, dx, rotated(x, y, inserter)
            while (x + dx, y + dy) in occupied:
                dx, dy, inserter = -dy, dx, rotated(x, y, inserter)
            print(occupied, (x + dx, y + dy))
            occupied.add((x + dx, y + dy))
            pos, inserter = next(iter(inserter.items()))
            assert pos not in entities
            entities[pos] = inserter
            queue.append(((x + dx, y + dy), ingridient))
    return entities
