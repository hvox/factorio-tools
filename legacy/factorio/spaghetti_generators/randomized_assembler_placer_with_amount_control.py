from dataclasses import dataclass
from fractions import Fraction
from queue import PriorityQueue
from random import shuffle

from ..blueprints import Blueprint
from ..materials import Material

ASSEMBLER = "assembling-machine-1"
ASSEMBLER_SPEED = Fraction("0.5")

BELT = "transport-belt"
BELt_SPEED = Fraction(15)

INSERTER = "inserter"


class PositionQueue(PriorityQueue):
    def __init__(self, *args) -> None:
        super().__init__()
        assert len(args) % 2 == 0
        for x, y in (args[i:i+2] for i in range(0, len(args), 2)):
            self.push(x, y)

    def push(self, x: int, y: int) -> None:
        min_dist, max_dist = sorted(list(map(abs, (x, y))))
        super().put((max_dist, min_dist, x, y))

    def pop(self) -> tuple[int, int]:
        return super().get()[2:4]


@dataclass
class GridCell:
    assembler: int
    material: Material
    position: tuple[int, int]
    sources: list[tuple[int, int]]

    def add_source(self, x: int, y: int):
        self.sources.append((x, y))


@dataclass
class Grid:
    cells: dict[tuple[int, int], GridCell | tuple[Material, Fraction]]

    def cell(self, x: int, y: int, material: Material, assembler: int) -> GridCell:
        cell = GridCell(assembler, material, (x, y), [])
        self.cells[x, y] = cell
        return cell

    def want(self, x: int, y: int, material: Material, amount: Fraction) -> tuple[int, int]:
        self.cells[x, y] = (material, amount)
        return x, y

    def __getitem__(self, position: tuple[int, int]) -> tuple[int, int, Material, Fraction]:
        x, y = position
        assert isinstance(self.cells[x, y], tuple)
        material, amount = self.cells[x, y]
        return x, y, material, amount

    def get_neighbors(self, x: int, y: int):
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            x2, y2 = x + dx, y + dy
            if (x2, y2) not in self.cells:
                yield x2, y2

    def to_blueprint(self) -> Blueprint:
        cells = [cell for cell in self.cells.values() if isinstance(cell, GridCell)]
        outputs = {}
        for cell in cells:
            for source in cell.sources:
                outputs[source] = cell.position
        entities = {}
        for cell in cells:
            x, y = cell.position
            x1, y1 = 4 * x + 1.5, 4 * y + 1.5
            if cell.assembler:
                attrs = {"direction": 0, "recipe": cell.material.get_recipe_name()}
                entities[x1, y1] = (ASSEMBLER, attrs)
                for x2, y2 in cell.sources:
                    dx, dy = x2 - x, y2 - y
                    attrs = {"direction": to_direction(dx, dy)}
                    entities[x1 + 2 * dx, y1 + 2 * dy] = (INSERTER, attrs)
                continue
            if (x, y) in outputs:
                x2, y2 = outputs[x, y]
                dx, dy = x2 - x, y2 - y
                attrs = {"direction": to_direction(dx, dy)}
                entities[x1, y1] = (BELT, dict(attrs))
                entities[x1 + dx, y1 + dy] = (BELT, dict(attrs))
                if not self.cells[outputs[x, y]].assembler:
                    entities[x1 + 2*dx, y1 + 2*dy] = (BELT, dict(attrs))
            for x2, y2 in cell.sources:
                dx, dy = x2 - x, y2 - y
                attrs = {"direction": to_direction(-dx, -dy)}
                entities[x1 + 1 * dx, y1 + 1 * dy] = (BELT, dict(attrs))
                if (x2, y2) in self.cells and self.cells[x2, y2].assembler:
                    attrs = {"direction": to_direction(dx, dy)}
                    entities[x1 + 2 * dx, y1 + 2 * dy] = (INSERTER, attrs)
                else:
                    entities[x1 + 2 * dx, y1 + 2 * dy] = (BELT, dict(attrs))
        return Blueprint((1, 1), entities)


def to_direction(dx: int, dy: int) -> int:
    if dx == 0:
        return 0 if dy > 0 else 4
    return 2 if dx > 0 else 6


def design_factory_for(material: Material, amount: Fraction) -> Grid:
    queue, grid = PositionQueue(0, 0), Grid({(0, 0): (material, amount)})
    while not queue.empty():
        x, y, material, amount = grid[queue.pop()]
        print(queue.qsize(), material.name.ljust(12), amount)
        if max(abs(x), abs(y)) > 5:
            grid.cell(x, y, material, 0)
            continue
        unoccupied = list(grid.get_neighbors(x, y))
        shuffle(unoccupied)
        if not material.ingredients or len(unoccupied) < len(material.ingredients):
            assert unoccupied
            print("\tsource:", *unoccupied[0], material, amount)
            grid.cell(x, y, material, 0).add_source(*unoccupied[0])
            queue.push(*grid.want(*unoccupied[0], material, amount))
            continue
        pps = ASSEMBLER_SPEED / material.creation_time
        if amount > pps:
            cell = grid.cell(x, y, material, 0)
            while amount / pps <= len(unoccupied) - 1:
                unoccupied.pop()
            for x, y in unoccupied:
                print("\tsource:", x, y, material, amount)
                queue.push(*grid.want(x, y, material, amount / len(unoccupied)))
                cell.add_source(x, y)
            continue
        cell = grid.cell(x, y, material, 1)
        sources = zip(unoccupied, material.ingredients.items())
        for (x, y), (source_material, source_amount) in sources:
            print("\tsource:", x, y, source_material)
            queue.push(*grid.want(x, y, source_material, amount * source_amount))
            cell.add_source(x, y)
    return grid
