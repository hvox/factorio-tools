import itertools

from .blueprints import Blueprint

# Transport belts: 4 directions.
INSERTERS = "▼◀▲▶"
# Transport belts: 4 directions.
TRANSPORT_BELTS = "↑→↓←"
# Underground belts: type(input or output) and 4 directions
UNDERGROUND_BELTS = "↟↠↡↞" + "↥↦↧↤"
# Splitters: 4 directions and priority
SPLITTERS = "⇧⇨⇩⇦⇑⇒⇓⇐"
# Assembler: just one assembler
ASSEMBLER = "+"

IGNORED_CHARS = " ││└┘┌┐─"


def tile_to_str(tile: Blueprint) -> str:
    w, h = tile.size
    assert w == h
    grid = [[" "] * w for _ in range(h)]
    for (x, y), (name, attrs) in tile.entities.items():
        if name == "inserter":
            char = INSERTERS[attrs["direction"] // 2]
            grid[int(y - 0.5)][int(x - 0.5)] = char
        elif name == "transport-belt":
            char = TRANSPORT_BELTS[attrs["direction"] // 2]
            grid[int(y - 0.5)][int(x - 0.5)] = char
        elif name == "underground-belt":
            code = attrs["direction"] // 2
            code += 4 * ["input", "output"].index(attrs["type"])
            grid[int(y - 0.5)][int(x - 0.5)] = UNDERGROUND_BELTS[code]
        elif name == "splitter":
            direction = attrs["direction"] // 2
            priority = attrs.get("output_priority")
            p1, p2 = (priority == "left", priority == "right")
            (dx1, dy1), (dx2, dy2) = [
                [(-1, -0.5), (0, -0.5)], [(-0.5, 0), (-0.5, -1)],
                [(0, -0.5), (-1, -0.5)], [(-0.5, -1), (-0.5, 0)],
            ][direction]
            grid[int(y + dy1)][int(x + dx1)] = SPLITTERS[4 * p1 + direction]
            grid[int(y + dy2)][int(x + dx2)] = SPLITTERS[4 * p2 + direction]
        elif name == "assembling-machine-2":
            grid[int(y + 0.5)][int(x - 1.5):int(x + 1.5)] = "┌─┐"
            grid[int(y - 0.5)][int(x - 1.5):int(x + 1.5)] = "│+│"
            grid[int(y - 1.5)][int(x - 1.5):int(x + 1.5)] = "└─┘"
        else:
            raise ValueError(f"What is {name}?")
    rows = [list(" ".join(row)) for row in reversed(grid)]
    for row in rows:
        for i in range(1, len(row), 2):
            if row[i - 1] == "─" or row[i + 1] == "─":
                row[i] = "─"
    return "\n".join(map("".join, rows))


def str_to_tile(grid_string: str) -> Blueprint:
    lines = list(reversed(grid_string.strip("\n").split("\n")))
    size = len(lines)
    grid = [[line[i] for i in range(-size * 2 + 1, 0, 2)] for line in lines]
    entities = {}
    for y, x in itertools.product(range(size), range(size)):
        char = grid[y][x]
        if char in IGNORED_CHARS:
            continue
        elif char in INSERTERS:
            attrs = {"direction": INSERTERS.index(char) * 2}
            entities[x + 0.5, y + 0.5] = ("inserter", attrs)
        elif char in TRANSPORT_BELTS:
            attrs = {"direction": TRANSPORT_BELTS.index(char) * 2}
            entities[x + 0.5, y + 0.5] = ("transport-belt", attrs)
        elif char in UNDERGROUND_BELTS:
            code = UNDERGROUND_BELTS.index(char)
            attrs = {"direction": code % 4 * 2}
            attrs["type"] = ["input", "output"][code // 4]
            entities[x + 0.5, y + 0.5] = ("underground-belt", attrs)
        elif char in SPLITTERS:
            assert False, "TODO: implement that part"
            # TODO: I am here. Trying to parse a pair of arrows, which are one splitter
            # I should do that based on direction and deleting the partner cell
        elif char in ASSEMBLER:
            attrs = {"direction": 0}
            entities[x + 0.5, y + 0.5] = ("assembling-machine-2", attrs)
        else:
            raise ValueError("What is {char!r}?")
    return Blueprint((size, size), entities)


DEFAULT_TILES = {
    "1-input": "*p(4bMnD+>WduCRgkqwYC?*Q%))(pv^@aK(*062cwr$(CZQGV5x&",
    "2-input": "{Qv*||NsC0|No~&3$&mGEwC#ipp1Yr0?G(@lnKQ|F;PsE3ElcaeWAWkU&I==ZQHhO+qP}nvPKs",
    "3-input": "{Qv*||NsC0|Nr;@|NsC0|NsC0|5`4zTxhw_a-rqQVpm2$83APklo9Z#32K6xpeCr9g>HSJzEEGNFMtXvsGvfuVcWKC+qP}nwk?Zv0R",
}
