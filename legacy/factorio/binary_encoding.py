import contextlib
import sys

with contextlib.suppress(AttributeError):
    sys.set_int_max_str_digits(2**31 - 1)
ENTITY = tuple[str, dict[str, str | int]]
ENTITIES = dict[tuple[float, float], ENTITY]
ENCODING = {
    (): '0',
    ('CUSTOM',): '10110000',
    ('assembling-machine-2', 0): '10100',
    ('fast-inserter', 0): '101101',
    ('fast-inserter', 2): '10001',
    ('fast-inserter', 6): '10111011',
    ('inserter', 2): '10111010',
    ('inserter', 6): '101111',
    ('long-handed-inserter', 0): '1011100',
    ('small-electric-pole', 0): '10011',
    ('splitter', 6): '10110001',
    ('transport-belt', 0): '1011001',
    ('transport-belt', 2): '111',
    ('transport-belt', 4): '10101',
    ('transport-belt', 6): '110',
    ('underground-belt', 2, 'type', 'input'): '10000',
    ('underground-belt', 2, 'type', 'output'): '10010',
}


def str_to_bin(string: str) -> str:
    result = [bin(len(string))[2:].zfill(8)]
    for char in string:
        result.append(bin(ord(char))[2:].zfill(8))
    return "".join(result)


def bin_to_str(binary: str, i: int = 0) -> tuple[str, int]:
    length = int(binary[i: i + 8], 2)
    result = []
    for _ in range(length):
        i += 8
        result.append(chr(int(binary[i: i + 8], 2)))
    return "".join(result), i + 8


def encode_entity(entity: None | ENTITY) -> str:
    if entity is None:
        return ENCODING[()]
    attributes = entity[1].copy()
    direction = attributes.pop("direction")
    key = (entity[0], direction) + sum(sorted(attributes.items()), ())
    if key in ENCODING:
        return ENCODING[key]
    result = str_to_bin(entity[0]) + bin(direction)[2:].zfill(3)
    result += bin(len(key) // 2 - 1)[2:].zfill(8)
    for i in range(2, len(key), 2):
        result += str_to_bin(key[i])
        result += str_to_bin(key[i + 1])
    return ENCODING[("CUSTOM",)] + result


def decode_entity(binary: str, i: int = 0) -> tuple[ENTITY | None, int]:
    for key, code in ENCODING.items():
        if binary[i: i + len(code)] == code:
            break
    if not key:
        return None, i + len(code)
    if key != ("CUSTOM",):
        attributes = {"direction": key[1]}
        for j in range(2, len(key), 2):
            attributes[key[j]] = key[j + 1]
        return (key[0], attributes), i + len(code)
    name, i = bin_to_str(binary, i + len(code))
    direction = int(binary[i: i + 3], 2)
    attrs_count = int(binary[i + 3: i + 11], 2)
    i, attributes = i + 11, {"direction": direction}
    for _ in range(attrs_count):
        attr_name, i = bin_to_str(binary, i)
        attr_value, i = bin_to_str(binary, i)
        attributes[attr_name] = attr_value
    return (name, attributes), i


def encode_tile(width: int, height: int, entities: ENTITIES) -> str:
    assert width < 64 and height < 64
    grid = [None] * width * height
    for pos, entity in entities.items():
        x, y = (int(x) if x.is_integer() else int(x - 0.5) for x in pos)
        assert 0 <= x < width and 0 <= y < height
        grid[y * width + x] = entity
    result = [bin(width)[2:].zfill(6), bin(height)[2:].zfill(6)]
    for entity in grid:
        result.append(encode_entity(entity))
    return "".join(result)


def decode_tile(code: str) -> tuple[tuple[int, int], ENTITIES]:
    width, height = int(code[:6], 2), int(code[6:12], 2)
    entities, i = {}, 12
    for pos in range(width * height):
        int_y, int_x = divmod(pos, width)
        entity, i = decode_entity(code, i)
        if entity is None:
            continue
        name, attrs = entity
        if name == "splitter":
            if attrs["direction"] % 4 == 2:
                # splitter points to right or left
                x, y = int_x + 0.5, float(int_y)
            else:
                # splitter points up or down
                x, y = float(int_x), int_y + 0.5
        else:
            x, y = int_x + 0.5, int_y + 0.5
        entities[x, y] = entity
    return (width, height), entities
