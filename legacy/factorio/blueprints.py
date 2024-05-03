import base64
import json
import math
import zlib
from dataclasses import dataclass
from typing import Self

from . import binary_encoding

DIRECTED_STRUCTURES = {"transport-belt", "splitter", "express-loader", "underground-belt"}


def as_int(x: float) -> int:
    if x.is_integer():
        return int(x)
    raise ValueError(f"{x} is not really an integer")


Entities = dict[tuple[float, float], tuple[str, dict[str, int | str]]]


@dataclass
class Blueprint:
    size: tuple[int, int]
    entities: dict[tuple[float, float], tuple[str, dict[str, str | int]]]

    def __init__(self, size, entities):
        self.size = size
        self.entities = entities

    @classmethod
    def from_entities(cls, entities: Entities) -> Self:
        xs, ys = [x for x, _ in entities], [y for y, _ in entities]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        min_x, min_y = (math.floor(x - 0.5) for x in (min_x, min_y))
        max_x, max_y = (math.ceil(x + 0.5) for x in (max_x, max_y))
        self = cls((max_x - min_x, max_y - min_y), {})
        for (x, y), entity in entities.items():
            self.entities[x - min_x, y - min_y] = entity
        return self

    def __str__(self):
        width, height = self.size
        result = [f"Blueprint {width}x{height}"]
        for (x, y), (name, attrs) in self.entities.items():
            info = " ".join(f"{k}:{v}" for k, v in attrs.items())
            result.append(f"{x:6} {y:6} {name:20} {info}")
        return "\n".join(result)

    @classmethod
    def decode(cls, base64_encoded: str) -> Self:
        if base64_encoded[0] != "0":
            raise ValueError("Argument is not a valid blueprint")
        zlib_compressed = base64.b64decode(base64_encoded[1:])
        json_string = zlib.decompress(zlib_compressed).decode()
        json_obj = json.loads(json_string)["blueprint"]
        size = (1, 1)
        if json_size := json_obj.get("snap-to-grid"):
            size = (json_size["x"], json_size["y"])
        entities = {}
        for attrs in json_obj["entities"]:
            name, _ = attrs.pop("name"), attrs.pop("entity_number")
            position = attrs.pop("position")
            x, y = float(position["x"]), float(size[1]-position["y"])
            attrs["direction"] = attrs.get("direction", 0)
            if "neighbours" in attrs:
                del attrs["neighbours"]
            entities[x, y] = (name, dict(sorted(attrs.items())))
        entities = dict(sorted(entities.items(), key=lambda x: (x[0][1], x[0][0])))
        return cls(size, entities)

    def encode(self) -> str:
        (width, height), entities = self.size, []
        for i, ((x, y), (name, attrs)) in enumerate(self.entities.items()):
            attrs = attrs | {"name": name, "entity_number": i + 1}
            if attrs["direction"] == 0:
                del attrs["direction"]
            entities.append(attrs | {"position": {"x": x, "y": height - y}})
        json_obj = {"entities": entities}
        if self.size != (1, 1):
            json_obj["snap-to-grid"] = {"x": width, "y": height}
        json_str = json.dumps({"blueprint": json_obj}, separators=(",", ":"))
        compressed = zlib.compress(json_str.encode(), 9)
        return "0" + base64.b64encode(compressed).decode()

    def __int__(self) -> int:
        binary = binary_encoding.encode_tile(*self.size, self.entities)
        return int("1" + binary, 2)

    @classmethod
    def from_int(cls, index: int) -> Self:
        return cls(*binary_encoding.decode_tile(bin(index)[3:]))

    def to_tile_code(self) -> str:
        number = int(self)
        bts = number.to_bytes((number.bit_length() + 7) // 8, "little")
        return base64.b85encode(bts).decode()

    @classmethod
    def from_tile_code(cls, code: str) -> Self:
        bts = base64.b85decode(code)
        number = int.from_bytes(bts, "little")
        return cls.from_int(number)

    def rotated(self, quarters: int) -> Self:
        origin_x, origin_y = (x / 2 for x in self.size)
        entities = {}
        for (x, y), (name, attrs) in self.entities.items():
            dx, dy, attrs = x - origin_x, y - origin_y, dict(attrs)
            attrs["direction"] = (attrs["direction"] - 2 * quarters) % 8
            dx, dy = [(dx, dy), (-dy, dx), (-dx, -dy), (dy, -dx)][quarters % 4]
            entities[origin_x + dx, origin_y + dy] = (name, attrs)
        return Blueprint(self.size, entities)


class BlueprintGrid:
    tile_size: tuple[int, int]
    tiles: dict[tuple[int, int], Blueprint]

    def __init__(self, tile_size, tiles):
        self.tile_size = tile_size
        self.tiles = tiles

    @classmethod
    def new(tile_size: tuple[int, int]) -> Self:
        return BlueprintGrid(tile_size, {})

    def __setitem__(self, position: tuple[int, int], blueprint: Blueprint):
        assert self.tile_size == blueprint.size
        self.tiles[position] = blueprint

    def __getitem__(self, position: tuple[int, int]) -> Blueprint:
        return self.tiles[position]

    def to_blueprint(self) -> Blueprint:
        entities = {}
        tile_width, tile_height = self.tile_size
        for (tile_x, tile_y), tile in self.tiles.items():
            x, y = tile_x * tile_width, tile_y * tile_height
            for (dx, dy), entity in tile.entities.items():
                entities[x + dx, y + dy] = entity
        return Blueprint(self.tile_size, entities)


def decode(encoded: str) -> dict[tuple[float, float], tuple[str, dict[str, str]]]:
    __import__("warnings").warn("factorio.blueprints.decode is depricated")
    assert encoded[0] == "0"
    compressed = base64.b64decode(encoded[1:])
    decoded = json.loads(zlib.decompress(compressed).decode())
    entities = {}
    for entity in decoded["blueprint"]["entities"]:
        name = entity["name"]
        x = entity["position"]["x"]
        y = -entity["position"]["y"]
        attrs = {k: v for k, v in entity.items() if k not in ("name", "position", "entity_number")}
        if name in DIRECTED_STRUCTURES and "direction" not in attrs:
            attrs["direction"] = 0
        entities[x, y] = (name, attrs)
    x0 = math.floor(min(x for x, _ in entities))
    y0 = math.floor(min(y for _, y in entities))
    return {(x - x0, y - y0): item for (x, y), item in entities.items()}


def encode(entities: dict[tuple[float, float], tuple[str, dict[str, str]]]) -> str:
    __import__("warnings").warn("factorio.blueprints.encode is depricated")
    entities_json = []
    for i, ((x, y), (name, attrs)) in enumerate(entities.items()):
        entity_json = dict(attrs)
        entity_json["entity_number"] = i + 1
        entity_json["position"] = {"x": x, "y": -y}
        entity_json["name"] = name
        if entity_json.get("direction") == 0:
            del entity_json["direction"]
        entities_json.append(entity_json)
    json_obj = {"blueprint": {"entities": entities_json}}
    compressed = zlib.compress(json.dumps(json_obj, separators=(',', ':')).encode(), 9)
    return "0" + base64.b64encode(compressed).decode()


def rotated(
    origin_x: float, origin_y: float,
    entities: dict[tuple[float, float], tuple[str, dict[str, str]]],
) -> dict[tuple[float, float], tuple[str, dict[str, str]]]:
    __import__("warnings").warn("factorio.blueprints.rotated is depricated")
    rotated_entities = {}
    for (x, y), (name, entity) in entities.items():
        dx, dy = x - origin_x, y - origin_y
        rotated_entity = dict(entity)
        if "direction" in rotated_entity:
            rotated_entity["direction"] = (rotated_entity["direction"] - 2) % 8
        rotated_entities[origin_x - dy, origin_y + dx] = (name, rotated_entity)
    return rotated_entities
