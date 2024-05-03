import graphlib

sort_topologically = lambda g: graphlib.TopologicalSorter(g).static_order()
BELT_THROUGHPUT = 15
materials = {
    "transport belt",
    "copper cable",
    "iron plate",
    "underground belt",
    "splitter",
    "iron gear wheel",
    "copper plate",
    "electronic circuit",
    "artillery shell",
    "radar",
    "plastic bar",
    "explosive cannon shell",
    "steel plate",
}
recipes = {
    "iron gear wheel": {"iron plate": 2},
    "transport belt": {"iron gear wheel": 0.5, "iron plate": 0.5},
    "underground belt": {"iron plate": 5, "transport belt": 2.5},
    "copper cable": {"copper plate": 0.5},
    "electronic circuit": {"copper cable": 3, "iron plate": 1},
    "splitter": {"iron plate": 5, "transport belt": 4, "electronic circuit": 5},
    "artillery shell": {
        "radar": 1,
        "explosives": 8,
        "explosive cannon shell": 4,
    },
    "radar": {"iron plate": 10, "iron gear wheel": 5, "electronic circuit": 5},
    "explosive cannon shell": {
        "steel plate": 2,
        "plastic bar": 2,
        "explosives": 2,
    },
}
crafting_times = {
    "iron gear wheel": 0.5,
    "transport belt": 0.25,
    "underground belt": 0.5,
    "copper cable": 0.25,
    "electronic circuit": 0.5,
    "splitter": 1,
    "artillery shell": 15,
    "radar": 0.5,
    "explosive cannon shell": 8,
}


def vector_sum(*dicts):
    if len(dicts) != 2:
        if len(dicts) == 1:
            return dicts[0]
        return vector_sum(dicts[0], vector_sum(dicts[1], *dicts[2:]))
    result = dict(dicts[0])
    for key, value in dicts[1].items():
        result[key] = result[key] + value if key in result else value
    return result


def vector_dot(v: dict, x):
    return {k: v * x for k, v in v.items()}


def get_total_raw_materials(material, amount=1):
    if not (recipe := recipes.get(material, None)):
        return {material: amount}
    raw = {}
    for material, x in recipe.items():
        x *= amount
        for raw_material, y in get_total_raw_materials(material, x).items():
            raw[raw_material] = raw.get(raw_material, 0) + y
    return raw


def get_total_raw_time(material):
    return crafting_times.get(material, 0) + sum(
        get_total_raw_time(ingr) for ingr in recipes.get(material, set())
    )


def get_intermediate_materials(material, amount=1):
    if not (recipe := recipes.get(material, None)):
        return {material: amount}
    materials = {material: amount}
    for material, x in recipe.items():
        for mat, x in get_intermediate_materials(material, x * amount).items():
            materials[mat] = materials.get(mat, 0) + x
    return materials


def get_assembly_plan(target_materials):
    materials = vector_sum(
        *(get_intermediate_materials(m, x) for m, x in target_materials.items())
    )
    G = {ingr: recipes.get(ingr, set()) for ingr in materials}
    return [(mat, materials[mat]) for mat in sort_topologically(G)]


targets = []
for target in input("targets: ").split(","):
    if ":" in target:
        target, target_amount = target.split(":")
        target_amount = float(target_amount)
    else:
        target_amount = 1.0
    target = target.strip().lower()
    if target not in materials:
        print(f"ERROR: what is {target}?")
        continue
    dt = get_total_raw_time(target)
    raw = get_total_raw_materials(target, target_amount)
    print(target, ":", dt, " required materials:", raw)
    targets.append((target, target_amount))
print(" ---- the plan ----")
for material, amount in get_assembly_plan(dict(targets)):
    if material not in recipes:
        print(f"run {amount} belts of {material}")
        continue
    ingredients = {m: x * amount for m, x in recipes[material].items()}
    asms = 2 * crafting_times[material] * amount * BELT_THROUGHPUT
    print(
        f"use {asms} assemblers to make {amount} belts of {material} "
        + f"out of {ingredients}"
    )
