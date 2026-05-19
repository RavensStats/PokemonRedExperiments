# adapted from https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py

import os
import json

MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}


def build_valid_tile_mask() -> "object":
    """Return a boolean mask of tiles that belong to any known map rectangle.

    This is not a collision/walkability map (we don't have per-tile collision here).
    It's a practical denominator for exploration percentage that ignores padding/blank
    areas outside the stitched map rectangles.
    """
    import numpy as np

    mask = np.zeros(GLOBAL_MAP_SHAPE, dtype=bool)
    for map_id, info in MAP_DATA.items():
        if map_id < 0:
            # -1 is the full Kanto bounding box entry
            continue
        map_x, map_y = info["coordinates"]
        width, height = info["tileSize"]
        r0 = int(map_y + MAP_ROW_OFFSET)
        c0 = int(map_x + MAP_COL_OFFSET)
        r1 = min(r0 + int(height), GLOBAL_MAP_SHAPE[0])
        c1 = min(c0 + int(width), GLOBAL_MAP_SHAPE[1])
        if r0 < 0 or c0 < 0:
            continue
        mask[r0:r1, c0:c1] = True
    return mask


VALID_TILE_MASK = build_valid_tile_mask()

# Handle KeyErrors
def local_to_global(r: int, c: int, map_n: int):
    try:
        (
            map_x,
            map_y,
        ) = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
