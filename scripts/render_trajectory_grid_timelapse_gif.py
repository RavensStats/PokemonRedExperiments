from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path('sweeps/sweep_20260517_175436')
CONFIGS = [
    'base_no_events',
    'explicit_events_promoted',
    'dynamic_discovered_events',
    'dynamic_no_promoted_in_rank',
]
BG_PATH = Path('visualization/poke_map/pokemap_full_calibrated_CROPPED_1.png')
OUT_PATH = Path('trajectory_grid_timelapse.gif')
SCALE = 0.25


def game_coord_to_global_coord(x, y, map_idx):
    global_offset = np.array([1056 - 16 * 12, 331])
    map_offsets = {
        0: np.array([0, 0]),
        1: np.array([-10, 72]),
        2: np.array([-10, 180]),
        12: np.array([0, 36]),
        13: np.array([0, 144]),
        14: np.array([30, 172]),
        15: np.array([80, 190]),
        33: np.array([-50, 64]),
        37: np.array([-9, 2]),
        38: np.array([-9, 25 - 32]),
        39: np.array([9 + 12, 2]),
        40: np.array([25 - 4, -6]),
        41: np.array([30, 47]),
        42: np.array([30, 55]),
        43: np.array([30, 72]),
        44: np.array([30, 64]),
        47: np.array([21, 136]),
        49: np.array([21, 108]),
        50: np.array([21, 108]),
        51: np.array([-35, 137]),
        52: np.array([-10, 189]),
        53: np.array([-10, 198]),
        54: np.array([-21, 169]),
        55: np.array([-19, 177]),
        56: np.array([-30, 163]),
        57: np.array([-19, 177]),
        58: np.array([-25, 154]),
        59: np.array([83, 227]),
        60: np.array([123, 227]),
        61: np.array([152, 227]),
        68: np.array([65, 190]),
    }
    offset = map_offsets.get(map_idx, np.array([0, 0]))
    if map_idx not in map_offsets:
        x, y = 0, 0
    coord = global_offset + 16 * (offset + np.array([x, y]))
    return coord


def draw_point(arr, x, y, color, size=3):
    h, w = arr.shape[:2]
    x0 = max(0, x - size)
    y0 = max(0, y - size)
    x1 = min(w, x + size + 1)
    y1 = min(h, y + size + 1)
    if x0 >= x1 or y0 >= y1:
        return
    arr[y0:y1, x0:x1, :3] = color[:3]
    arr[y0:y1, x0:x1, 3] = 255


def make_panel(panel_rgba, label):
    img = Image.fromarray(panel_rgba)
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 260, 44), fill=(0, 0, 0, 160))
    draw.text((18, 16), label, fill=(255, 255, 255, 255))
    return img


def main(max_frames=80):
    bg_img = Image.open(BG_PATH).convert('RGBA')
    bg = np.array(bg_img.resize((int(bg_img.width * SCALE), int(bg_img.height * SCALE)), resample=Image.Resampling.BILINEAR))
    data = {}
    for cfg in CONFIGS:
        p = ROOT / cfg / 'run_000_seed0' / 'trajectory.csv.gz'
        df = pd.read_csv(p, compression='gzip')
        df = df[df['map'] != 'map'].reset_index(drop=True)
        data[cfg] = df

    min_len = min(len(df) for df in data.values())
    frame_indices = np.linspace(0, min_len - 1, min(max_frames, min_len)).astype(int)

    colors = {
        'base_no_events': np.array([255, 90, 90, 255], dtype=np.uint8),
        'explicit_events_promoted': np.array([90, 255, 120, 255], dtype=np.uint8),
        'dynamic_discovered_events': np.array([90, 160, 255, 255], dtype=np.uint8),
        'dynamic_no_promoted_in_rank': np.array([255, 210, 90, 255], dtype=np.uint8),
    }

    panel_buffers = {cfg: bg.copy() for cfg in CONFIGS}
    last_idx = {cfg: -1 for cfg in CONFIGS}
    frames = []

    for frame_idx in frame_indices:
        panels = []
        for cfg in CONFIGS:
            start = last_idx[cfg] + 1
            if frame_idx >= start:
                for idx in range(start, frame_idx + 1):
                    row = data[cfg].iloc[idx]
                    coord = game_coord_to_global_coord(int(row['x']), -int(row['y']), int(row['map']))
                    draw_point(panel_buffers[cfg], int(coord[0] * SCALE), int(coord[1] * SCALE), colors[cfg], size=1)
                last_idx[cfg] = frame_idx
            panels.append(make_panel(panel_buffers[cfg], f'{cfg}  step {frame_idx}/{min_len - 1}'))

        top = np.concatenate([np.array(panels[0]), np.array(panels[1])], axis=1)
        bottom = np.concatenate([np.array(panels[2]), np.array(panels[3])], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        frames.append(Image.fromarray(grid[..., :3]))

    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=False,
    )
    print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
