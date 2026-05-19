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
CHAR_PATH = Path('visualization/poke_map/characters.png')
OUT_PATH = Path('trajectory_grid_timelapse_player.gif')
VIEW_RADIUS = 180
PANEL_SCALE = 2
SPRITE_SCALE = 7


def get_sprite_by_coords(img, x, y):
    sy = 34 + 17 * y
    sx = 9 + 17 * x
    alpha_v = np.array([255, 127, 39, 255], dtype=np.uint8)
    sprite = img[sy:sy + 16, sx:sx + 16]
    return np.where((sprite == alpha_v).all(axis=2).reshape(16, 16, 1), np.array([[[0, 0, 0, 0]]]), sprite).astype(np.uint8)


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


def draw_point(arr, x, y, color, size=2):
    h, w = arr.shape[:2]
    x0 = max(0, x - size)
    y0 = max(0, y - size)
    x1 = min(w, x + size + 1)
    y1 = min(h, y + size + 1)
    if x0 >= x1 or y0 >= y1:
        return
    arr[y0:y1, x0:x1, :3] = color[:3]
    arr[y0:y1, x0:x1, 3] = 255


def crop_view(panel_rgba, center_x, center_y, radius=VIEW_RADIUS):
    h, w = panel_rgba.shape[:2]
    left = max(0, center_x - radius)
    top = max(0, center_y - radius)
    right = min(w, center_x + radius)
    bottom = min(h, center_y + radius)
    crop = panel_rgba[top:bottom, left:right]
    return np.array(Image.fromarray(crop).resize((radius * 2 * PANEL_SCALE, radius * 2 * PANEL_SCALE), resample=Image.Resampling.BILINEAR))


def draw_sprite(panel_rgba, sprite, x, y):
    img = Image.fromarray(panel_rgba)
    sprite_img = Image.fromarray(sprite).resize(
        (sprite.shape[1] * SPRITE_SCALE, sprite.shape[0] * SPRITE_SCALE),
        resample=Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(img)
    halo_r = max(sprite_img.width, sprite_img.height) // 2 + 3
    draw.ellipse((x - halo_r, y - halo_r, x + halo_r, y + halo_r), outline=(255, 255, 0, 255), width=3)
    img.paste(sprite_img, (x - sprite_img.width // 2, y - sprite_img.height // 2), sprite_img)
    return np.array(img)


def make_panel(panel_rgba, label):
    img = Image.fromarray(panel_rgba)
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 320, 46), fill=(0, 0, 0, 160))
    draw.text((18, 16), label, fill=(255, 255, 255, 255))
    return img


def main(max_frames=80):
    bg_img = Image.open(BG_PATH).convert('RGBA')
    bg = np.array(bg_img)
    chars_img = np.array(Image.open(CHAR_PATH))
    walk_sprites = [get_sprite_by_coords(chars_img, x, 0) for x in [1, 4, 6, 8]]

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
                    draw_point(panel_buffers[cfg], int(coord[0]), int(coord[1]), colors[cfg], size=2)
                last_idx[cfg] = frame_idx

            panel_img = panel_buffers[cfg].copy()
            cur = data[cfg].iloc[frame_idx]
            cur_coord = game_coord_to_global_coord(int(cur['x']), -int(cur['y']), int(cur['map']))
            if frame_idx > 0:
                prev = data[cfg].iloc[frame_idx - 1]
                prev_coord = game_coord_to_global_coord(int(prev['x']), -int(prev['y']), int(prev['map']))
                delta = cur_coord - prev_coord
                dx, dy = int(delta[0]), int(delta[1])
                if abs(dx) >= abs(dy):
                    sprite = walk_sprites[0] if dx >= 0 else walk_sprites[1]
                else:
                    sprite = walk_sprites[3] if dy >= 0 else walk_sprites[2]
                panel_img = draw_sprite(panel_img, sprite, int(cur_coord[0]), int(cur_coord[1]))

            zoom_panel = crop_view(panel_img, int(cur_coord[0]), int(cur_coord[1]))
            panels.append(make_panel(zoom_panel, f'{cfg}  step {frame_idx}/{min_len - 1}'))

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
