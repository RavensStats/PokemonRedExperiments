import pandas as pd
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import math


def make_all_coords_arrays(filtered_dfs):
    return np.array([tdf[['x', 'y', 'map']].to_numpy().astype(np.uint8) for tdf in filtered_dfs]).transpose(1,0,2)


def game_coord_to_global_coord(x, y, map_idx):
    map_offsets = {
        0: np.array([0,0]),
        1: np.array([-10, 72]),
        2: np.array([-10, 180]),
        12: np.array([0, 36]),
        13: np.array([0, 144]),
        14: np.array([30, 172]),
        15: np.array([80, 190]),
        33: np.array([-50, 64]),
        37: np.array([-9, 2]),
        38: np.array([-9, 25-32]),
        39: np.array([9+12, 2]),
        40: np.array([25-4, -6]),
        41: np.array([30, 47]),
        42: np.array([30, 55]),
        43: np.array([30, 72]),
        44: np.array([30, 64]),
        47: np.array([21,136]),
        49: np.array([21,108]),
        50: np.array([21,108]),
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
    if map_idx in map_offsets:
        offset = map_offsets[map_idx]
    else:
        offset = np.array([0,0])
        x, y = 0, 0
    return offset + np.array([x,y])


def compute_flow(all_coords, inter_steps=1, add_start=True):
    errors = []
    sprites_rendered = 0
    all_flows = {}
    step_count = len(all_coords)
    state = [{'dir': 0, 'map': 40} for _ in all_coords[0]]
    for idx in tqdm(range(0, step_count)):
        step = all_coords[idx]
        if idx > 0:
            prev_step = all_coords[idx-1]
        elif add_start:
            prev_step = np.tile(np.array([5, 3, 40]), (all_coords.shape[1], 1))
        else:
            prev_step = all_coords[idx]
        for fract in np.arange(0,1,1/inter_steps):
            for run in range(len(step)):
                cur = step[run]
                prev = prev_step[run]
                cx, cy, px, py = map(int, [cur[0], cur[1], prev[0], prev[1]])
                dx = cx - px
                dy = cy - py
                total_delta = abs(dx) + abs(dy)
                if total_delta > 1:
                    state[run]['map'] = cur[2]
                dx = min(max(dx, -1), 1)
                dy = -1*min(max(dy, -1), 1)
                if cur[2] == prev[2]:
                    pass
                p_coord = game_coord_to_global_coord(cx, -cy, state[run]['map'])
                prev_p_coord = game_coord_to_global_coord(px, -py, prev[2])
                diff = p_coord - prev_p_coord
                if np.linalg.norm(diff) > 2:
                    continue
                coords_tup = tuple(prev_p_coord.tolist())
                if coords_tup in all_flows:
                    all_flows[coords_tup] += diff
                else:
                    all_flows[coords_tup] = diff
                sprites_rendered += 1
    return all_flows


def render_arrows(fname, all_flows, arrow_sprite_pth):
    arrow_img = Image.open(arrow_sprite_pth)
    # resize arrow to small cell (16x16) for consistent tiling
    arrow_img = arrow_img.resize((16, 16))
    min_x = min([k[0] for k in all_flows.keys()])
    max_x = max([k[0] for k in all_flows.keys()])
    min_y = min([k[1] for k in all_flows.keys()])
    max_y = max([k[1] for k in all_flows.keys()])
    grid_dims = (max_x - min_x, max_y - min_y)
    cell_dim = arrow_img.size[0]
    full_img = np.zeros( ((grid_dims[0]+1) * cell_dim, (grid_dims[1]+1) * cell_dim, 4 ), dtype=np.uint8)
    for coord, total_move in all_flows.items():
        angle = math.atan2(-total_move[0], total_move[1])
        rotated_arrow = arrow_img.rotate(180*angle/math.pi, resample=Image.Resampling.BICUBIC)
        nx = coord[0] - min_x
        ny = coord[1] - min_y
        full_img[ nx * cell_dim : (nx + 1) * cell_dim, ny * cell_dim : (ny + 1) * cell_dim ] = np.array(rotated_arrow)
    final_img = Image.fromarray(full_img)
    final_img.save(f"{fname}.png")


if __name__ == '__main__':
    root = Path('sweeps/sweep_20260517_175436')
    trajs = list(root.glob('*/run_000_seed0/trajectory.csv.gz'))
    if len(trajs) == 0:
        print('No trajectory files found under', root)
        raise SystemExit(1)
    dfs = []
    for p in trajs:
        tdf = pd.read_csv(p, compression='gzip')
        dfs.append(tdf[tdf['map'] != 'map'])
    # truncate to min length to avoid ragged arrays
    min_len = min([len(t) for t in dfs])
    dfs = [t.iloc[:min_len].reset_index(drop=True) for t in dfs]
    base_coords = make_all_coords_arrays(dfs)
    print('base_coords shape', base_coords.shape)
    # compute flow using 1 inter step
    all_flows = compute_flow(base_coords, inter_steps=1)
    render_arrows('map_flow_sweep_20260517_175436', all_flows, 'visualization/poke_map/transparent_arrow.png')
