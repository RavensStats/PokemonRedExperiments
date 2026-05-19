#!/usr/bin/env python3

"""Scan standalone PPO checkpoints in runs/ using the default env config."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing poke_*_steps.zip files")
    parser.add_argument("--steps", type=int, default=200, help="Rollout steps per checkpoint")
    args = parser.parse_args()

    ckpts = sorted(args.root.glob("poke_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"scanning {len(ckpts)} checkpoints")

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "init.state",
        "max_steps": 163840,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": args.root / "tmp_scan",
        "gb_path": "PokemonRed.gb",
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
        "input_jitter_enable": False,
        "input_jitter_prob": 0.0,
        "input_jitter_mode": "lag",
        "discovered_events_enable": False,
        "discovered_events_promoted_path": "",
        "discovered_events_reward_weight": 0.0,
    }

    results = []
    for ckpt in ckpts:
        env = RedGymEnv(env_config)
        model = PPO.load(str(ckpt), env=env)
        obs, _ = env.reset()
        coords = []
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            try:
                x, y, m = env.get_game_coords()
            except Exception:
                stats = getattr(env, "agent_stats", [])
                last = stats[-1] if stats else {}
                x, y, m = last.get("x", 0), last.get("y", 0), last.get("map", 0)
            coords.append((int(x), int(y), int(m)))
            if terminated or truncated:
                break

        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ms = [c[2] for c in coords]
            unique = len(set(coords))
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            map_count = len(set(ms))
        else:
            unique = span_x = span_y = map_count = 0

        results.append((unique, span_x + span_y, span_x, span_y, map_count, ckpt))
        print(f"{ckpt} unique={unique} span=({span_x},{span_y}) maps={map_count}")

    print("\nTOP MOVERS:")
    for unique, score, span_x, span_y, map_count, ckpt in sorted(results, reverse=True)[:10]:
        print(f"unique={unique} span=({span_x},{span_y}) maps={map_count} ckpt={ckpt}")


if __name__ == "__main__":
    main()
