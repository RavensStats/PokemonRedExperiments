#!/usr/bin/env python3

"""Scan available checkpoints and rank them by short deterministic movement."""

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv


def load_run_dirs(root: Path):
    run_dirs = []
    for ckpt in root.glob("**/poke_*_steps.zip"):
        run_dir = ckpt.parent
        if run_dir.name.startswith("run_") and run_dir not in run_dirs:
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Sweep root to scan")
    parser.add_argument("--steps", type=int, default=200, help="Rollout steps per checkpoint")
    args = parser.parse_args()

    run_dirs = load_run_dirs(args.root)
    print(f"scanning {len(run_dirs)} runs")
    results = []

    for run_dir in run_dirs:
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        ta = meta.get("train_args", {})
        env_config = {
            "headless": True,
            "save_final_state": False,
            "early_stop": False,
            "action_freq": int(ta.get("action_freq", 24)),
            "init_state": str(ta.get("init_state", "init.state")),
            "max_steps": int(ta.get("ep_length", 163840)),
            "print_rewards": False,
            "save_video": False,
            "fast_video": True,
            "session_path": run_dir / "tmp_scan",
            "gb_path": str(ta.get("gb_path", "PokemonRed.gb")),
            "debug": False,
            "reward_scale": float(ta.get("reward_scale", 0.5)),
            "explore_weight": float(ta.get("explore_weight", 0.25)),
            "input_jitter_enable": False,
            "input_jitter_prob": 0.0,
            "input_jitter_mode": "lag",
            "discovered_events_enable": bool(ta.get("discovered_events_enable", False)),
            "discovered_events_promoted_path": str(ta.get("discovered_events_promoted_path", "")),
            "discovered_events_reward_weight": float(ta.get("discovered_events_reward_weight", 0.0)),
        }

        ckpt = sorted(run_dir.glob("poke_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]), reverse=True)[0]
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

        results.append((unique, span_x + span_y, span_x, span_y, map_count, run_dir))
        print(f"{run_dir} unique={unique} span=({span_x},{span_y}) maps={map_count}")

    print("\nTOP MOVERS:")
    for unique, score, span_x, span_y, map_count, run_dir in sorted(results, reverse=True)[:10]:
        print(f"unique={unique} span=({span_x},{span_y}) maps={map_count} run={run_dir}")


if __name__ == "__main__":
    main()
