#!/usr/bin/env python3
"""
Export trajectory (x, y, map) data from a trained agent for visualization.
Runs one episode and saves trajectory as gzipped CSV.
"""

import argparse
import json
import gzip
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Adjust path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "v2"))

from red_gym_env_v2 import RedGymEnv


def _load_train_args_from_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return dict(meta.get("train_args", {}) or {})
    except Exception:
        return {}


def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the latest poke_*_steps.zip checkpoint."""
    matches = list(run_dir.glob("poke_*_steps.zip"))
    if not matches:
        return None
    # Sort by number of steps (descending)
    return sorted(matches, key=lambda p: int(p.stem.split("_")[1]), reverse=True)[0]


def export_trajectory(run_dir: Path, output_path: Optional[Path] = None, num_steps: int = 100000) -> Path:
    """
    Run one episode with a trained model and export trajectory.
    
    Args:
        run_dir: Directory containing trained model and metadata
        output_path: Where to save trajectory CSV (default: run_dir/trajectory.csv.gz)
        num_steps: Max steps to run (episode will terminate early if game ends)
    
    Returns:
        Path to saved trajectory file
    """
    
    if output_path is None:
        output_path = run_dir / "trajectory.csv.gz"
    
    # Load metadata and checkpoint
    train_args = _load_train_args_from_metadata(run_dir)
    checkpoint_path = _find_latest_checkpoint(run_dir)
    
    if not checkpoint_path:
        raise FileNotFoundError(f"No poke_*_steps.zip checkpoint found in {run_dir}")
    
    print(f"[trajectory-export] checkpoint: {checkpoint_path}")
    
    # Build environment config from training args
    ep_length = int(train_args.get("ep_length", 163840))
    action_freq = int(train_args.get("action_freq", 24))
    init_state = str(train_args.get("init_state", "init.state"))
    gb_path = str(train_args.get("gb_path", "PokemonRed.gb"))
    reward_scale = float(train_args.get("reward_scale", 0.5))
    explore_weight = float(train_args.get("explore_weight", 0.25))
    discovered_events_enable = bool(train_args.get("discovered_events_enable", False))
    discovered_events_reward_weight = float(train_args.get("discovered_events_reward_weight", 0.0))
    
    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": action_freq,
        "init_state": init_state,
        "max_steps": ep_length,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": run_dir / "eval_trajectory",
        "gb_path": gb_path,
        "debug": False,
        "reward_scale": reward_scale,
        "explore_weight": explore_weight,
        "input_jitter_enable": False,
        "input_jitter_prob": 0.0,
        "input_jitter_mode": "lag",
        "discovered_events_enable": discovered_events_enable,
        "discovered_events_promoted_path": train_args.get("discovered_events_promoted_path", ""),
        "discovered_events_reward_weight": discovered_events_reward_weight,
    }
    
    print(f"[trajectory-export] creating environment...")
    env = RedGymEnv(env_config)
    
    print(f"[trajectory-export] loading model from {checkpoint_path}")
    model = PPO.load(str(checkpoint_path), env=env)
    
    # Run episode and collect trajectory
    trajectory = []
    obs, _info = env.reset()
    step = 0
    
    print(f"[trajectory-export] running episode (max {num_steps} steps)...")
    while step < num_steps:
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)

        # Extract position using env.get_game_coords() when available
        try:
            x_pos, y_pos, map_n = env.get_game_coords()
            trajectory.append({
                'step': step,
                'x': int(x_pos),
                'y': int(y_pos),
                'map': int(map_n),
            })
        except Exception:
            # Fallback: if env exposes agent_stats, use last stats
            try:
                stats = getattr(env, 'agent_stats', [])
                if stats:
                    last = stats[-1]
                    trajectory.append({
                        'step': step,
                        'x': int(last.get('x', 0)),
                        'y': int(last.get('y', 0)),
                        'map': int(last.get('map', 0)),
                    })
            except Exception:
                pass

        step += 1
        if bool(terminated or truncated):
            break
    
    print(f"[trajectory-export] collected {len(trajectory)} trajectory points")
    
    # Save as gzipped CSV
    df = pd.DataFrame(trajectory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        df.to_csv(f, index=False)
    
    print(f"[trajectory-export] wrote: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export trajectory from trained agent")
    parser.add_argument("run_dir", type=Path, help="Run directory with checkpoint and metadata")
    parser.add_argument("--output", type=Path, default=None, help="Output trajectory path (default: run_dir/trajectory.csv.gz)")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max steps to run (default: 100000)")
    
    args = parser.parse_args()
    
    try:
        output_path = export_trajectory(args.run_dir, args.output, args.max_steps)
        print(f"\nTrajectory saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
