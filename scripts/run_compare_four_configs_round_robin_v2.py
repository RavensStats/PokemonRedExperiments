#!/usr/bin/env python3
"""Run four V2 sweep variants in a round-robin order by seed.

Configs:
  1) base_no_events
  2) explicit_events_promoted
  3) dynamic_discovered_events
  4) dynamic_no_promoted_in_rank

After all training runs complete, the script runs the jitter and perception-noise
post-evaluations on each config sweep directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    extra_args: list[str]


@dataclass(frozen=True)
class RunSpec:
    config: ConfigSpec
    seed: int
    run_index: int
    run_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four V2 training configs round-robin by seed")
    parser.add_argument("--runs", type=int, default=20, help="Number of seeds per config")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed")
    parser.add_argument("--root-base-dir", type=str, default="sweeps", help="Base output directory")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for the sweep folder")
    parser.add_argument("--num-cpu", type=int, default=4)
    parser.add_argument("--ep-length", type=int, default=2048 * 80)
    parser.add_argument("--total-timesteps", type=int, default=330000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gb-path", type=str, default="PokemonRed.gb")
    parser.add_argument("--init-state", type=str, default="init.state")
    parser.add_argument("--action-freq", type=int, default=24)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-flush-every", type=int, default=1000)

    parser.add_argument("--eval-input-jitter-prob", type=float, default=0.1)
    parser.add_argument("--eval-input-jitter-mode", type=str, default="lag")
    parser.add_argument("--eval-input-jitter-episodes", type=int, default=1)
    parser.add_argument("--eval-input-jitter-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-perception-noise-radii", type=str, default="0,1,2,4,8")
    parser.add_argument("--eval-perception-noise-mode", type=str, default="uniform")
    parser.add_argument("--eval-perception-noise-episodes", type=int, default=1)
    parser.add_argument("--eval-perception-noise-deterministic", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def _safe_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)


def _make_root(base_dir: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"sweep_{timestamp}"
    if tag:
        name += f"_{_safe_tag(tag)}"
    root = Path(base_dir) / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_run_cmd(args: argparse.Namespace, run: RunSpec) -> list[str]:
    cmd = [
        sys.executable,
        str(Path("v2") / "baseline_fast_v2.py"),
        "--run-dir",
        str(run.run_dir),
        "--seed",
        str(run.seed),
        "--num-cpu",
        str(args.num_cpu),
        "--ep-length",
        str(args.ep_length),
        "--total-timesteps",
        str(args.total_timesteps),
        "--action-freq",
        str(args.action_freq),
        "--init-state",
        str(args.init_state),
        "--gb-path",
        str(args.gb_path),
        "--device",
        str(args.device),
        "--reward-scale",
        "0.5",
        "--explore-weight",
        "1.0",
        "--batch-size",
        "512",
        "--n-epochs",
        "1",
        "--gamma",
        "0.997",
        "--ent-coef",
        "0.05",
        "--save-trajectory" if args.save_trajectory else "--no-save-trajectory",
        "--trajectory-flush-every",
        str(args.trajectory_flush_every),
        "--stream-user",
        "v2-default",
        "--stream-color",
        "#447799",
    ]
    cmd.append("--headless" if args.headless else "--no-headless")
    cmd.append("--no-stream" if args.no_stream else "--stream")
    cmd.extend(run.config.extra_args)
    return cmd


def _write_run_metadata(run: RunSpec, cmd: list[str], args: argparse.Namespace) -> None:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": run.run_index,
        "seed": run.seed,
        "config": run.config.name,
        "run_dir": str(run.run_dir),
        "command": cmd,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "train_args": {
            "num_cpu": args.num_cpu,
            "ep_length": args.ep_length,
            "total_timesteps": args.total_timesteps,
            "action_freq": args.action_freq,
            "init_state": args.init_state,
            "gb_path": args.gb_path,
            "device": args.device,
            "headless": args.headless,
            "no_stream": args.no_stream,
            "save_trajectory": args.save_trajectory,
            "trajectory_flush_every": args.trajectory_flush_every,
            "explore_weight": 1.0,
            "ent_coef": 0.05,
        },
        "config_args": run.config.extra_args,
    }
    _write_json(run.run_dir / "run_metadata.json", payload)


def _write_exit_code(run_dir: Path, rc: int) -> None:
    (run_dir / "exit_code.txt").write_text(f"{int(rc)}\n", encoding="utf-8")


def _run_subprocess(cmd: list[str], run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ))
        return int(proc.wait())


def _is_completed(run_dir: Path) -> bool:
    exit_path = run_dir / "exit_code.txt"
    if not exit_path.exists():
        return False
    try:
        return int(exit_path.read_text(encoding="utf-8").strip()) == 0
    except Exception:
        return False


def _run_training(args: argparse.Namespace, runs: Iterable[RunSpec]) -> bool:
    failed = False
    total = 0
    for run in runs:
        total += 1
        print(f"[{total}] {run.config.name} seed={run.seed}")
        if args.resume and _is_completed(run.run_dir):
            print(f"  skipping completed run_dir={run.run_dir}")
            continue

        cmd = _make_run_cmd(args, run)
        _write_run_metadata(run, cmd, args)
        rc = _run_subprocess(cmd, run.run_dir)
        _write_exit_code(run.run_dir, rc)
        print(f"  rc={rc} run_dir={run.run_dir}")
        if rc != 0:
            failed = True
            break
    return not failed


def _run_post_eval(script: str, sweep_dir: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, script, str(sweep_dir)] + extra_args
    print("Running:", " ".join(cmd))
    return int(subprocess.call(cmd, env=dict(os.environ)))


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def main() -> int:
    args = _parse_args()
    if args.runs <= 0:
        print("--runs must be > 0")
        return 2

    sweep_root = _make_root(args.root_base_dir, args.tag)
    print(f"Sweep root: {sweep_root}")

    configs = [
        ConfigSpec("base_no_events", []),
        ConfigSpec("explicit_events_promoted", ["--discovered-events", "--discovered-events-promoted-path", "v2/events.json", "--discovered-events-reward-weight", "0.5"]),
        ConfigSpec("dynamic_discovered_events", ["--discovered-events", "--discovered-events-reward-weight", "0.5"]),
        ConfigSpec("dynamic_no_promoted_in_rank", ["--discovered-events", "--discovered-events-promoted-path", "", "--discovered-events-reward-weight", "0.5"]),
    ]

    for config in configs:
        (sweep_root / config.name).mkdir(parents=True, exist_ok=True)

    runs: list[RunSpec] = []
    for seed_offset in range(args.runs):
        seed = args.seed_start + seed_offset
        for config in configs:
            run_index = seed_offset
            run_dir = sweep_root / config.name / f"run_{run_index:03d}_seed{seed}"
            runs.append(RunSpec(config=config, seed=seed, run_index=run_index, run_dir=run_dir))

    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed_start": args.seed_start,
        "runs_per_config": args.runs,
        "configs": [c.name for c in configs],
        "runs": [
            {"config": r.config.name, "seed": r.seed, "run_index": r.run_index, "run_dir": str(r.run_dir)}
            for r in runs
        ],
    }
    _write_json(sweep_root / "sweep_plan.json", plan)

    print(f"Round-robin order: seed {args.seed_start} across all configs, then seed {args.seed_start + 1}, etc.")
    print("Training will complete before any post-eval runs start.")

    ok = _run_training(args, runs)
    if not ok:
        print("Training stopped early; post-evals not run.")
        return 1

    print("All training runs complete.")

    jitter_extra = [
        "--prefer",
        "max",
        "--episodes",
        str(args.eval_input_jitter_episodes),
        "--deterministic" if args.eval_input_jitter_deterministic else "--no-deterministic",
        "--jitter-prob",
        str(args.eval_input_jitter_prob),
        "--jitter-mode",
        str(args.eval_input_jitter_mode),
    ]
    perception_extra = [
        "--prefer",
        "max",
        "--episodes",
        str(args.eval_perception_noise_episodes),
        "--deterministic" if args.eval_perception_noise_deterministic else "--no-deterministic",
        "--noise-radii",
        str(args.eval_perception_noise_radii),
        "--noise-mode",
        str(args.eval_perception_noise_mode),
    ]

    for config in configs:
        config_sweep_dir = sweep_root / config.name
        print(f"Post-evals for {config.name}: {config_sweep_dir}")
        rc = _run_post_eval(str(Path("scripts") / "eval_input_jitter_best_v2.py"), config_sweep_dir, jitter_extra)
        if rc != 0:
            return rc
        rc = _run_post_eval(str(Path("scripts") / "eval_perception_noise_best_v2.py"), config_sweep_dir, perception_extra)
        if rc != 0:
            return rc

    print("Sweep and post-evals completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
