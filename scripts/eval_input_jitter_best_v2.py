import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


_CHECKPOINT_RE = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate input jitter robustness using the best-performing checkpoint from a completed V2 sweep. "
            "Selects the top run from summary.csv (or generates it), finds the latest poke_* checkpoint, "
            "then runs a headless evaluation episode with input jitter enabled."
        )
    )
    p.add_argument("sweep_dir", type=str, help="Sweep folder (contains run_* subfolders)")

    p.add_argument(
        "--prefer",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Use env_stats_max/* or env_stats/* tags when summarizing",
    )

    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to run",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic=True for model.predict()",
    )

    # Jitter config
    p.add_argument(
        "--jitter-prob",
        type=float,
        default=0.1,
        help="Probability of jitter per step (used if --jitter-probs not provided)",
    )
    p.add_argument(
        "--jitter-probs",
        type=str,
        default="",
        help="Comma-separated jitter probabilities to sweep (e.g. 0,0.05,0.1,0.2)",
    )
    p.add_argument(
        "--jitter-mode",
        type=str,
        default="lag",
        choices=["lag", "sticky", "repeat", "drift", "direction", "random", "rand"],
        help="Jitter type",
    )

    # Success criteria (simple success/failure)
    p.add_argument(
        "--success-badges",
        type=int,
        default=1,
        help="Success if final badge count >= this (set 0 to disable)",
    )
    p.add_argument(
        "--success-milestone-score",
        type=int,
        default=-1,
        help="Success if final game_completion_score >= this (set -1 to disable)",
    )
    p.add_argument(
        "--success-events-completed",
        type=int,
        default=-1,
        help="Success if final events_completed >= this (set -1 to disable)",
    )
    p.add_argument(
        "--success-requires",
        type=str,
        default="any",
        choices=["any", "all"],
        help="If multiple success thresholds are enabled, require any vs all",
    )

    # Env overrides (fallback if run_metadata.json is missing)
    p.add_argument("--ep-length", type=int, default=2048 * 80, help="Max steps per eval episode")
    p.add_argument("--action-freq", type=int, default=24, help="Emulator ticks per agent action")
    p.add_argument("--init-state", type=str, default="init.state", help="Initial game state (.state)")
    p.add_argument("--gb-path", type=str, default="PokemonRed.gb", help="Path to Pokemon Red ROM")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)

    # Output
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: <best_run_dir>/jitter_eval.json)",
    )

    return p.parse_args()


def _run_summarize(sweep_dir: Path, prefer: str) -> Path:
    out_csv = sweep_dir / "summary.csv"
    cmd = [sys.executable, str(Path("scripts") / "summarize_sweep_v2.py"), str(sweep_dir), "--prefer", str(prefer)]
    import subprocess

    subprocess.run(cmd, check=True)
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected summary CSV not found: {out_csv}")
    return out_csv


def _pick_best_run_from_csv(summary_csv: Path) -> Path:
    import csv

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in summary: {summary_csv}")
    run_dir = (rows[0].get("run_dir") or "").strip()
    if not run_dir:
        raise RuntimeError(f"Missing run_dir in summary: {summary_csv}")
    return Path(run_dir)


def _find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    best: Tuple[int, Path] | None = None
    for p in run_dir.glob("*.zip"):
        m = _CHECKPOINT_RE.match(p.name)
        if not m:
            continue
        steps = int(m.group("steps"))
        if best is None or steps > best[0]:
            best = (steps, p)
    return best[1] if best else None


def _load_train_args_from_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return dict(meta.get("train_args", {}) or {})
    except Exception:
        return {}


def _make_env_config(run_dir: Path, args: argparse.Namespace) -> dict:
    train_args = _load_train_args_from_metadata(run_dir)

    # Default to training-time settings when available; otherwise fallback to CLI.
    def _pick(key: str, fallback):
        v = train_args.get(key, None)
        return fallback if v is None else v

    ep_length = int(_pick("ep_length", int(args.ep_length)))
    action_freq = int(_pick("action_freq", int(args.action_freq)))
    init_state = str(_pick("init_state", str(args.init_state)))
    gb_path = str(_pick("gb_path", str(args.gb_path)))
    headless = bool(_pick("headless", bool(args.headless)))
    reward_scale = float(_pick("reward_scale", 0.5))
    explore_weight = float(_pick("explore_weight", 0.25))

    # Make a dedicated eval folder inside the run so files (maps, logs) don't collide with training outputs.
    eval_dir = run_dir / "eval_jitter"
    eval_dir.mkdir(parents=True, exist_ok=True)

    return {
        "headless": headless,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": action_freq,
        "init_state": init_state,
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": eval_dir,
        "gb_path": gb_path,
        "debug": False,
        "reward_scale": reward_scale,
        "explore_weight": explore_weight,
        # Input jitter (prob is filled per-sweep value in caller)
        "input_jitter_enable": True,
        "input_jitter_prob": 0.0,
        "input_jitter_mode": str(args.jitter_mode),
        # Keep discovered-events off for eval unless the user explicitly needs it.
        "discovered_events_enable": False,
        "discovered_events_promoted_path": "",
        "discovered_events_reward_weight": 0.0,
        "discovered_events_flush_every": 0,
    }


def _run_eval_episode(model, env, *, deterministic: bool) -> dict:
    obs, _info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action, _state = model.predict(obs, deterministic=bool(deterministic))
        obs, reward, _terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if bool(truncated):
            break

    # env.agent_stats is a list of dicts; last entry is a convenient episode summary
    final_stats = env.agent_stats[-1] if getattr(env, "agent_stats", None) else {}
    return {
        "steps": int(steps),
        "total_reward_delta": float(total_reward),
        "final_stats": final_stats,
        "input_jitter_count": int(final_stats.get("input_jitter_count", 0) or 0),
    }


def _parse_prob_list(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _success_from_stats(final_stats: dict, args: argparse.Namespace) -> bool:
    enabled_checks = []

    badges_thr = int(args.success_badges)
    if badges_thr > 0:
        enabled_checks.append(int(final_stats.get("badge", 0) or 0) >= badges_thr)

    ms_thr = int(args.success_milestone_score)
    if ms_thr >= 0:
        enabled_checks.append(int(final_stats.get("game_completion_score", 0) or 0) >= ms_thr)

    ev_thr = int(args.success_events_completed)
    if ev_thr >= 0:
        enabled_checks.append(int(final_stats.get("events_completed", 0) or 0) >= ev_thr)

    # If all thresholds are disabled, treat every run as successful (but still report raw stats).
    if not enabled_checks:
        return True

    if str(args.success_requires) == "all":
        return all(enabled_checks)
    return any(enabled_checks)


def main() -> int:
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep dir not found: {sweep_dir}")
        return 2

    summary_csv = sweep_dir / "summary.csv"
    if not summary_csv.exists():
        summary_csv = _run_summarize(sweep_dir, str(args.prefer))

    best_run_dir = _pick_best_run_from_csv(summary_csv)
    if not best_run_dir.exists():
        print(f"Best run dir not found: {best_run_dir}")
        return 2

    ckpt = _find_latest_checkpoint(best_run_dir)
    if ckpt is None:
        print(f"No checkpoints found in: {best_run_dir}")
        return 2

    # Lazy imports so summarize-only usage doesn't require SB3
    # Note: v2/ is not a Python package, so add it to sys.path for imports.
    v2_dir = Path(__file__).resolve().parents[1] / "v2"
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

    from stable_baselines3 import PPO
    from red_gym_env_v2 import RedGymEnv

    base_env_config = _make_env_config(best_run_dir, args)

    probs = _parse_prob_list(str(args.jitter_probs))
    if not probs:
        probs = [float(args.jitter_prob)]

    # SB3 PPO expects the .zip path
    print(f"[jitter-eval] best_run={best_run_dir}")
    print(f"[jitter-eval] checkpoint={ckpt}")
    print(f"[jitter-eval] jitter: mode={args.jitter_mode} probs={probs}")

    results = []
    for p_jit in probs:
        env_config = dict(base_env_config)
        env_config["input_jitter_prob"] = float(p_jit)
        env = RedGymEnv(env_config)
        model = PPO.load(str(ckpt), env=env, custom_objects={"lr_schedule": 0, "clip_range": 0})

        episodes = []
        successes = 0
        for ep in range(int(args.episodes)):
            ep_res = {
                "episode": int(ep),
                **_run_eval_episode(model, env, deterministic=bool(args.deterministic)),
            }
            ep_res["success"] = bool(_success_from_stats(ep_res.get("final_stats", {}) or {}, args))
            if ep_res["success"]:
                successes += 1
            episodes.append(ep_res)

        results.append(
            {
                "jitter_prob": float(p_jit),
                "jitter_mode": str(args.jitter_mode),
                "episodes": episodes,
                "successes": int(successes),
                "episodes_n": int(args.episodes),
                "success_rate": float(successes) / float(max(int(args.episodes), 1)),
            }
        )

        try:
            env.close()
        except Exception:
            pass

    out_path = Path(args.out) if str(args.out).strip() else (best_run_dir / "jitter_eval.json")
    json_base_env_config = _json_safe(base_env_config)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sweep_dir": str(sweep_dir),
        "summary_csv": str(summary_csv),
        "best_run_dir": str(best_run_dir),
        "checkpoint": str(ckpt),
        "success_criteria": {
            "badges_at_least": int(args.success_badges),
            "milestone_score_at_least": int(args.success_milestone_score),
            "events_completed_at_least": int(args.success_events_completed),
            "requires": str(args.success_requires),
        },
        "base_env_config": json_base_env_config,
        "eval": {
            "deterministic": bool(args.deterministic),
            "sweep": _json_safe(results),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[jitter-eval] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
