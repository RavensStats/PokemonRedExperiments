import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gzip


@dataclass(frozen=True)
class RunSpec:
    index: int
    seed: int
    run_dir: Path
    resume: str
    promoted_path: str = ""


class _StdoutCapture:
    def __init__(
        self,
        cmd: List[str],
        run_dir: Path,
        *,
        gzip_stdout: bool,
        gzip_level: int,
    ) -> None:
        self._cmd = cmd
        self._run_dir = run_dir
        self._gzip_stdout = bool(gzip_stdout)
        self._gzip_level = int(gzip_level)

        self._plain_fp = None
        self._gz_fp = None
        self._proc_stdout = None
        self._thread = None

    def popen_kwargs(self) -> dict:
        if not self._gzip_stdout:
            stdout_path = self._run_dir / "stdout.log"
            self._plain_fp = stdout_path.open("w", encoding="utf-8")
            self._plain_fp.write("COMMAND:\n" + " ".join(self._cmd) + "\n\n")
            self._plain_fp.flush()
            return {"stdout": self._plain_fp, "stderr": subprocess.STDOUT}

        pointer_path = self._run_dir / "stdout.log"
        gz_path = self._run_dir / "stdout.log.gz"
        pointer_path.write_text(
            "COMMAND:\n"
            + " ".join(self._cmd)
            + "\n\n"
            + "NOTE: Full stdout/stderr is stored in stdout.log.gz (gzip, lossless).\n",
            encoding="utf-8",
        )

        self._gz_fp = gzip.open(gz_path, mode="wb", compresslevel=self._gzip_level)
        header = ("COMMAND:\n" + " ".join(self._cmd) + "\n\n").encode("utf-8")
        self._gz_fp.write(header)
        self._gz_fp.flush()

        return {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}

    def attach_process(self, proc: subprocess.Popen) -> None:
        if not self._gzip_stdout:
            return
        if proc.stdout is None:
            raise RuntimeError("Expected proc.stdout when gzip stdout capture enabled")

        self._proc_stdout = proc.stdout

        def _pump() -> None:
            assert self._proc_stdout is not None
            assert self._gz_fp is not None
            try:
                while True:
                    chunk = self._proc_stdout.read(1024 * 64)
                    if not chunk:
                        break
                    self._gz_fp.write(chunk)
            finally:
                try:
                    self._gz_fp.flush()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_pump, name=f"stdout-gzip-{self._run_dir.name}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self._proc_stdout is not None:
            try:
                self._proc_stdout.close()
            except Exception:
                pass
        if self._plain_fp is not None:
            try:
                self._plain_fp.close()
            except Exception:
                pass
        if self._gz_fp is not None:
            try:
                self._gz_fp.close()
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative V2 training: run a short sweep stage, pick best checkpoint, resume for next stage, repeat."
        )
    )

    # Iteration control
    parser.add_argument("--stages", type=int, default=3, help="Number of stages to run")
    parser.add_argument("--stage-hours", type=float, default=2.0, help="Wall-clock hours per stage")
    parser.add_argument(
        "--stage-hours-0",
        type=float,
        default=0.0,
        help="Wall-clock hours for stage 0 (0 means use --stage-hours)",
    )
    parser.add_argument(
        "--stage-hours-rest",
        type=float,
        default=0.0,
        help="Wall-clock hours for stages 1..N (0 means use --stage-hours)",
    )
    parser.add_argument(
        "--terminate-at-deadline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate active runs when the stage time budget is reached",
    )

    # Stage 0 settings (multi-seed tournament)
    parser.add_argument("--initial-runs", type=int, default=5, help="Number of candidate runs in stage 0")
    parser.add_argument(
        "--initial-max-parallel",
        type=int,
        default=1,
        help="Max parallel runs in stage 0 (set to initial-runs to give each full time concurrently)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds for stage 0 (overrides --seed-start)",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First seed for stage 0")

    # Later stages settings (usually 1 continuation)
    parser.add_argument("--continue-runs", type=int, default=1, help="Runs to launch in stages 1..N (default: 1)")
    parser.add_argument(
        "--continue-max-parallel",
        type=int,
        default=1,
        help="Max parallel runs in stages 1..N",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sweeps",
        help="Base output directory; creates iterative_<timestamp>_* subfolder",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag added to output folder name")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="",
        help="Optional fixed root dir (enables pause/resume by re-running with --resume-root)",
    )
    parser.add_argument(
        "--resume-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume an existing --root-dir from last saved stage state",
    )

    # Ranking
    parser.add_argument(
        "--prefer",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Use env_stats_max/* or env_stats/* when summarizing",
    )

    # Experiment mode
    parser.add_argument(
        "--mode",
        type=str,
        default="explicit",
        choices=["explicit", "discovered_explicit", "discovered_no_explicit"],
        help=(
            "explicit: no discovered-event shaping; "
            "discovered_explicit: discover+promote+shape, rank by explicit metrics; "
            "discovered_no_explicit: discover+promote+shape, rank ignoring events_completed"
        ),
    )

    # Discovered-event promotion/shaping knobs (used in discovered_* modes)
    parser.add_argument("--promote-top-k", type=int, default=10, help="Promotion: use top K runs")
    parser.add_argument("--promote-max-events", type=int, default=200, help="Promotion: max events to promote")
    parser.add_argument(
        "--promote-min-total-runs",
        type=int,
        default=1,
        help="Promotion: require event to appear in >= this many runs",
    )
    parser.add_argument(
        "--discovered-events-reward-weight",
        type=float,
        default=0.05,
        help="Shaping weight for promoted discovered events (discovered_* modes)",
    )
    parser.add_argument(
        "--discovered-events-flush-every",
        type=int,
        default=500,
        help="Flush discovered_events.json every N env steps",
    )

    # Stdout capture
    parser.add_argument(
        "--stdout-gzip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write full process output to stdout.log.gz (lossless)",
    )
    parser.add_argument(
        "--stdout-gzip-level",
        type=int,
        default=6,
        help="Gzip compression level (1=fastest, 9=smallest)",
    )

    # Forwarded to v2/baseline_fast_v2.py
    parser.add_argument("--num-cpu", type=int, default=4)
    parser.add_argument("--ep-length", type=int, default=2048 * 80)
    parser.add_argument("--total-timesteps", type=int, default=0)
    parser.add_argument("--train-steps-batch", type=int, default=0)
    parser.add_argument("--action-freq", type=int, default=24)
    parser.add_argument("--init-state", type=str, default="init.state")
    parser.add_argument("--gb-path", type=str, default="PokemonRed.gb")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-scale", type=float, default=0.5)
    parser.add_argument("--explore-weight", type=float, default=0.25)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-user", type=str, default="v2-default")
    parser.add_argument("--stream-color", type=str, default="#447799")
    parser.add_argument("--stream-extra", type=str, default="")

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def _parse_seeds(seeds_str: str, seed_start: int, runs: int) -> List[int]:
    if seeds_str.strip():
        return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    return [int(seed_start) + i for i in range(int(runs))]


def _safe_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (tag or "").strip())


def _make_root_dir(base_out_dir: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"iterative_{timestamp}"
    if tag:
        name += f"_{_safe_tag(tag)}"
    root = Path(base_out_dir) / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def _make_or_use_root_dir(args: argparse.Namespace) -> Path:
    if str(args.root_dir).strip():
        root = Path(str(args.root_dir))
        if root.exists() and not bool(args.resume_root):
            raise FileExistsError(
                f"Root dir already exists: {root} (pass --resume-root to resume)"
            )
        root.mkdir(parents=True, exist_ok=True)
        return root
    # default: timestamped
    return _make_root_dir(args.out_dir, args.tag)


def _load_resume_state(root_dir: Path) -> dict:
    p = root_dir / "iterative_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_resume_state(root_dir: Path, state: dict) -> None:
    p = root_dir / "iterative_state.json"
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _build_train_cmd(args: argparse.Namespace, spec: RunSpec) -> List[str]:
    train_script = Path("v2") / "baseline_fast_v2.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--run-dir",
        str(spec.run_dir),
        "--seed",
        str(spec.seed),
        "--num-cpu",
        str(args.num_cpu),
        "--ep-length",
        str(args.ep_length),
        "--total-timesteps",
        str(args.total_timesteps),
        "--train-steps-batch",
        str(args.train_steps_batch),
        "--action-freq",
        str(args.action_freq),
        "--init-state",
        str(args.init_state),
        "--gb-path",
        str(args.gb_path),
        ("--headless" if bool(args.headless) else "--no-headless"),
        "--reward-scale",
        str(args.reward_scale),
        "--explore-weight",
        str(args.explore_weight),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--gamma",
        str(args.gamma),
        "--ent-coef",
        str(args.ent_coef),
        "--device",
        str(args.device),
        ("--stream" if bool(args.stream) else "--no-stream"),
        "--stream-user",
        str(args.stream_user),
        "--stream-color",
        str(args.stream_color),
        "--stream-extra",
        str(args.stream_extra),
        ("--wandb" if bool(args.wandb) else "--no-wandb"),
    ]
    if spec.resume:
        cmd.extend(["--resume", str(spec.resume)])

    # Discovered-events integration (only in discovered_* modes)
    if str(args.mode).startswith("discovered"):
        cmd.extend(
            [
                "--discovered-events",
                "--discovered-events-reward-weight",
                str(args.discovered_events_reward_weight),
                "--discovered-events-flush-every",
                str(args.discovered_events_flush_every),
            ]
        )
        if str(spec.promoted_path).strip():
            cmd.extend(["--discovered-events-promoted-path", str(spec.promoted_path)])
    return cmd


def _choose_rank_mode(mode: str) -> str:
    return "no_explicit_events" if str(mode) == "discovered_no_explicit" else "explicit"


def _pick_best_run_from_csv_with_rank_mode(summary_csv: Path, *, rank_mode: str) -> Path:
    def nz(v: str) -> float:
        try:
            s = str(v).strip()
            return float(s) if s else float("-inf")
        except Exception:
            return float("-inf")

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in summary: {summary_csv}")

    def key(r: dict) -> tuple:
        if rank_mode == "no_explicit_events":
            return (
                nz(r.get("badge_count", "")),
                nz(r.get("map_explored_pct", "")),
                nz(r.get("highest_pokemon_level", "")),
            )
        return (
            nz(r.get("badge_count", "")),
            nz(r.get("events_completed", "")),
            nz(r.get("map_explored_pct", "")),
            nz(r.get("highest_pokemon_level", "")),
        )

    best = max(rows, key=key)
    run_dir = str(best.get("run_dir", "")).strip()
    if not run_dir:
        raise RuntimeError(f"Missing run_dir in best row: {summary_csv}")
    return Path(run_dir)


def _run_promotion(
    stage_dir: Path,
    *,
    rank_mode: str,
    top_k: int,
    max_events: int,
    min_total_runs: int,
) -> Path:
    out_path = stage_dir / "promoted_discovered_events.json"
    cmd = [
        sys.executable,
        str(Path("scripts") / "promote_discovered_events_v2.py"),
        str(stage_dir),
        "--rank-mode",
        str(rank_mode),
        "--top-k",
        str(int(top_k)),
        "--max-events",
        str(int(max_events)),
        "--min-total-runs",
        str(int(min_total_runs)),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _write_run_metadata(args: argparse.Namespace, stage_dir: Path, stage_index: int, spec: RunSpec) -> None:
    meta = {
        "stage": int(stage_index),
        "index": spec.index,
        "seed": spec.seed,
        "run_dir": str(spec.run_dir),
        "resume": str(spec.resume) if spec.resume else "",
        "mode": str(args.mode),
        "rank_mode": _choose_rank_mode(str(args.mode)),
        "promoted_discovered_events": str(spec.promoted_path) if str(spec.promoted_path).strip() else "",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "train_args": {
            "num_cpu": args.num_cpu,
            "ep_length": args.ep_length,
            "total_timesteps": args.total_timesteps,
            "train_steps_batch": args.train_steps_batch,
            "action_freq": args.action_freq,
            "init_state": args.init_state,
            "gb_path": args.gb_path,
            "headless": bool(args.headless),
            "reward_scale": args.reward_scale,
            "explore_weight": args.explore_weight,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "device": args.device,
            "stream": bool(args.stream),
            "stream_user": args.stream_user,
            "stream_color": args.stream_color,
            "stream_extra": args.stream_extra,
            "wandb": bool(args.wandb),
            "discovered_events_reward_weight": float(args.discovered_events_reward_weight)
            if str(args.mode).startswith("discovered")
            else 0.0,
            "discovered_events_flush_every": int(args.discovered_events_flush_every)
            if str(args.mode).startswith("discovered")
            else 0,
        },
    }
    (spec.run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def _run_summarize(stage_dir: Path, prefer: str) -> Path:
    out_csv = stage_dir / "summary.csv"
    cmd = [sys.executable, str(Path("scripts") / "summarize_sweep_v2.py"), str(stage_dir), "--prefer", str(prefer)]
    subprocess.run(cmd, check=True)
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected summary CSV not found: {out_csv}")
    return out_csv


def _pick_best_run_from_csv(summary_csv: Path) -> Path:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in summary: {summary_csv}")
    run_dir = rows[0].get("run_dir", "").strip()
    if not run_dir:
        raise RuntimeError(f"Missing run_dir in summary: {summary_csv}")
    return Path(run_dir)


_CHECKPOINT_RE = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$")


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


def _stage_state_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_state.json"


def _load_stage_state(stage_dir: Path) -> dict:
    p = _stage_state_path(stage_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_stage_state(stage_dir: Path, state: dict) -> None:
    p = _stage_state_path(stage_dir)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _run_stage(
    args: argparse.Namespace,
    *,
    root_dir: Path,
    stage_index: int,
    run_specs: List[RunSpec],
    max_parallel: int,
    stage_hours: float,
) -> None:
    stage_dir = root_dir / f"stage_{stage_index:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # We run the stage in "waves" when runs > max_parallel. Each wave runs for stage_hours.
    stage_seconds = float(stage_hours) * 3600.0
    if stage_seconds < 0:
        stage_seconds = 0.0

    state = _load_stage_state(stage_dir)
    next_spec_index = int(state.get("next_spec_index", 0))
    wave_elapsed = float(state.get("wave_elapsed_seconds", 0.0))

    stage_started_at = float(state.get("stage_started_at", 0.0)) or time.time()
    state.setdefault("stage", int(stage_index))
    state.setdefault("stage_started_at", stage_started_at)
    state["stage_hours"] = float(stage_hours)
    state["max_parallel"] = int(max_parallel)
    _save_stage_state(stage_dir, state)

    exit_codes: dict[int, int] = {}
    exit_codes_path = stage_dir / "exit_codes.json"
    if exit_codes_path.exists():
        try:
            exit_codes.update(json.loads(exit_codes_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    def ensure_run_dir_and_metadata(spec: RunSpec) -> RunSpec:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = spec.run_dir / "run_metadata.json"
        if not meta_path.exists():
            _write_run_metadata(args, stage_dir, stage_index, spec)
        # If resuming an interrupted wave, resume from latest checkpoint within the run dir.
        latest = _find_latest_checkpoint(spec.run_dir)
        if latest is not None:
            return RunSpec(
                index=spec.index,
                seed=spec.seed,
                run_dir=spec.run_dir,
                resume=str(latest),
                promoted_path=spec.promoted_path,
            )
        return spec

    def run_wave(wave_specs: List[RunSpec], *, remaining_seconds: float) -> None:
        nonlocal exit_codes

        active: List[tuple[RunSpec, subprocess.Popen, _StdoutCapture]] = []
        deadline = time.time() + float(remaining_seconds) if float(remaining_seconds) > 0 else None

        def start_run(spec: RunSpec) -> None:
            cmd = _build_train_cmd(args, spec)
            capture = _StdoutCapture(
                cmd,
                spec.run_dir,
                gzip_stdout=bool(args.stdout_gzip),
                gzip_level=int(args.stdout_gzip_level),
            )
            popen_kwargs = capture.popen_kwargs()
            proc = subprocess.Popen(cmd, env=os.environ.copy(), **popen_kwargs)
            capture.attach_process(proc)
            active.append((spec, proc, capture))
            print(f"[stage {stage_index}] Started run {spec.index} (seed={spec.seed})")

        # Start everything in this wave
        for spec in wave_specs:
            start_run(spec)

        try:
            while active:
                if deadline is not None and time.time() >= deadline:
                    if bool(args.terminate_at_deadline):
                        print(f"[stage {stage_index}] Wave budget reached: terminating active runs...")
                        for spec, proc, _ in active:
                            try:
                                proc.terminate()
                                (spec.run_dir / "terminated_by_iterative.txt").write_text("1")
                            except Exception:
                                pass
                    break

                time.sleep(2)
                still_active: List[tuple[RunSpec, subprocess.Popen, _StdoutCapture]] = []
                for spec, proc, capture in active:
                    rc = proc.poll()
                    if rc is None:
                        still_active.append((spec, proc, capture))
                        continue
                    capture.close()
                    exit_codes[spec.index] = int(rc)
                    (spec.run_dir / "exit_code.txt").write_text(str(rc))
                    print(f"[stage {stage_index}] Finished run {spec.index} (seed={spec.seed}) rc={rc}")
                active = still_active

        except KeyboardInterrupt:
            print(f"[stage {stage_index}] KeyboardInterrupt: terminating active runs...")
            for spec, proc, capture in active:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    capture.close()
                except Exception:
                    pass
                try:
                    (spec.run_dir / "terminated_by_iterative.txt").write_text("1")
                except Exception:
                    pass
            raise
        finally:
            # Best-effort close any remaining captures
            for _, _, cap in active:
                try:
                    cap.close()
                except Exception:
                    pass

    # Execute waves
    while next_spec_index < len(run_specs):
        batch = run_specs[next_spec_index : next_spec_index + int(max_parallel)]
        batch = [ensure_run_dir_and_metadata(s) for s in batch]

        remaining = stage_seconds - float(wave_elapsed)
        if remaining <= 0:
            remaining = 0.0

        wave_started = time.time()
        try:
            run_wave(batch, remaining_seconds=remaining)
        except KeyboardInterrupt:
            # Persist partial wave progress so we can resume later.
            wave_elapsed += time.time() - wave_started
            _save_stage_state(
                stage_dir,
                {
                    **state,
                    "next_spec_index": int(next_spec_index),
                    "wave_elapsed_seconds": float(wave_elapsed),
                },
            )
            exit_codes_path.write_text(json.dumps(exit_codes, indent=2), encoding="utf-8")
            raise

        # Wave finished (by time budget). Advance to next batch.
        wave_elapsed = 0.0
        next_spec_index += len(batch)
        state["next_spec_index"] = int(next_spec_index)
        state["wave_elapsed_seconds"] = float(wave_elapsed)
        _save_stage_state(stage_dir, state)
        exit_codes_path.write_text(json.dumps(exit_codes, indent=2), encoding="utf-8")

    # Stage complete
    stage_ended_at = time.time()
    (stage_dir / "stage_metadata.json").write_text(
        json.dumps(
            {
                "stage": int(stage_index),
                "started_at": datetime.fromtimestamp(stage_started_at).isoformat(timespec="seconds"),
                "ended_at": datetime.fromtimestamp(stage_ended_at).isoformat(timespec="seconds"),
                "elapsed_seconds": stage_ended_at - stage_started_at,
                "stage_hours": float(stage_hours),
                "runs_requested": len(run_specs),
                "max_parallel": int(max_parallel),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # stage_state.json indicates resumable in-progress work; remove on completion.
    try:
        _stage_state_path(stage_dir).unlink()
    except Exception:
        pass


def _effective_stage_hours(args: argparse.Namespace, stage_index: int) -> float:
    base = float(args.stage_hours)
    if stage_index == 0 and float(args.stage_hours_0) > 0:
        return float(args.stage_hours_0)
    if stage_index > 0 and float(args.stage_hours_rest) > 0:
        return float(args.stage_hours_rest)
    return base


def main() -> int:
    args = _parse_args()

    if int(args.stages) <= 0:
        print("--stages must be > 0")
        return 2
    if float(args.stage_hours) < 0:
        print("--stage-hours must be >= 0")
        return 2
    if float(args.stage_hours_0) < 0:
        print("--stage-hours-0 must be >= 0")
        return 2
    if float(args.stage_hours_rest) < 0:
        print("--stage-hours-rest must be >= 0")
        return 2
    if int(args.initial_runs) <= 0:
        print("--initial-runs must be > 0")
        return 2
    if int(args.initial_max_parallel) <= 0:
        print("--initial-max-parallel must be > 0")
        return 2
    if int(args.continue_runs) <= 0:
        print("--continue-runs must be > 0")
        return 2
    if int(args.continue_max_parallel) <= 0:
        print("--continue-max-parallel must be > 0")
        return 2

    try:
        root_dir = _make_or_use_root_dir(args)
    except FileExistsError as e:
        print(str(e))
        return 2
    print(f"Iterative root dir: {root_dir}")

    resume_checkpoint: str = ""
    winner_seed: Optional[int] = None
    promoted_path: str = ""
    rank_mode = _choose_rank_mode(str(args.mode))

    # Resume-root: derive next stage and resume inputs from last completed stage.
    if bool(args.resume_root):
        completed = []
        for child in sorted(root_dir.glob("stage_*")):
            if not child.is_dir():
                continue
            if (child / "summary.csv").exists() and (child / "best_checkpoint.txt").exists():
                completed.append(child)
        if completed:
            last = completed[-1]
            ckpt = (last / "best_checkpoint.txt").read_text(encoding="utf-8").strip()
            resume_checkpoint = ckpt
            # promoted events are stage-local; if present, use for next stage
            pep = last / "promoted_events_path.txt"
            if pep.exists():
                promoted_path = pep.read_text(encoding="utf-8").strip()
            try:
                best_run_dir = Path((last / "best_run_dir.txt").read_text(encoding="utf-8").strip())
                meta = json.loads((best_run_dir / "run_metadata.json").read_text(encoding="utf-8"))
                winner_seed = int(meta.get("seed")) if meta.get("seed") is not None else None
            except Exception:
                pass

            # set starting stage index
            try:
                start_stage = int(last.name.split("_", 1)[1]) + 1
            except Exception:
                start_stage = 0
        else:
            start_stage = 0
    else:
        start_stage = 0

    _save_resume_state(
        root_dir,
        {
            "mode": str(args.mode),
            "rank_mode": str(rank_mode),
            "resume_checkpoint": str(resume_checkpoint),
            "winner_seed": winner_seed,
            "promoted_path": str(promoted_path),
            "start_stage": int(start_stage),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    for stage_index in range(int(start_stage), int(args.stages)):
        if stage_index == 0:
            seeds = _parse_seeds(args.seeds, int(args.seed_start), int(args.initial_runs))
            if len(seeds) != int(args.initial_runs):
                print(f"Expected {args.initial_runs} seeds, got {len(seeds)}")
                return 2

            stage_dir = root_dir / f"stage_{stage_index:02d}"
            run_specs = [
                RunSpec(
                    index=i,
                    seed=int(seed),
                    run_dir=stage_dir / f"run_{i:03d}_seed{int(seed)}",
                    resume="",
                    promoted_path="",
                )
                for i, seed in enumerate(seeds)
            ]
            max_parallel = int(args.initial_max_parallel)
        else:
            if not resume_checkpoint:
                print(f"No resume checkpoint available after stage {stage_index - 1}; stopping.")
                break
            seed = int(winner_seed) if winner_seed is not None else int(args.seed_start)
            stage_dir = root_dir / f"stage_{stage_index:02d}"
            run_specs = [
                RunSpec(
                    index=i,
                    seed=seed,
                    run_dir=stage_dir / f"run_{i:03d}_seed{seed}",
                    resume=resume_checkpoint,
                    promoted_path=promoted_path if str(args.mode).startswith("discovered") else "",
                )
                for i in range(int(args.continue_runs))
            ]
            max_parallel = int(args.continue_max_parallel)

        stage_hours = _effective_stage_hours(args, int(stage_index))
        _run_stage(
            args,
            root_dir=root_dir,
            stage_index=stage_index,
            run_specs=run_specs,
            max_parallel=max_parallel,
            stage_hours=float(stage_hours),
        )

        # Rank and pick winner
        stage_dir = root_dir / f"stage_{stage_index:02d}"
        try:
            summary_csv = _run_summarize(stage_dir, prefer=str(args.prefer))
        except subprocess.CalledProcessError:
            print(
                "Failed to summarize stage (missing tensorboard?). Install with: pip install tensorboard"
            )
            return 2

        best_run_dir = _pick_best_run_from_csv_with_rank_mode(summary_csv, rank_mode=rank_mode)
        best_ckpt = _find_latest_checkpoint(best_run_dir)

        (stage_dir / "best_run_dir.txt").write_text(str(best_run_dir), encoding="utf-8")
        (stage_dir / "best_checkpoint.txt").write_text(str(best_ckpt) if best_ckpt else "", encoding="utf-8")

        # Read seed from run_metadata.json if possible
        try:
            meta = json.loads((best_run_dir / "run_metadata.json").read_text())
            winner_seed = int(meta.get("seed"))
        except Exception:
            winner_seed = winner_seed

        if best_ckpt is None:
            print(f"[stage {stage_index}] No checkpoint found in best run dir: {best_run_dir}")
            resume_checkpoint = ""
        else:
            resume_checkpoint = str(best_ckpt)
            print(f"[stage {stage_index}] Best checkpoint: {best_ckpt}")

        _save_resume_state(
            root_dir,
            {
                "mode": str(args.mode),
                "rank_mode": str(rank_mode),
                "resume_checkpoint": str(resume_checkpoint),
                "winner_seed": winner_seed,
                "promoted_path": str(promoted_path),
                "last_completed_stage": int(stage_index),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        # Promotion step for discovered_* modes (write stage-local promoted JSON for next stage)
        if str(args.mode).startswith("discovered") and stage_index < (int(args.stages) - 1):
            try:
                promoted = _run_promotion(
                    stage_dir,
                    rank_mode=rank_mode,
                    top_k=int(args.promote_top_k),
                    max_events=int(args.promote_max_events),
                    min_total_runs=int(args.promote_min_total_runs),
                )
                promoted_path = str(promoted)
                (stage_dir / "promoted_events_path.txt").write_text(promoted_path, encoding="utf-8")
                print(f"[stage {stage_index}] Promoted discovered events: {promoted}")
            except subprocess.CalledProcessError:
                print(f"[stage {stage_index}] Promotion failed; continuing without promoted shaping")
                promoted_path = ""

    print("Iterative training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
