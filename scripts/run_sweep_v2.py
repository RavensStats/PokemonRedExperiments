import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import gzip


@dataclass(frozen=True)
class RunSpec:
    index: int
    seed: int
    run_dir: Path


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

        # Write a tiny pointer file for convenience.
        pointer_path = self._run_dir / "stdout.log"
        gz_path = self._run_dir / "stdout.log.gz"
        pointer_path.write_text(
            "COMMAND:\n"
            + " ".join(self._cmd)
            + "\n\n"
            + "NOTE: Full stdout/stderr is stored in stdout.log.gz (gzip, lossless).\n",
            encoding="utf-8",
        )

        # We'll capture bytes from the subprocess and stream them into gzip.
        self._gz_fp = gzip.open(gz_path, mode="wb", compresslevel=self._gzip_level)
        header = ("COMMAND:\n" + " ".join(self._cmd) + "\n\n").encode("utf-8")
        self._gz_fp.write(header)
        self._gz_fp.flush()

        # Use a pipe so we can write into gzip ourselves.
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
    parser = argparse.ArgumentParser(description="Run many V2 training runs (overnight sweep)")

    parser.add_argument("--runs", type=int, default=5, help="Number of runs to execute")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds (overrides --seed-start)",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (used if --seeds not provided)")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="sweeps",
        help="Base output directory (a timestamped subfolder is created)",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag included in the sweep folder name")

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="How many independent training processes to run at once (default: 1, safest on Windows)",
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="Wall-clock time budget. When reached, stops launching new runs (0 disables).",
    )
    parser.add_argument(
        "--terminate-at-deadline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, terminates active runs when the time budget is reached.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep going if a run fails",
    )

    parser.add_argument(
        "--stdout-gzip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), write full process output to stdout.log.gz (lossless) and keep a tiny stdout.log pointer.",
    )
    parser.add_argument(
        "--stdout-gzip-level",
        type=int,
        default=6,
        help="Gzip compression level for stdout.log.gz (1=fastest, 9=smallest).",
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

    # Optional post-sweep evaluation: input jitter robustness on best checkpoint
    parser.add_argument(
        "--eval-input-jitter-after",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After sweep completes successfully, evaluate best checkpoint with input jitter enabled.",
    )
    parser.add_argument("--eval-input-jitter-prefer", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--eval-input-jitter-episodes", type=int, default=1)
    parser.add_argument("--eval-input-jitter-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-input-jitter-prob", type=float, default=0.1)
    parser.add_argument(
        "--eval-input-jitter-mode",
        type=str,
        default="lag",
        choices=["lag", "sticky", "repeat", "drift", "direction", "random", "rand"],
    )

    # Optional post-sweep evaluation: perception noise (sensor error) on best checkpoint
    parser.add_argument(
        "--eval-perception-noise-after",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After sweep completes successfully, evaluate best checkpoint under perception noise.",
    )
    parser.add_argument("--eval-perception-noise-prefer", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--eval-perception-noise-episodes", type=int, default=1)
    parser.add_argument("--eval-perception-noise-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-perception-noise-radius", type=int, default=0)
    parser.add_argument(
        "--eval-perception-noise-radii",
        type=str,
        default="",
        help="Comma-separated radii to sweep (e.g. 0,1,2,4,8). If empty, uses --eval-perception-noise-radius.",
    )
    parser.add_argument("--eval-perception-noise-mode", type=str, default="uniform", choices=["uniform", "normal"])

    return parser.parse_args()


def _parse_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds.strip():
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed_start + i for i in range(int(args.runs))]


def _make_sweep_dir(base_out_dir: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"sweep_{timestamp}"
    if tag:
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)
        name += f"_{safe_tag}"
    sweep_dir = Path(base_out_dir) / name
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def _build_train_cmd(args: argparse.Namespace, spec: RunSpec) -> List[str]:
    train_script = Path("v2") / "baseline_fast_v2.py"
    return [
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
        ("--headless" if args.headless else "--no-headless"),
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
        ("--stream" if args.stream else "--no-stream"),
        "--stream-user",
        str(args.stream_user),
        "--stream-color",
        str(args.stream_color),
        "--stream-extra",
        str(args.stream_extra),
        ("--wandb" if args.wandb else "--no-wandb"),
    ]


def _write_run_metadata(args: argparse.Namespace, spec: RunSpec) -> None:
    meta = {
        "index": spec.index,
        "seed": spec.seed,
        "run_dir": str(spec.run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "train_args": {
            "num_cpu": args.num_cpu,
            "ep_length": args.ep_length,
            "total_timesteps": args.total_timesteps,
            "train_steps_batch": args.train_steps_batch,
            "action_freq": args.action_freq,
            "init_state": args.init_state,
            "gb_path": args.gb_path,
            "headless": args.headless,
            "reward_scale": args.reward_scale,
            "explore_weight": args.explore_weight,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            "device": args.device,
            "stream": args.stream,
            "stream_user": args.stream_user,
            "stream_color": args.stream_color,
            "stream_extra": args.stream_extra,
            "wandb": args.wandb,
        },
    }
    (spec.run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def _launch_one(cmd: List[str], run_dir: Path) -> int:
    stdout_path = run_dir / "stdout.log"
    with stdout_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        return proc.wait()


def main() -> int:
    args = _parse_args()

    if int(args.runs) <= 0:
        print("--runs must be > 0")
        return 2
    if int(args.max_parallel) <= 0:
        print("--max-parallel must be > 0")
        return 2

    seeds = _parse_seeds(args)
    if len(seeds) != int(args.runs):
        print(f"Expected {args.runs} seeds, got {len(seeds)}")
        return 2

    sweep_dir = _make_sweep_dir(args.out_dir, args.tag)
    print(f"Sweep dir: {sweep_dir}")

    started_at = time.time()
    deadline: Optional[float] = None
    if float(args.hours) > 0:
        deadline = started_at + float(args.hours) * 3600.0

    specs: List[RunSpec] = []
    for i, seed in enumerate(seeds):
        run_dir = sweep_dir / f"run_{i:03d}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        specs.append(RunSpec(index=i, seed=seed, run_dir=run_dir))

    (sweep_dir / "sweep_plan.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "hours_budget": float(args.hours),
                "runs": [
                    {"index": s.index, "seed": s.seed, "run_dir": str(s.run_dir)}
                    for s in specs
                ],
            },
            indent=2,
        )
    )

    (sweep_dir / "sweep_metadata.json").write_text(
        json.dumps(
            {
                "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
                "deadline": datetime.fromtimestamp(deadline).isoformat(timespec="seconds") if deadline else None,
                "terminate_at_deadline": bool(args.terminate_at_deadline),
                "max_parallel": int(args.max_parallel),
            },
            indent=2,
        )
    )

    # Default: sequential execution (max-parallel=1). We keep the implementation simple
    # and robust for overnight runs on Windows.
    active: List[tuple[RunSpec, subprocess.Popen, _StdoutCapture]] = []
    next_index = 0
    exit_codes: dict[int, int] = {}

    def start_run(spec: RunSpec) -> None:
        _write_run_metadata(args, spec)
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
        print(f"Started run {spec.index} (seed={spec.seed})")

    try:
        while next_index < len(specs) or active:
            now = time.time()
            if deadline is not None and now >= deadline:
                if bool(args.terminate_at_deadline) and active:
                    print("Deadline reached: terminating active runs...")
                    for spec, proc in active:
                        try:
                            proc.terminate()
                            (spec.run_dir / "terminated_by_sweep.txt").write_text("1")
                        except Exception:
                            pass
                    # Don't launch any more; continue loop to let processes exit/record codes.
                # Stop launching new runs after deadline.
                next_index = len(specs)

            while next_index < len(specs) and len(active) < int(args.max_parallel):
                if deadline is not None and time.time() >= deadline:
                    break
                start_run(specs[next_index])
                next_index += 1

            # Wait for any active process to finish (simple polling).
            # Avoid tight loop.
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
                print(f"Finished run {spec.index} (seed={spec.seed}) rc={rc}")

                if rc != 0 and not bool(args.continue_on_error):
                    print("Run failed and --no-continue-on-error set; stopping sweep")
                    return 1

            active = still_active

    except KeyboardInterrupt:
        print("KeyboardInterrupt: terminating active runs...")
        for _, proc, capture in active:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                capture.close()
            except Exception:
                pass
        return 130

    (sweep_dir / "exit_codes.json").write_text(json.dumps(exit_codes, indent=2))

    ended_at = time.time()
    (sweep_dir / "sweep_metadata.json").write_text(
        json.dumps(
            {
                "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
                "ended_at": datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
                "elapsed_seconds": ended_at - started_at,
                "deadline": datetime.fromtimestamp(deadline).isoformat(timespec="seconds") if deadline else None,
                "hours_budget": float(args.hours),
                "terminate_at_deadline": bool(args.terminate_at_deadline),
                "max_parallel": int(args.max_parallel),
                "runs_requested": int(args.runs),
                "runs_started": len(exit_codes) + len(active),
                "runs_completed": len(exit_codes),
            },
            indent=2,
        )
    )
    failures = {k: v for k, v in exit_codes.items() if v != 0}
    if failures:
        print(f"Sweep complete with failures: {failures}")
        return 1

    print("Sweep complete")

    if bool(args.eval_input_jitter_after):
        eval_script = Path("scripts") / "eval_input_jitter_best_v2.py"
        cmd = [
            sys.executable,
            str(eval_script),
            str(sweep_dir),
            "--prefer",
            str(args.eval_input_jitter_prefer),
            "--episodes",
            str(int(args.eval_input_jitter_episodes)),
            ("--deterministic" if bool(args.eval_input_jitter_deterministic) else "--no-deterministic"),
            "--jitter-prob",
            str(float(args.eval_input_jitter_prob)),
            "--jitter-mode",
            str(args.eval_input_jitter_mode),
        ]
        print(f"[sweep] running jitter eval: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[sweep] jitter eval failed rc={e.returncode}")
            return 1

    if bool(args.eval_perception_noise_after):
        eval_script = Path("scripts") / "eval_perception_noise_best_v2.py"
        cmd = [
            sys.executable,
            str(eval_script),
            str(sweep_dir),
            "--prefer",
            str(args.eval_perception_noise_prefer),
            "--episodes",
            str(int(args.eval_perception_noise_episodes)),
            ("--deterministic" if bool(args.eval_perception_noise_deterministic) else "--no-deterministic"),
            "--noise-mode",
            str(args.eval_perception_noise_mode),
        ]
        if str(args.eval_perception_noise_radii).strip():
            cmd.extend(["--noise-radii", str(args.eval_perception_noise_radii)])
        else:
            cmd.extend(["--noise-radius", str(int(args.eval_perception_noise_radius))])

        print(f"[sweep] running perception eval: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[sweep] perception eval failed rc={e.returncode}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
