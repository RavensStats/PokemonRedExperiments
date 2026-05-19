import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(
        description=(
            "Run the three iterative modes (explicit vs discovered_explicit vs discovered_no_explicit) "
            "as separate processes, optionally in parallel."
        )
    )
    p.add_argument(
        "--max-parallel-modes",
        type=int,
        default=3,
        help="Max concurrent modes (1..3)",
    )
    p.add_argument(
        "--base-tag",
        type=str,
        default="compare",
        help="Tag prefix added to each run's --tag",
    )
    p.add_argument(
        "--root-base-dir",
        type=str,
        default="sweeps",
        help="Base directory where each mode gets a stable root folder (enables resume)",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If a mode's root dir exists, resume it instead of failing",
    )
    # Everything else is forwarded to scripts/run_iterative_sweep_v2.py
    args, rest = p.parse_known_args()
    return args, rest


def _cmd_for(mode: str, base_tag: str, rest: List[str]) -> List[str]:
    iterative = Path("scripts") / "run_iterative_sweep_v2.py"
    tag = f"{base_tag}_{mode}"
    return [sys.executable, str(iterative), "--mode", mode, "--tag", tag, *rest]


def main() -> int:
    args, rest = _parse_args()
    max_parallel = max(1, min(int(args.max_parallel_modes), 3))

    modes = ["explicit", "discovered_explicit", "discovered_no_explicit"]
    base = Path(str(args.root_base_dir))
    # Stable per-mode root dirs for pause/resume
    queue = []
    for m in modes:
        root_dir = base / f"iterative_{str(args.base_tag)}_{m}"
        cmd = _cmd_for(m, str(args.base_tag), rest)
        cmd.extend(["--root-dir", str(root_dir)])
        if bool(args.resume):
            cmd.append("--resume-root")
        queue.append((m, cmd))

    running: List[Tuple[str, subprocess.Popen]] = []
    completed: List[Tuple[str, int]] = []

    env = os.environ.copy()

    def start_next() -> None:
        nonlocal queue
        if not queue:
            return
        mode, cmd = queue.pop(0)
        print(f"[compare] starting {mode}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env)
        running.append((mode, proc))

    while queue or running:
        while queue and len(running) < max_parallel:
            start_next()

        time.sleep(2)
        still: List[Tuple[str, subprocess.Popen]] = []
        for mode, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((mode, proc))
            else:
                completed.append((mode, int(rc)))
                print(f"[compare] finished {mode} rc={rc}")
        running = still

    worst = max((rc for _, rc in completed), default=0)
    if worst != 0:
        print("[compare] one or more modes failed")
        for mode, rc in completed:
            print(f"  - {mode}: rc={rc}")
        return 2

    print("[compare] all modes complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
