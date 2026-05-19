#!/usr/bin/env python3
"""Run four training configs for comparison:
 1) base (no discovered events)
 2) explicit events (use v2/events.json as promoted list)
 3) dynamic discovered events
 4) dynamic discovered events but do not include promoted events in ranking (empty promoted path)

This is a convenience smoke runner that invokes v2/baseline_fast_v2.py with small timesteps.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def make_cmd(run_dir: Path, extra_args: dict, common: dict) -> list:
    script = Path("v2") / "baseline_fast_v2.py"
    cmd = [sys.executable, str(script), "--run-dir", str(run_dir)]
    for k, v in common.items():
        if isinstance(v, bool):
            cmd.append(("--" + k) if v else ("--no-" + k))
        else:
            cmd.extend(["--" + k.replace("_", "-"), str(v)])
    for k, v in extra_args.items():
        if isinstance(v, bool):
            cmd.append(("--" + k) if v else ("--no-" + k))
        else:
            cmd.extend(["--" + k.replace("_", "-"), str(v)])
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="sweeps_four_configs")
    parser.add_argument("--ep-length", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=2000)
    parser.add_argument("--num-cpu", type=int, default=1)
    parser.add_argument("--gb-path", type=str, default="PokemonRed.gb")
    parser.add_argument("--init-state", type=str, default="init.state")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = out_base / f"four_configs_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    common = {
        "num-cpu": args.num_cpu,
        "ep-length": args.ep_length,
        "total-timesteps": args.total_timesteps,
        "gb-path": args.gb_path,
        "init-state": args.init_state,
        "headless": args.headless,
    }

    runs = [
        ("base_no_events", {}),
        ("explicit_events_promoted", {"discovered-events": False, "discovered-events-promoted-path": "v2/events.json", "discovered-events-reward-weight": 0.5}),
        ("dynamic_discovered_events", {"discovered-events": True, "discovered-events-reward-weight": 0.5}),
        ("dynamic_no_promoted_in_rank", {"discovered-events": True, "discovered-events-promoted-path": "", "discovered-events-reward-weight": 0.5}),
    ]

    for name, extra in runs:
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = make_cmd(run_dir, extra, common)
        print("Running:", " ".join(cmd))
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as f:
            f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
            f.flush()
            try:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(**os_environ()))
                rc = proc.wait()
            except Exception as e:
                f.write(f"ERROR: {e}\n")
                rc = 1
        print(f"Run {name} finished rc={rc} -> {run_dir}")


def os_environ():
    # Minimal copy of environment to avoid sharing mutable mapping
    import os

    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
