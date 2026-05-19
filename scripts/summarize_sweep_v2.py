import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    seed: Optional[int]
    exit_code: Optional[int]
    badge_count: Optional[float]
    map_explored_pct: Optional[float]
    events_completed: Optional[float]
    highest_pokemon_level: Optional[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a V2 sweep folder into a CSV")
    parser.add_argument("sweep_dir", type=str, help="Path to sweep folder (contains run_* subfolders)")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (default: <sweep_dir>/summary.csv)",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Use env_stats_max/* or env_stats/* tags",
    )
    return parser.parse_args()


def _find_run_dirs(sweep_dir: Path) -> List[Path]:
    if not sweep_dir.exists() or not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")

    run_dirs: List[Path] = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        # heuristic: sweep runner writes run_metadata.json
        if (child / "run_metadata.json").exists():
            run_dirs.append(child)
    return run_dirs


def _safe_read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _read_seed(run_dir: Path) -> Optional[int]:
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if "seed" in meta:
                return int(meta["seed"])
        except Exception:
            pass
    # fallback: parse from folder name like run_000_seed123
    name = run_dir.name
    if "seed" in name:
        try:
            return int(name.split("seed", 1)[1])
        except Exception:
            return None
    return None


def _find_event_files(run_dir: Path) -> List[Path]:
    # We log custom stats into <run_dir>/histogram via TensorboardCallback.
    # SB3 may also create other event files; we just search broadly.
    return sorted(run_dir.rglob("events.out.tfevents.*"))


def _load_scalars(event_files: List[Path]) -> Dict[str, List[Tuple[int, float]]]:
    """Returns {tag: [(step, value), ...]} across all event files."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as e:
        raise RuntimeError(
            "tensorboard is required to summarize runs. Install with: pip install tensorboard"
        ) from e

    merged: Dict[str, List[Tuple[int, float]]] = {}
    # EventAccumulator only accepts a directory; we load per parent dir to avoid missing runs.
    # Some runs may have multiple event directories; we load each directory once.
    seen_dirs = set(p.parent for p in event_files)
    for log_dir in sorted(seen_dirs):
        ea = event_accumulator.EventAccumulator(
            str(log_dir),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
            },
        )
        try:
            ea.Reload()
        except Exception:
            continue

        for tag in ea.Tags().get("scalars", []):
            try:
                events = ea.Scalars(tag)
            except Exception:
                continue
            merged.setdefault(tag, [])
            merged[tag].extend([(int(ev.step), float(ev.value)) for ev in events])

    # sort each tag by step
    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])

    return merged


def _last_value(scalars: Dict[str, List[Tuple[int, float]]], tag: str) -> Optional[float]:
    if tag not in scalars or not scalars[tag]:
        return None
    return float(scalars[tag][-1][1])


def _summarize_run(run_dir: Path, prefer: str) -> RunSummary:
    seed = _read_seed(run_dir)
    exit_code = _safe_read_int(run_dir / "exit_code.txt")

    event_files = _find_event_files(run_dir)
    if not event_files:
        return RunSummary(
            run_dir=run_dir,
            seed=seed,
            exit_code=exit_code,
            badge_count=None,
            map_explored_pct=None,
            events_completed=None,
            highest_pokemon_level=None,
        )

    scalars = _load_scalars(event_files)

    prefix = "env_stats_max" if prefer == "max" else "env_stats"

    # Keys logged by v2/red_gym_env_v2.py via TensorboardCallback:
    badge = _last_value(scalars, f"{prefix}/badge")
    explore_pct = _last_value(scalars, f"{prefix}/explore_pct")
    events_completed = _last_value(scalars, f"{prefix}/events_completed")
    max_level = _last_value(scalars, f"{prefix}/max_level")

    return RunSummary(
        run_dir=run_dir,
        seed=seed,
        exit_code=exit_code,
        badge_count=badge,
        map_explored_pct=explore_pct,
        events_completed=events_completed,
        highest_pokemon_level=max_level,
    )


def _write_csv(out_path: Path, summaries: Iterable[RunSummary]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summaries)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "seed",
                "exit_code",
                "badge_count",
                "map_explored_pct",
                "events_completed",
                "highest_pokemon_level",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "run_dir": str(r.run_dir),
                    "seed": r.seed,
                    "exit_code": r.exit_code,
                    "badge_count": r.badge_count,
                    "map_explored_pct": r.map_explored_pct,
                    "events_completed": r.events_completed,
                    "highest_pokemon_level": r.highest_pokemon_level,
                }
            )


def main() -> int:
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out) if args.out else (sweep_dir / "summary.csv")

    run_dirs = _find_run_dirs(sweep_dir)
    if not run_dirs:
        print(f"No run directories found under: {sweep_dir}")
        return 2

    summaries: List[RunSummary] = []
    for run_dir in run_dirs:
        try:
            summaries.append(_summarize_run(run_dir, prefer=args.prefer))
        except RuntimeError as e:
            print(str(e))
            return 2

    # Sort by the requested performance metrics (descending), then by seed
    def sort_key(r: RunSummary):
        def nz(v: Optional[float]) -> float:
            return float(v) if v is not None else float("-inf")

        return (
            nz(r.badge_count),
            nz(r.events_completed),
            nz(r.map_explored_pct),
            nz(r.highest_pokemon_level),
            -(r.seed or 0),
        )

    summaries.sort(key=sort_key, reverse=True)

    _write_csv(out_path, summaries)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
