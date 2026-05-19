import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Series:
    steps: List[int]
    values: List[float]


@dataclass(frozen=True)
class RunCurves:
    run_dir: Path
    seed: Optional[int]
    exit_code: Optional[int]
    curves: Dict[str, Series]


METRIC_TO_TAG = {
    "badge_count": "badge",
    "events_completed": "events_completed",
    "map_explored_pct": "explore_pct",
    "highest_pokemon_level": "max_level",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and compare V2 sweep efficiency curves")
    parser.add_argument("sweep_dir", type=str, help="Sweep folder containing run_* subfolders")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Directory to write plots/CSVs (default: <sweep_dir>/plots)",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Use env_stats_max/* or env_stats/* tags",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="badge_count,events_completed,map_explored_pct,highest_pokemon_level",
        help="Comma-separated metrics to plot",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Plot only the top K runs (ranked by --rank-by final value). Use 0 for all.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="badge_count",
        choices=list(METRIC_TO_TAG.keys()),
        help="Metric used to rank runs for plotting",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="",
        help="Optional thresholds like 'badge_count=1,events_completed=10' to compute time-to-threshold.",
    )

    parser.add_argument(
        "--goals",
        type=str,
        default="",
        help=(
            "Absolute goal thresholds per metric, e.g. 'badge_count=1|2|3,events_completed=10|25'. "
            "Use '|' to specify multiple thresholds for the same metric."
        ),
    )
    parser.add_argument(
        "--relative-goals",
        type=str,
        default="0.25|0.5|0.75",
        help=(
            "Relative-to-final goal fractions, e.g. '0.25|0.5|0.75'. "
            "Computes timesteps to reach that fraction of the run's final value for each metric. "
            "Set to empty string to disable."
        ),
    )
    return parser.parse_args()


def _find_run_dirs(sweep_dir: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for child in sorted(sweep_dir.iterdir()):
        if child.is_dir() and (child / "run_metadata.json").exists():
            run_dirs.append(child)
    return run_dirs


def _read_seed(run_dir: Path) -> Optional[int]:
    try:
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        return int(meta.get("seed"))
    except Exception:
        return None


def _read_exit_code(run_dir: Path) -> Optional[int]:
    try:
        return int((run_dir / "exit_code.txt").read_text().strip())
    except Exception:
        return None


def _find_event_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.rglob("events.out.tfevents.*"))


def _load_scalars(event_files: List[Path]) -> Dict[str, List[Tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as e:
        raise RuntimeError(
            "tensorboard is required for plotting. Install with: pip install tensorboard"
        ) from e

    merged: Dict[str, List[Tuple[int, float]]] = {}
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

    for tag in merged:
        merged[tag].sort(key=lambda x: x[0])

    return merged


def _to_series(points: List[Tuple[int, float]]) -> Series:
    if not points:
        return Series([], [])
    steps = [s for s, _ in points]
    values = [v for _, v in points]
    return Series(steps, values)


def _last_value(series: Series) -> Optional[float]:
    if not series.values:
        return None
    return float(series.values[-1])


def _auc(series: Series) -> Optional[float]:
    # trapezoidal area under curve vs step
    if len(series.steps) < 2:
        return None
    area = 0.0
    for i in range(1, len(series.steps)):
        x0, x1 = series.steps[i - 1], series.steps[i]
        y0, y1 = series.values[i - 1], series.values[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def _step_to_threshold(series: Series, threshold: float) -> Optional[int]:
    for s, v in zip(series.steps, series.values):
        if v >= threshold:
            return int(s)
    return None


def _parse_thresholds(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not spec.strip():
        return out
    for item in spec.split(","):
        if not item.strip() or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in METRIC_TO_TAG:
            out[k] = float(v)
    return out


def _parse_goal_thresholds(spec: str) -> Dict[str, List[float]]:
    """Parse 'metric=1|2|3,other=10|25' into {metric: [1.0,2.0,3.0], ...}."""
    goals: Dict[str, List[float]] = {}
    if not spec or not spec.strip():
        return goals
    for item in spec.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in METRIC_TO_TAG:
            continue
        vals = []
        for part in v.split("|"):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        if vals:
            goals[k] = sorted(set(vals))
    return goals


def _parse_relative_goals(spec: str) -> List[float]:
    if not spec or not spec.strip():
        return []
    vals: List[float] = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    # keep unique, sorted
    vals = sorted(set(vals))
    # sanity clamp
    return [v for v in vals if 0.0 < v <= 1.0]


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _load_run_curves(run_dir: Path, prefer: str, metrics: List[str]) -> RunCurves:
    event_files = _find_event_files(run_dir)
    scalars = _load_scalars(event_files) if event_files else {}

    prefix = "env_stats_max" if prefer == "max" else "env_stats"
    curves: Dict[str, Series] = {}
    for metric in metrics:
        tag = f"{prefix}/{METRIC_TO_TAG[metric]}"
        curves[metric] = _to_series(scalars.get(tag, []))

    return RunCurves(
        run_dir=run_dir,
        seed=_read_seed(run_dir),
        exit_code=_read_exit_code(run_dir),
        curves=curves,
    )


def main() -> int:
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (sweep_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in METRIC_TO_TAG:
            print(f"Unknown metric: {m}. Valid: {list(METRIC_TO_TAG.keys())}")
            return 2

    run_dirs = _find_run_dirs(sweep_dir)
    if not run_dirs:
        print(f"No run directories found under: {sweep_dir}")
        return 2

    try:
        runs = [_load_run_curves(rd, prefer=args.prefer, metrics=metrics) for rd in run_dirs]
    except RuntimeError as e:
        print(str(e))
        return 2

    # Rank runs by final value of rank-by metric
    rank_metric = args.rank_by

    def rank_key(r: RunCurves):
        v = _last_value(r.curves.get(rank_metric, Series([], [])))
        return float(v) if v is not None else float("-inf")

    runs.sort(key=rank_key, reverse=True)
    if int(args.top_k) > 0:
        runs_to_plot = runs[: int(args.top_k)]
    else:
        runs_to_plot = runs

    thresholds = _parse_thresholds(args.threshold)
    absolute_goals = _parse_goal_thresholds(args.goals)
    relative_goals = _parse_relative_goals(args.relative_goals)

    # Excel-friendly sentinel for "goal not reached" so sorting works.
    missing_goal_sentinel = 10**18

    # Version 1: AUC-based efficiency report
    eff_auc_path = out_dir / "efficiency_auc.csv"
    with eff_auc_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_dir", "seed", "exit_code"]
        for m in metrics:
            fieldnames += [f"final_{m}", f"auc_{m}"]
            if m in thresholds:
                fieldnames.append(f"step_to_{m}_ge_{thresholds[m]}")

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in runs:
            row = {"run_dir": str(r.run_dir), "seed": r.seed, "exit_code": r.exit_code}
            for m in metrics:
                s = r.curves[m]
                row[f"final_{m}"] = _last_value(s)
                row[f"auc_{m}"] = _auc(s)
                if m in thresholds:
                    row[f"step_to_{m}_ge_{thresholds[m]}"] = _step_to_threshold(s, thresholds[m])
            writer.writerow(row)

    # Version 2: Badge-time efficiency report (timesteps to 1st/2nd/3rd badge)
    eff_badge_path = out_dir / "efficiency_badge_times.csv"
    with eff_badge_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_dir",
            "seed",
            "exit_code",
            "final_badge_count",
            "timesteps_to_badge_1",
            "timesteps_to_badge_2",
            "timesteps_to_badge_3",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in runs:
            badge_series = r.curves.get("badge_count")
            if badge_series is None:
                badge_series = Series([], [])
            row = {
                "run_dir": str(r.run_dir),
                "seed": r.seed,
                "exit_code": r.exit_code,
                "final_badge_count": _last_value(badge_series),
                "timesteps_to_badge_1": _step_to_threshold(badge_series, 1.0),
                "timesteps_to_badge_2": _step_to_threshold(badge_series, 2.0),
                "timesteps_to_badge_3": _step_to_threshold(badge_series, 3.0),
            }
            writer.writerow(row)

    # Version 3: General time-to-goal report for each metric
    eff_goals_path = out_dir / "efficiency_goal_times.csv"
    goal_rows: List[Dict[str, object]] = []
    goal_columns: List[str] = []
    with eff_goals_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_dir", "seed", "exit_code"]

        # Absolute goals
        for m in metrics:
            for g in absolute_goals.get(m, []):
                col = f"timesteps_to_{m}_ge_{g}"
                fieldnames.append(col)
                goal_columns.append(col)

        # Relative goals (fraction of final)
        for m in metrics:
            for frac in relative_goals:
                pct = int(round(frac * 100))
                col = f"timesteps_to_{m}_ge_{pct}pct_final"
                fieldnames.append(col)
                goal_columns.append(col)

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in runs:
            row: Dict[str, object] = {"run_dir": str(r.run_dir), "seed": r.seed, "exit_code": r.exit_code}

            # Absolute goals
            for m in metrics:
                series = r.curves[m]
                for g in absolute_goals.get(m, []):
                    col = f"timesteps_to_{m}_ge_{g}"
                    v = _step_to_threshold(series, g)
                    row[col] = int(v) if v is not None else missing_goal_sentinel

            # Relative goals
            for m in metrics:
                series = r.curves[m]
                final_val = _last_value(series)
                for frac in relative_goals:
                    pct = int(round(frac * 100))
                    col = f"timesteps_to_{m}_ge_{pct}pct_final"
                    if final_val is None:
                        row[col] = missing_goal_sentinel
                    else:
                        v = _step_to_threshold(series, final_val * frac)
                        row[col] = int(v) if v is not None else missing_goal_sentinel

            goal_rows.append(row)

        # Sort by goal columns ascending: fastest to reach goals first.
        def goal_sort_key(r: Dict[str, object]):
            key: List[int] = []
            for col in goal_columns:
                v = r.get(col, missing_goal_sentinel)
                try:
                    key.append(int(v))
                except Exception:
                    key.append(missing_goal_sentinel)
            # Tie-breakers
            seed = r.get("seed")
            key.append(int(seed) if seed is not None else missing_goal_sentinel)
            return tuple(key)

        goal_rows.sort(key=goal_sort_key)
        for row in goal_rows:
            writer.writerow(row)

    # Plot curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required for plots. Install with: pip install matplotlib")
        print(f"Wrote AUC efficiency CSV: {eff_auc_path}")
        print(f"Wrote badge-time efficiency CSV: {eff_badge_path}")
        print(f"Wrote goal-time efficiency CSV: {eff_goals_path}")
        return 0

    for m in metrics:
        plt.figure(figsize=(10, 6))
        for r in runs_to_plot:
            s = r.curves[m]
            if not s.steps:
                continue
            label = r.run_dir.name
            if r.seed is not None:
                label += f" (seed={r.seed})"
            plt.plot(s.steps, s.values, linewidth=1.5, label=label)

        plt.title(f"{m} vs timesteps ({args.prefer})")
        plt.xlabel("timesteps")
        plt.ylabel(m)

        # Goal lines
        if m in thresholds:
            plt.axhline(y=thresholds[m], color="gray", linestyle="--", linewidth=1)
        for g in absolute_goals.get(m, []):
            plt.axhline(y=g, color="lightgray", linestyle=":", linewidth=1)
        if len(runs_to_plot) <= 12:
            plt.legend(fontsize=8)
        plt.tight_layout()
        out_path = out_dir / f"curve_{m}.png"
        plt.savefig(out_path)
        plt.close()

    # Goal-time graphs
    # 1) Per-goal plot (sorted runs on x, timesteps on y)
    if goal_rows and goal_columns:
        import numpy as np

        # limit labels to keep plots readable
        max_labels = 30
        run_labels = [Path(str(r["run_dir"])) .name for r in goal_rows]

        for col in goal_columns:
            vals: List[float] = []
            for r in goal_rows:
                v = r.get(col, missing_goal_sentinel)
                try:
                    vv = float(v)
                except Exception:
                    vv = float(missing_goal_sentinel)
                if int(vv) >= missing_goal_sentinel:
                    vals.append(float("nan"))
                else:
                    vals.append(vv)

            x = np.arange(len(vals))
            plt.figure(figsize=(12, 5))
            plt.plot(x, vals, marker="o", linewidth=1)
            plt.title(col)
            plt.xlabel("run (sorted by fastest-to-goals)")
            plt.ylabel("timesteps")

            if len(run_labels) <= max_labels:
                plt.xticks(x, run_labels, rotation=90, fontsize=7)
            else:
                plt.xticks([])

            plt.tight_layout()
            out_path = out_dir / f"goal_{_safe_filename(col)}.png"
            plt.savefig(out_path)
            plt.close()

        # 2) Heatmap overview (runs x goals)
        mat = []
        for r in goal_rows:
            row_vals: List[float] = []
            for col in goal_columns:
                v = r.get(col, missing_goal_sentinel)
                try:
                    vv = float(v)
                except Exception:
                    vv = float(missing_goal_sentinel)
                if int(vv) >= missing_goal_sentinel:
                    row_vals.append(float("nan"))
                else:
                    row_vals.append(vv)
            mat.append(row_vals)

        mat_np = np.array(mat, dtype=float)
        plt.figure(figsize=(max(10, len(goal_columns) * 0.6), max(6, len(goal_rows) * 0.25)))
        plt.imshow(mat_np, aspect="auto", interpolation="nearest")
        plt.colorbar(label="timesteps")
        plt.title("Goal time heatmap (NaN = goal not reached)")
        plt.xlabel("goal")
        plt.ylabel("run (sorted)")
        plt.xticks(np.arange(len(goal_columns)), goal_columns, rotation=90, fontsize=7)
        if len(run_labels) <= max_labels:
            plt.yticks(np.arange(len(run_labels)), run_labels, fontsize=7)
        else:
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(out_dir / "goal_times_heatmap.png")
        plt.close()

    print(f"Wrote plots to: {out_dir}")
    print(f"Wrote AUC efficiency CSV: {eff_auc_path}")
    print(f"Wrote badge-time efficiency CSV: {eff_badge_path}")
    print(f"Wrote goal-time efficiency CSV: {eff_goals_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
