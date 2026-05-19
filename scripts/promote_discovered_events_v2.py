import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RunRow:
    run_dir: Path
    badge_count: float
    map_explored_pct: float
    events_completed: float
    highest_pokemon_level: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote discovered RAM events into a frozen shaping list for next stage")
    p.add_argument("stage_dir", type=str, help="Path to a stage folder that contains run_* subfolders")
    p.add_argument(
        "--summary",
        type=str,
        default="",
        help="Path to summary.csv (default: <stage_dir>/summary.csv)",
    )
    p.add_argument(
        "--rank-mode",
        type=str,
        default="explicit",
        choices=["explicit", "no_explicit_events"],
        help="How to rank runs when choosing 'top' runs for promotion scoring",
    )
    p.add_argument("--top-k", type=int, default=10, help="Use the top K ranked runs as positives")
    p.add_argument("--max-events", type=int, default=200, help="Maximum number of events to promote")
    p.add_argument(
        "--min-total-runs",
        type=int,
        default=1,
        help="Require event to appear in at least this many runs (prevents single-run noise)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: <stage_dir>/promoted_discovered_events.json)",
    )
    return p.parse_args()


def _safe_float(x: Optional[str]) -> float:
    try:
        if x is None:
            return float("-inf")
        s = str(x).strip()
        if not s:
            return float("-inf")
        return float(s)
    except Exception:
        return float("-inf")


def _load_summary(summary_csv: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            run_dir = Path(str(r.get("run_dir", "")).strip())
            rows.append(
                RunRow(
                    run_dir=run_dir,
                    badge_count=_safe_float(r.get("badge_count")),
                    map_explored_pct=_safe_float(r.get("map_explored_pct")),
                    events_completed=_safe_float(r.get("events_completed")),
                    highest_pokemon_level=_safe_float(r.get("highest_pokemon_level")),
                )
            )
    return rows


def _rank_key(row: RunRow, rank_mode: str) -> Tuple[float, float, float, float]:
    if rank_mode == "no_explicit_events":
        return (row.badge_count, row.map_explored_pct, row.highest_pokemon_level, float("-inf"))
    # "explicit" matches the repo's default intent: badge -> events -> explore -> max_level
    return (row.badge_count, row.events_completed, row.map_explored_pct, row.highest_pokemon_level)


def _load_discovered_events(run_dir: Path) -> List[str]:
    """Return list of discovered event ids for a run."""
    out: List[str] = []
    files = sorted(run_dir.glob("discovered_events_env*.json"))
    # Backward-compat in case a single-file format exists.
    legacy = run_dir / "discovered_events.json"
    if legacy.exists():
        files.append(legacy)

    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            events = payload.get("events", []) if isinstance(payload, dict) else []
            for e in events:
                if isinstance(e, dict) and e.get("id"):
                    out.append(str(e["id"]))
        except Exception:
            continue

    return sorted(set(out))


def main() -> int:
    args = _parse_args()

    stage_dir = Path(args.stage_dir)
    summary_csv = Path(args.summary) if args.summary else (stage_dir / "summary.csv")
    if not summary_csv.exists():
        print(f"summary.csv not found: {summary_csv}")
        return 2

    rows = _load_summary(summary_csv)
    if not rows:
        print(f"No rows in summary: {summary_csv}")
        return 2

    rows.sort(key=lambda r: _rank_key(r, str(args.rank_mode)), reverse=True)
    top_k = max(1, min(int(args.top_k), len(rows)))
    top = rows[:top_k]

    # Collect event occurrence across runs and in top runs
    total_occ: Dict[str, int] = {}
    top_occ: Dict[str, int] = {}

    for r in rows:
        ids = _load_discovered_events(r.run_dir)
        for eid in ids:
            total_occ[eid] = total_occ.get(eid, 0) + 1

    for r in top:
        ids = _load_discovered_events(r.run_dir)
        for eid in ids:
            top_occ[eid] = top_occ.get(eid, 0) + 1

    # Score: precision(top)/freq(total) with a rarity bias.
    scored: List[Tuple[float, str]] = []
    for eid, tot in total_occ.items():
        if int(tot) < int(args.min_total_runs):
            continue
        t = int(top_occ.get(eid, 0))
        if t <= 0:
            continue
        precision = t / float(top_k)
        rarity = 1.0 / float(tot)
        score = precision * (0.5 + 0.5 * rarity)
        scored.append((score, eid))

    scored.sort(reverse=True)
    selected = [eid for _, eid in scored[: int(args.max_events)]]

    out_path = Path(args.out) if args.out else (stage_dir / "promoted_discovered_events.json")
    payload = {
        "version": 1,
        "stage_dir": str(stage_dir),
        "rank_mode": str(args.rank_mode),
        "top_k": int(top_k),
        "selected_count": int(len(selected)),
        "events": [{"id": eid, "weight": 1.0} for eid in selected],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    if selected:
        print(f"Top promoted: {selected[:10]}")
    else:
        print("No events promoted (none met criteria)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
