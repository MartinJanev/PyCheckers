# checkers/analysis/bench_heuristics.py  (fixed outDir handling)
from __future__ import annotations
import argparse, os, pprint
from typing import List, Dict
from tqdm import tqdm
import pandas as _pd

from .arena import run_matchups  # writes summary.csv + h2h.csv for us

# Pull the registry so we don’t duplicate a default list here
try:
    from ..core.heuristics import HEURISTICS

    DEFAULT_SET: List[str] = sorted(HEURISTICS.keys())
except Exception:
    DEFAULT_SET = [
        "material", "piece_square", "mobility", "center",
        "material_adv", "attack_bias", "connectivity",
        "promotion_potential", "back_rank", "mobility_safe",
        "double_corner", "clustering", "opp_mobility_restrict", "exchange_favorability",
        "king_adv_pressure"
    ]


def _parse_weights(items: List[str]) -> Dict[str, float]:
    w: Dict[str, float] = {}
    for it in items:
        try:
            name, val = it.split("=", 1)
            w[name.strip()] = float(val)
        except Exception:
            print(f"Bad --weight '{it}', expected name=float", flush=True)
    return w


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark heuristics (single depth or depth sweep)."
    )

    # Depth control: either one --depth or multiple --depths
    ap.add_argument("--depth", type=int, default=None, help="single search depth")
    ap.add_argument("--depths", nargs="*", type=int, default=None,
                    help="list of depths to sweep (e.g. 2 3 4 5)")

    ap.add_argument("--algos", nargs="+", default=["minimax"],
                    choices=["minimax", "expectimax", "random"],
                    help="one or more search algorithms to run")
    ap.add_argument("--use-tt", action="store_true")
    ap.add_argument("--base-games", type=int, default=6,
                    help=("baseline games per pairing before weighting; "
                          "each heuristic gets at least this many games vs each other"))
    ap.add_argument("--min-games", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--no-swap", action="store_true", help="do not swap colors")
    ap.add_argument("--set", nargs="*", default=DEFAULT_SET, help="heuristic names to include")
    ap.add_argument("--no-progress", action="store_true", help="disable progress output inside arena")
    ap.add_argument("--weight", action="append", default=[], help="override per-heuristic weight like name=1.25")
    ap.add_argument("--parallel-pairs", type=int, default=1, help="number of matchup pairs to run in parallel")
    ap.add_argument("--max-total-games", type=int, default=None, help="cap total games across the whole tournament")

    # Output control (depth_dependence-style defaults; optional manual pattern)
    ap.add_argument("--outDir", default=None,
                    help=("Output directory pattern with placeholders {depth} and {algo}. "
                          "Example: 'stats/depth_dependence/{depth}/{algo}'. "
                          "If omitted: single depth -> 'stats/bench_7/{algo}', "
                          "sweep -> 'stats/depth_dependence/{depth}/{algo}'"))

    args = ap.parse_args()

    w_overrides = _parse_weights(args.weight)
    progress_mode = None if args.no_progress else "tqdm"

    # Resolve depths
    if args.depths:
        depths = list(dict.fromkeys(int(d) for d in args.depths))  # unique, keep order
    elif args.depth is not None:
        depths = [int(args.depth)]
    else:
        depths = [5]  # sensible default

    # Resolve output pattern
    if args.outDir is None:
        # depth_dependence-style defaults
        out_pat = "stats/bench_7/{algo}" if len(depths) == 1 else "stats/depth_dependence/{depth}/{algo}"
    else:
        out_pat = args.outDir  # may include {depth} and/or {algo}, or be a plain folder

    for depth in depths:
        for algo in args.algos:
            out_dir = out_pat.format(depth=depth, algo=algo)
            out_dir = os.path.normpath(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            summary_csv_path = os.path.join(out_dir, "summary.csv")

            with tqdm(total=1, desc=f"{algo} @ depth {depth}", disable=args.no_progress) as p:
                h2h, summary = run_matchups(
                    heur_names=args.set,
                    algo=algo,
                    depth=depth,
                    use_tt=args.use_tt,
                    base_games_each=args.base_games,
                    swap_colors=not args.no_swap,
                    seed=42,
                    csv_out=summary_csv_path,  # arena writes summary.csv and h2h.csv
                    progress=progress_mode,  # progress bar inside arena
                    weights=w_overrides if w_overrides else None,
                    min_games_each=args.min_games,
                    max_games_each=args.max_games,
                    early_stop=not args.no_early_stop,
                    parallel_pairs=args.parallel_pairs,
                    max_total_games=args.max_total_games,
                )
                p.update(1)

            print(f"\n=== [depth {depth} | {algo}] Summary (sorted by win%) ===")
            for name, s in sorted(summary.items(), key=lambda kv: (-kv[1]['win_pct'], kv[0])):
                print(f"{name:18s}  win%={s['win_pct']:5.1f}  "
                      f"W/L/D={int(s['wins'])}/{int(s['losses'])}/{int(s['draws'])}  "
                      f"avg_plies={s['avg_plies']:.1f}  avg_time={s['avg_time_s']:.3f}s")

            print(f"\n=== [depth {depth} | {algo}] Head-to-Head (wins A vs B) ===")
            try:
                df = _pd.DataFrame(h2h).T
                print(df.fillna(0).astype(int))
            except Exception:
                pprint.pprint(h2h)
            print(f"\nResults written to {summary_csv_path}")


if __name__ == "__main__":
    main()
