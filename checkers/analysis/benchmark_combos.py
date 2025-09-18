from __future__ import annotations
import argparse
import csv
import itertools
import os
import random
import time
from typing import List, Tuple, Any
from tqdm import tqdm

from ..core.state import CheckersState
from ..core.engine import choose_ai_move
from ..core.heuristics import BASE_HEURISTICS, HEURISTICS
import multiprocessing as mp


# ---------------- Utility shims ---------------- #
def _run_match(task):
    """Worker: run one combo vs baseline for a fixed number of games.
    task = (cfg_dict, names_tuple, weights_list, games_per_pair, seed)
    Returns a result dict like the rows used in CSV.
    """
    (cfg, names, weights, games_per_pair, seed) = task

    # local seeding for reproducibility/diversity across workers
    try:
        random.seed(seed)
    except Exception:
        pass

    # reconstruct heuristic inside the worker (avoid pickling closures)
    hs = [BASE_HEURISTICS[n] for n in names]

    def h(state):
        return sum(w * f(state) for w, f in zip(weights, hs))

    baseline = HEURISTICS.get('mobility_safe')

    w1 = w2 = d = 0
    for g in range(games_per_pair):
        if g % 2 == 0:
            r = play_game(cfg.copy(), h, baseline)
        else:
            r = play_game(cfg.copy(), baseline, h)
            r = -r
        if r > 0:
            w1 += 1
        elif r < 0:
            w2 += 1
        else:
            d += 1

    label = "[" + ",".join(f"{w:.3f}" for w in weights) + "]"
    return {'names': ",".join(names), 'weights': label, 'wins': w1, 'losses': w2, 'draws': d, 'score': w1 - w2}


def _try_get(obj: Any, names: tuple[str, ...]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def get_initial_state() -> CheckersState:
    for attr in ("initial", "initial_state", "start", "from_start"):
        fn = getattr(CheckersState, attr, None)
        if callable(fn):
            return fn()
    return CheckersState()


def get_turn(state: CheckersState) -> str:
    return getattr(state, "turn", getattr(state, "player", getattr(state, "current", "b")))


def get_legal_moves(state: CheckersState):
    fn = _try_get(state, ("legal_moves", "moves", "get_legal_moves"))
    return list(fn()) if callable(fn) else []


def apply_move(state: CheckersState, mv):
    fn = _try_get(state, ("apply_move", "apply", "play", "make_move", "do_move"))
    res = fn(mv)
    return res if res is not None else state


def is_terminal(state: CheckersState) -> bool:
    fn = _try_get(state, ("is_terminal", "terminal", "is_over", "game_over", "over"))
    if callable(fn):
        return bool(fn())
    if isinstance(fn, bool):
        return fn
    return len(get_legal_moves(state)) == 0


def get_result(state: CheckersState) -> str:
    fn = _try_get(state, ("result", "outcome", "winner"))
    res = fn() if callable(fn) else None
    if res in ("b", "w", "draw"):
        return res
    if isinstance(res, str):
        low = res.lower()
        if "black" in low or low == "b": return "b"
        if "white" in low or low == "w": return "w"
        if "draw" in low or "tie" in low: return "draw"
    if len(get_legal_moves(state)) == 0:
        return "w" if get_turn(state) == "b" else "b"
    return "draw"


# ---------------- Benchmark logic ---------------- #

def dirichlet_weights(k: int, alpha: float = 1.0) -> List[float]:
    xs = [random.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(xs) or 1.0
    return [x / s for x in xs]


def play_game(cfg: dict, h_black, h_white, max_plies: int = 300) -> int:
    state = get_initial_state()
    ply = 0
    while not is_terminal(state) and ply < max_plies:
        h = h_black if get_turn(state) == 'b' else h_white
        cfg['heuristic'] = h
        mv = choose_ai_move(state, cfg)
        if mv is None:
            break
        state = apply_move(state, mv)
        ply += 1
    res = get_result(state)
    return +1 if res == 'b' else -1 if res == 'w' else 0


def play_match(cfg: dict, h1, h2, games: int = 4) -> Tuple[int, int, int]:
    w1 = w2 = d = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(cfg, h1, h2)
        else:
            r = play_game(cfg, h2, h1)
            r = -r
        if r > 0:
            w1 += 1
        elif r < 0:
            w2 += 1
        else:
            d += 1
    return w1, w2, d


def make_combo(names, strength):
    """
    Given a tuple/list of heuristic names and a strength parameter,
    returns a (label, heuristic_fn) pair.
    The label is a string encoding the weights, the heuristic_fn is a function
    that computes the weighted sum of the selected heuristics.
    """
    ws = dirichlet_weights(len(names), alpha=strength)
    label = "[" + ",".join(f"{w:.3f}" for w in ws) + "]"
    hs = [BASE_HEURISTICS[n] for n in names]

    def h(state):
        return sum(w * f(state) for w, f in zip(ws, hs))

    return label, h


def benchmark(
        algo: str,
        depth: int,
        ttable: bool,
        base_names: List[str],
        num_sets: int,
        games_per_pair: int,
        seed: int,
        strength: float,
        out_dir: str = os.path.join('stats', 'combos'),
        tag: str | None = None,
        parallel_pairs: int = 1
):
    random.seed(seed)
    cfg = {
        'algo': algo,
        'depth': depth,
        'use_tt': ttable,
        'transposition_table': ttable,
        'heuristic': None
    }
    all_combos = list(itertools.combinations(base_names, 5))
    combos = all_combos[:num_sets] if num_sets and num_sets < len(all_combos) else all_combos

    # Build tasks: (cfg, names, weights, games_per_pair, seed)
    tasks = []
    for idx, names in enumerate(combos, 1):
        ws = dirichlet_weights(len(names), alpha=strength)
        tasks.append((cfg.copy(), names, ws, games_per_pair, seed + idx))

    baseline = HEURISTICS.get('mobility_safe')
    t0 = time.time()

    # Progress & streaming controls via env (no CLI change)
    progress = os.environ.get('BENCH_PROGRESS', '1') != '0'
    stream_csv = os.environ.get('BENCH_STREAM', '0') == '1'

    total_combos = len(combos)
    total_games = total_combos * games_per_pair

    # Prepare CSV writer (optional streaming)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = f"benchmark_combination_{stamp}"
    if tag:
        base += f"_{tag}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    csv_file = None
    csv_writer = None
    if stream_csv:
        os.makedirs(out_dir, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=['names', 'weights', 'wins', 'losses', 'draws', 'score'])
        csv_writer.writeheader()

    rows = []
    if progress:
        p_games = tqdm(total=total_games, desc="Games", position=0)
    else:
        p_games = None

    if parallel_pairs and parallel_pairs > 1:
        # Windows-safe context
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=parallel_pairs, maxtasksperchild=64) as pool:
            for res in pool.imap_unordered(_run_match, tasks, chunksize=1):
                rows.append(res)
                if csv_writer:
                    csv_writer.writerow(res)
                    csv_file.flush()
                if p_games:
                    p_games.update(games_per_pair)
    else:
        # Fallback: run serially
        for t in tasks:
            res = _run_match(t)
            rows.append(res)
            if csv_writer:
                csv_writer.writerow(res)
                csv_file.flush()
            if p_games:
                p_games.update(games_per_pair)

    if p_games:
        p_games.close()
    if csv_file:
        csv_file.close()

    dt = time.time() - t0
    rows.sort(key=lambda r: (r['score'], r['wins']), reverse=True)
    print("\nTop 10 combos:")
    for r in rows[:10]:
        print(
            f"+{'+'.join(r['names'].split(','))} | {r['weights']} -> W{r['wins']} L{r['losses']} D{r['draws']} score={r['score']}")
    print(f"\nEvaluated {len(rows)} combos in {dt:.1f}s")
    # If we didn't stream, write CSV at the end
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(csv_path) or os.environ.get('BENCH_STREAM', '0') != '1':
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['names', 'weights', 'wins', 'losses', 'draws', 'score'])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark combinations of five heuristics against a baseline mix.')
    parser.add_argument('--algo', default='expectimax', choices=['minimax', 'expectimax'])
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--ttable', action='store_true')
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--sample', type=int, default=40,
                        help='Number of heuristic combinations to sample (if we want all, ')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--from-list', nargs='*', default=None, help="List of heuristic names to use (default: all)")
    parser.add_argument('--out', default=os.path.join('stats', 'combos'))
    parser.add_argument('--tag', default=None)
    parser.add_argument('--parallel-pairs', type=int, default=1, help='Number of matches to run in parallel')
    parser.add_argument('--paralel-pairs', type=int, help=argparse.SUPPRESS)  # alias (misspelling)
    args = parser.parse_args()
    if getattr(args, 'paralel_pairs', None) and not args.parallel_pairs:
        args.parallel_pairs = args.paralel_pairs
    pool = args.from_list if args.from_list else sorted([n for n in BASE_HEURISTICS.keys()])
    if len(pool) < 3:
        raise SystemExit('Need at least five heuristics in the pool.')
    benchmark(algo=args.algo, depth=args.depth, ttable=args.ttable, base_names=pool, num_sets=args.sample,
              games_per_pair=args.games, seed=args.seed, strength=args.strength, out_dir=args.out, tag=args.tag,
              parallel_pairs=args.parallel_pairs)
