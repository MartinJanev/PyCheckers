from __future__ import annotations
import csv, os, random, time, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

from ..core.state import CheckersState
from ..core.engine import EngineConfig, choose_ai_move
from ..core.heuristics import HEURISTICS
from checkers.util.rules import DrawRulesTracker, DrawRulesConfig


from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import threading


# --------- lightweight console progress ---------
def format_time(secs: float) -> str:
    if secs < 60:
        return f"{secs:4.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m:02d}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m"


class ConsoleProgress:
    def __init__(self, total: int, width: int = 40, title: str = ""):
        """
        Simple console progress bar.
        :param total: Total number of steps (must be > 0).
        :param width: Width of the progress bar in characters.
        :param title: Optional title to display above the progress bar.
        """
        self.total = max(1, total)
        self.width = width
        self.done = 0
        self.start_ts = time.perf_counter()
        self._last_len = 0
        self.title = title

        if self.title:
            sys.stdout.write(self.title + "\n")
            sys.stdout.flush()

    def update(self, step: int = 1, status: str = "") -> None:
        """
        Update the progress bar by a given step.
        This will update the progress bar, showing the current progress,
        percentage, estimated time remaining, and an optional status message.
        :param step: Number of steps to increment the progress by (default is 1).
        If step is negative, it will decrement the progress.
        :param status: Optional status message to display alongside the progress bar.
        If status is provided, it will be printed at the end of the progress bar.
        If status is empty, it will not be printed.
        If status is None, it will not be printed.
        If status is not provided, it will not be printed.
        If status is an empty string, it will not be printed.
        If status is a string, it will be printed at the end of the progress bar.
        If status is a callable, it will be called with the current progress and total.
        """
        self.done += step
        frac = min(1.0, self.done / self.total)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.start_ts
        rate = self.done / elapsed if elapsed > 0 else 0
        remain = (self.total - self.done) / rate if rate > 0 else 0.0
        line = f"[{bar}] {self.done}/{self.total}  {frac * 100:5.1f}%  ETA {format_time(remain)}  {status}"
        sys.stdout.write("\r" + " " * self._last_len)
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish(self, note: str = "") -> None:
        """
        Finalize the progress bar, optionally printing a note.
        This will print the final state of the progress bar and a note.
        If the progress bar is not at 100%, it will still show the final state.
        :param note: Note to print after the progress bar is finished.
        If note is provided, it will be printed on a new line.
        :return:
        """
        self.update(0, status=note)
        sys.stdout.write("\n")
        sys.stdout.flush()


class TqdmProgress:
    """
    Thin wrapper around tqdm that mirrors ConsoleProgress' API.
    Uses tqdm if available, otherwise falls back to ConsoleProgress.
    """

    def __init__(self, total: int, title: str = ""):
        """
        Initialize a TqdmProgress instance.
        If tqdm is not available, falls back to ConsoleProgress.
        :param total: Total number of steps (must be > 0).
        :param title: Optional title for the progress bar.
        """
        try:
            from tqdm import tqdm as _tqdm
            self._tqdm = _tqdm(total=total, desc=title, dynamic_ncols=True, leave=False)
            self._ok = True
        except Exception:
            self._ok = False
            self._fallback = ConsoleProgress(total, title=title)

    def update(self, step: int = 1, status: str = "") -> None:
        if self._ok:
            self._tqdm.update(step)
            if status:
                self._tqdm.set_postfix_str(status)
        else:
            self._fallback.update(step, status)

    def finish(self, note: str = "") -> None:
        if self._ok:
            if note:
                self._tqdm.set_postfix_str(note)
            self._tqdm.close()
        else:
            self._fallback.finish(note)


@dataclass
class GameResult:
    winner: str  # 'w', 'b', 'draw'
    plies: int
    duration_s: float


def play_game(cfg_w: EngineConfig,
              cfg_b: EngineConfig,
              max_plies: int = 175,
              seed: Optional[int] = None) -> GameResult:
    if seed is not None:
        random.seed(seed)

    s = CheckersState()
    start = time.perf_counter()
    plies = 0

    rules = DrawRulesTracker(DrawRulesConfig(
        no_capture_plies_threshold=80,     # 40 full moves = 80 plies
        repetition_threshold=3
    ))
    rules.start(s)

    while plies < max_plies:
        term = s.terminal()  # 'w', 'b', 'draw' or None
        if term:
            return GameResult(term, plies, time.perf_counter() - start)

        cfg = cfg_w if s.turn == 'w' else cfg_b
        mv = choose_ai_move(s, cfg)
        s = s.apply_move(mv)
        plies += 1
        dr = rules.on_move(s, mv)
        if dr == 'draw':
            return GameResult('draw', plies, time.perf_counter() - start)

    return GameResult('draw', plies, time.perf_counter() - start)


ProgressFn = Callable[[int, int, Dict[str, Any]], None]

# -------------------- dynamic weighting --------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    """
    Heuristic weights for the default set of heuristics.
    These weights are used to compute the number of games per pair.
    For example, if heuristic A has weight 2.0 and B has weight 1.0, then A vs B 
    will play 2.0 + 1.0 = 3.0 * base_games games, rounded to the nearest integer. 
    The weights are not used to compute the heuristic values themselves,
    they are only used to scale the number of games played per pair.
    """

    # simpler/baseline
    "material": 2.8,
    "mobility": 2.6,
    "center": 2.4,
    "piece_square": 2.2,
    "promotion_potential": 1.9,
    "connectivity": 1.7,
    "attack_bias": 1.5,
    # stronger mixes (fewer games)
    "material_adv": 1.0,
    # additional heuristics
    "back_rank": 1.2,
    "mobility_safe": 1.3,
    "double_corner": 1.0,
    "clustering": 1.1,
    "opp_mobility_restrict": 1.3,
    "exchange_favorability": 1.0,
    "king_adv_pressure": 1.2,
}


def _weight_for(name: str, weights: Dict[str, float]) -> float:
    return float(weights.get(name, 1.0))


# --------- TOP-LEVEL worker (Windows‑safe for multiprocessing) ---------
def _simulate_pair(
        a: str,
        b: str,
        seed: int,
        *,
        algo: str,
        depth: int,
        use_tt: bool,
        base_games_each: int,
        swap_colors: bool,
        weights: Dict[str, float],
        min_games_each: int,
        max_games_each: Optional[int],
        early_stop: bool,
        progress_queue: Optional[Any] = None,
) -> tuple[Dict[tuple, int], Dict[str, Dict[str, float]], int]:
    """Run one (a,b) matchup; return pairwise h2h wins, pair totals, and games played."""
    rng = random.Random(seed)

    def cfg(hname: str) -> EngineConfig:
        return EngineConfig(algo=algo, depth=depth, heuristic=hname, use_tt=use_tt)

    def rounds_for_pair_local(x: str, y: str) -> int:
        wa = _weight_for(x, weights);
        wb = _weight_for(y, weights)
        r = int(round(base_games_each * (wa + wb) / 2.0))
        if max_games_each is not None:
            r = min(r, max_games_each)
        r = max(r, min_games_each)
        if swap_colors and r % 2 == 1:
            r += 1
        return max(1, r)

    rounds = rounds_for_pair_local(a, b)
    half = rounds // 2 if swap_colors else rounds
    schedule: List[tuple[str, str]] = []
    schedule += [(a, b)] * half
    schedule += [(b, a)] * (rounds - half)

    pair_h2h: Dict[tuple, int] = {(a, b): 0, (b, a): 0}
    pair_totals = {
        a: {"wins": 0, "losses": 0, "draws": 0, "games": 0, "avg_plies": 0.0, "avg_time_s": 0.0},
        b: {"wins": 0, "losses": 0, "draws": 0, "games": 0, "avg_plies": 0.0, "avg_time_s": 0.0},
    }

    wins = {a: 0, b: 0}
    majority = rounds // 2 + 1
    games_played = 0
    for (white_h, black_h) in schedule:
        seed_g = rng.randint(0, 1_000_000)
        res = play_game(cfg(white_h), cfg(black_h), seed=seed_g)
        games_played += 1
        # live progress tick per game (parallel-safe)
        if progress_queue is not None:
            try:
                progress_queue.put(1)
            except Exception:
                pass

        for h in (white_h, black_h):
            pair_totals[h]["games"] += 1
            pair_totals[h]["avg_plies"] += res.plies
            pair_totals[h]["avg_time_s"] += res.duration_s

        if res.winner == 'w':
            pair_h2h[(white_h, black_h)] += 1
            pair_totals[white_h]["wins"] += 1
            pair_totals[black_h]["losses"] += 1
            wins[white_h] += 1
        elif res.winner == 'b':
            pair_h2h[(black_h, white_h)] += 1
            pair_totals[black_h]["wins"] += 1
            pair_totals[white_h]["losses"] += 1
            wins[black_h] += 1
        else:
            pair_totals[white_h]["draws"] += 1
            pair_totals[black_h]["draws"] += 1

        if early_stop and max(wins[a], wins[b]) >= majority:
            # Fill remaining planned games so global bar stays consistent
            remaining = rounds - games_played
            if progress_queue is not None and remaining > 0:
                try:
                    progress_queue.put(remaining)
                except Exception:
                    pass
            break
    return pair_h2h, pair_totals, games_played


def run_matchups(
        heur_names: List[str],
        algo: str = "minimax",
        depth: int = 5,
        use_tt: bool = False,
        base_games_each: int = 6,
        swap_colors: bool = True,
        seed: int = 42,
        csv_out: Optional[str] = None,
        progress: Optional[str | ProgressFn] = "bar",
        weights: Optional[Dict[str, float]] = None,
        min_games_each: int = 2,
        max_games_each: Optional[int] = None,
        early_stop: bool = True,
        *,
        parallel_pairs: int = 1,
        max_total_games: Optional[int] = None,
        show_setup_progress: bool = True,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """
    Round-robin tournament with dynamic games per pair and optional parallelism.
    Shows a setup progress bar (planning) before the main per-game bar.
    Returns (head_to_head, summary).
    """
    # ---- Phase 0: validation ----
    for h in heur_names:
        if h not in HEURISTICS:
            raise ValueError(f"Unknown heuristic: {h}")

    if weights is None:
        weights = DEFAULT_WEIGHTS

    # We know number of pairs up-front:
    n = len(heur_names)
    num_pairs = n * (n - 1) // 2

    # Setup progress: estimate total setup steps (cheap, just visual)
    setup_steps = 0
    # build pairs
    setup_steps += num_pairs
    # compute planned rounds
    setup_steps += num_pairs
    # optional scaling
    if max_total_games is not None:
        setup_steps += num_pairs
    # seeding
    setup_steps += num_pairs
    # parallel args
    if parallel_pairs > 1:
        setup_steps += num_pairs
    # one finalization step
    setup_steps += 1

    setup_bar: Optional[ConsoleProgress] = None
    if isinstance(progress, str) and progress.lower() == "bar" and show_setup_progress:
        setup_bar = ConsoleProgress(setup_steps, title="Setup:")

    # ---- Phase 1: build pair list (with visual updates) ----
    pairs: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((heur_names[i], heur_names[j]))
            if setup_bar:
                setup_bar.update(1, status=f"pairs {len(pairs)}/{num_pairs}")

    # helper: rounds per pair (pre-scaling)
    def _rounds_pre(a: str, b: str) -> int:
        wa = _weight_for(a, weights)
        wb = _weight_for(b, weights)
        r = int(round(base_games_each * (wa + wb) / 2.0))
        if max_games_each is not None:
            r = min(r, max_games_each)
        r = max(r, min_games_each)
        if swap_colors and r % 2 == 1:
            r += 1
        return max(1, r)

    # ---- Phase 2: compute planned per-pair rounds ----
    planned_runs_per_pair: Dict[Tuple[str, str], int] = {}
    for p in pairs:
        r = _rounds_pre(*p)
        planned_runs_per_pair[p] = r
        if setup_bar:
            setup_bar.update(1, status=f"rounds for {p[0]} vs {p[1]} = {r}")

    planned_total_runs = sum(planned_runs_per_pair.values())

    # ---- Phase 3: optional global cap scaling ----
    scaled_rounds_per_pair: Dict[Tuple[str, str], int] = planned_runs_per_pair.copy()
    if max_total_games is not None and planned_total_runs > max_total_games:
        scale = max_total_games / planned_total_runs
        for p in pairs:
            r = max(1, int(round(planned_runs_per_pair[p] * scale)))
            if swap_colors and r % 2 == 1:
                r += 1
            scaled_rounds_per_pair[p] = r
            if setup_bar:
                setup_bar.update(1, status=f"scaling {p[0]} vs {p[1]} -> {r}")

        rounds_for_pair = lambda a, b: scaled_rounds_per_pair[(a, b)]  # noqa: E731 (safe: not used in multiprocessing)
    else:
        rounds_for_pair = lambda a, b: planned_runs_per_pair[(a, b)]  # noqa: E731

    # compute final total runs
    total_runs = sum(rounds_for_pair(a, b) for (a, b) in pairs)

    # ---- Phase 4: seeds per pair ----
    seeds = {(a, b): (hash((seed, a, b)) & 0x7fffffff) for (a, b) in pairs}
    if setup_bar:
        for _ in pairs:
            setup_bar.update(1, status="seeding pairs")

    # ---- Phase 5: parallel argument payloads (if any) ----
    args_iter = None
    if parallel_pairs > 1:
        args_iter = [(a, b, seeds[(a, b)]) for (a, b) in pairs]
        if setup_bar:
            for _ in pairs:
                setup_bar.update(1, status="preparing parallel args")

    if setup_bar:
        setup_bar.finish("setup done")

    # ---- Phase 6: main progress bar for games ----
    prog_bar: Optional[Any] = None
    callback: Optional[ProgressFn] = None
    if isinstance(progress, str):
        mode = progress.lower()
        if mode == "bar":
            prog_bar = ConsoleProgress(total_runs, title="Matches:")
        elif mode == "tqdm":
            prog_bar = TqdmProgress(total_runs, title="Matches:")
    elif callable(progress):
        callback = progress

    head_to_head: Dict[str, Dict[str, int]] = {h: {k: 0 for k in heur_names} for h in heur_names}
    totals = {h: {"wins": 0, "losses": 0, "draws": 0, "games": 0, "avg_plies": 0.0, "avg_time_s": 0.0}
              for h in heur_names}

    ran = 0

    if parallel_pairs > 1:
        worker = partial(
            _simulate_pair,
            algo=algo, depth=depth, use_tt=use_tt,
            base_games_each=base_games_each, swap_colors=swap_colors,
            weights=weights, min_games_each=min_games_each, max_games_each=max_games_each,
            early_stop=early_stop,
        )
        # Use a Manager Queue so workers can report per-game progress
        prog_q = None
        consumer_thread = None
        if prog_bar is not None:
            mgr = Manager()
            prog_q = mgr.Queue()
            worker = partial(worker, progress_queue=prog_q)

            def _consume(q, bar, total):
                done = 0
                while done < total:
                    try:
                        inc = q.get()
                    except Exception:
                        break
                    if not isinstance(inc, int):
                        inc = 1
                    done += inc
                    bar.update(inc, status=f"{done}/{total} games")
                bar.finish("done")

            consumer_thread = threading.Thread(
                target=_consume, args=(prog_q, prog_bar, total_runs), daemon=True
            )
            consumer_thread.start()

        with Pool(processes=min(parallel_pairs, cpu_count())) as pool:
            for (pair_h2h, pair_totals, games_played) in pool.starmap(worker, args_iter):
                # merge h2h
                for (x, y), wns in pair_h2h.items():
                    head_to_head[x][y] += wns
                # merge totals
                for h, t in pair_totals.items():
                    for k in ("wins", "losses", "draws", "games"):
                        totals[h][k] += t[k]
                    totals[h]["avg_plies"] += t["avg_plies"]
                    totals[h]["avg_time_s"] += t["avg_time_s"]

                ran += games_played

        # Ensure the consumer finishes and closes the bar
        if consumer_thread is not None:
            consumer_thread.join()
    else:
        rng = random.Random(seed)
        for a, b in pairs:
            rounds = rounds_for_pair(a, b)
            half = rounds // 2 if swap_colors else rounds
            schedule: List[Tuple[str, str]] = []
            schedule += [(a, b)] * half
            schedule += [(b, a)] * (rounds - half)
            wins = {a: 0, b: 0}
            majority = rounds // 2 + 1
            for gidx, (white_h, black_h) in enumerate(schedule):
                seed_g = rng.randint(0, 1_000_000)
                res = play_game(
                    EngineConfig(algo=algo, depth=depth, heuristic=white_h, use_tt=use_tt),
                    EngineConfig(algo=algo, depth=depth, heuristic=black_h, use_tt=use_tt),
                    seed=seed_g
                )
                for h in (white_h, black_h):
                    totals[h]["games"] += 1
                    totals[h]["avg_plies"] += res.plies
                    totals[h]["avg_time_s"] += res.duration_s
                if res.winner == 'w':
                    head_to_head[white_h][black_h] += 1
                    totals[white_h]["wins"] += 1
                    totals[black_h]["losses"] += 1
                    wins[white_h] += 1
                elif res.winner == 'b':
                    head_to_head[black_h][white_h] += 1
                    totals[black_h]["wins"] += 1
                    totals[white_h]["losses"] += 1
                    wins[black_h] += 1
                else:
                    totals[white_h]["draws"] += 1
                    totals[black_h]["draws"] += 1
                ran += 1
                if prog_bar is not None:
                    status = f"{white_h}(W) vs {black_h}(B) g {gidx + 1}/{rounds} -> {res.winner.upper()}  {ran}/{total_runs} games"
                    prog_bar.update(1, status=status)
                if callback is not None:
                    try:
                        callback(ran, total_runs, {
                            "white": white_h, "black": black_h,
                            "round_index": gidx + 1, "rounds": rounds,
                            "pair": (a, b), "result": res.winner, "plies": res.plies,
                            "wins_a": wins[a], "wins_b": wins[b],
                        })
                    except Exception:
                        pass
                if early_stop and max(wins[a], wins[b]) >= majority:
                    remaining = rounds - (gidx + 1)
                    if remaining > 0 and prog_bar is not None:
                        prog_bar.update(remaining, status=f"{a} vs {b} early stop")
                    ran += remaining
                    break

    if prog_bar is not None and parallel_pairs <= 1:
        prog_bar.finish("done")

    # finalize averages and win%
    summary: Dict[str, Dict[str, float]] = {}
    for h, t in totals.items():
        g = max(1, t["games"])
        avg_plies = t["avg_plies"] / g
        avg_time = t["avg_time_s"] / g
        winp = 100.0 * t["wins"] / g
        summary[h] = {
            "games": float(t["games"]),
            "wins": float(t["wins"]),
            "losses": float(t["losses"]),
            "draws": float(t["draws"]),
            "win_pct": winp,
            "avg_plies": avg_plies,
            "avg_time_s": avg_time,
        }

    if csv_out:
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["heuristic", "games", "wins", "losses", "draws", "win_pct", "avg_plies", "avg_time_s"])
            for h, s in sorted(summary.items(), key=lambda kv: (-kv[1]["win_pct"], kv[0])):
                w.writerow([
                    h, int(s["games"]), int(s["wins"]), int(s["losses"]), int(s["draws"]),
                    f"{s['win_pct']:.1f}", f"{s['avg_plies']:.1f}", f"{s['avg_time_s']:.3f}"
                ])

        # Write h2h.csv alongside summary
        h2h_csv_path = os.path.join(os.path.dirname(csv_out), "h2h.csv")
        with open(h2h_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            heuristics_list = sorted(head_to_head.keys())
            w.writerow([""] + heuristics_list)
            for h in heuristics_list:
                row = [h] + [head_to_head[h][op] for op in heuristics_list]
                w.writerow(row)

    return head_to_head, summary
