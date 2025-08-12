from __future__ import annotations
from typing import Optional, List, Dict, TypedDict, Tuple
import math, random
from .state import CheckersState
from .move import Move
from .heuristics import HEURISTICS, Heuristic


class TTEntry(TypedDict):
    depth: int
    value: float
    flag: int  # 0 exact, -1 upper, 1 lower
    move: Move | None


EXACT, UPPER, LOWER = 0, -1, 1


class EngineConfig(TypedDict, total=False):
    # NEW: algo selector
    algo: str  # "random" | "minimax" | "expectimax"
    depth: int
    heuristic: str
    use_tt: bool


def order_moves(state: CheckersState, moves: List[Move]) -> List[Move]:
    """
    Order moves for minimax/expectimax:
    - prioritize captures first
    - then order by length of capture path (longer captures first)
    - non-capture moves last
    :param state: CheckersState to evaluate moves against
    :param moves: List of moves to order
    :return: Sorted list of moves
    """
    return sorted(moves, key=lambda m: (not m.is_capture, -len(m.path)))


def evaluate_with(state: CheckersState, h: Heuristic) -> float:
    return h(state)


def zobrist_hash(state: CheckersState) -> int:
    """
    Compute a Zobrist hash for the given CheckersState.
    This is a unique identifier for the state, useful for transposition tables.
    The hash is computed based on the positions of pieces, the turn, and the board state
    using a simple XOR-based hashing scheme.
    The hash is 64 bits long.
    :param state: CheckersState to hash, which includes the board and turn
    :return: 64-bit integer hash of the state, suitable for use in a transposition table
    """
    acc = 1469598103934665603
    for r in range(8):
        for c in range(8):
            acc ^= hash((r, c, state.board[r][c]));
            acc *= 1099511628211
    acc ^= hash(state.turn);
    acc *= 1099511628211
    return acc & ((1 << 64) - 1)


# ---------------- Minimax (with optional TT) ----------------
def alphabeta(state: CheckersState, depth: int, alpha: float, beta: float,
              h: Heuristic, tt: Dict[int, TTEntry] | None) -> Tuple[float, Optional[Move]]:
    """
    Perform the Alpha-Beta pruning algorithm on the CheckersState.
    This function recursively explores the game tree to find the best move for the current player
    while pruning branches that won't affect the final decision
    :param state: CheckersState to evaluate, which includes the board and turn
    :param depth: int, the maximum depth to search in the game tree
    :param alpha: float, the best score that the maximizer currently can guarantee at that level or above
    :param beta: float, the best score that the minimizer currently can guarantee at that level or above
    :param h: Heuristic, the heuristic function to evaluate the state
    :param tt: Dict[int, TTEntry] | None, optional transposition table for caching results
    :return: Tuple[float, Optional[Move]] -> a tuple containing the best score for the current player
             and the best move to achieve that score
    """
    winner = state.terminal()
    if winner is not None:
        if winner == 'draw': return 0.0, None
        # score from side-to-move perspective
        return (math.inf, None) if winner == state.turn else (-math.inf, None)
    if depth == 0:
        return evaluate_with(state, h), None

    key = zobrist_hash(state) if tt is not None else None
    if tt is not None and key in tt:
        entry = tt[key]
        if entry["depth"] >= depth:
            val, flag, mv = entry["value"], entry["flag"], entry["move"]
            if flag == EXACT: return val, mv
            if flag == LOWER: alpha = max(alpha, val)
            if flag == UPPER: beta = min(beta, val)
            if alpha >= beta: return val, mv

    best_move: Optional[Move] = None
    moves = order_moves(state, state.legal_moves())
    if not moves:
        # no legal moves means the game is terminal and would have been caught above,
        # but guard anyway
        return (-math.inf if state.turn == 'b' else math.inf), None

    if state.turn == 'b':  # maximizing for Black
        max_eval = -math.inf
        for m in moves:
            val, _ = alphabeta(state.apply_move(m), depth - 1, alpha, beta, h, tt)
            if val > max_eval: max_eval, best_move = val, m
            alpha = max(alpha, val)
            if alpha >= beta: break
        if tt is not None and key is not None:
            tt[key] = {"depth": depth, "value": max_eval, "flag": EXACT, "move": best_move}
        return max_eval, best_move
    else:  # minimizing for White
        min_eval = math.inf
        for m in moves:
            val, _ = alphabeta(state.apply_move(m), depth - 1, alpha, beta, h, tt)
            if val < min_eval: min_eval, best_move = val, m
            beta = min(beta, val)
            if alpha >= beta: break
        if tt is not None and key is not None:
            tt[key] = {"depth": depth, "value": min_eval, "flag": EXACT, "move": best_move}
        return min_eval, best_move


# ---------------- Expectimax (uniform opponent) ----------------
def expectimax(state: CheckersState, depth: int, h: Heuristic) -> Tuple[float, Optional[Move]]:
    """
    Perform the Expectimax algorithm on the CheckersState.
    This function recursively explores the game tree to find the best move for the current player
    while treating the opponent as a uniform random player.
    It evaluates the expected value of moves for the opponent (White) and chooses the best move
    for the maximizing player (Black).
    :param state: CheckersState to evaluate, which includes the board and turn
    :param depth: Integer, the maximum depth to search in the game tree
    :param h: Heuristic, the heuristic function to evaluate the state
    :return: Tuple[float, Optional[Move]] -> a tuple containing the best expected score for the current player
             and the best move to achieve that score
    """
    winner = state.terminal()
    if winner is not None:
        if winner == 'draw': return 0.0, None
        return (math.inf, None) if winner == state.turn else (-math.inf, None)
    if depth == 0:
        return evaluate_with(state, h), None

    moves = state.legal_moves()
    if not moves:
        return (-math.inf if state.turn == 'b' else math.inf), None

    # When it's the MAX agent (AI is Black), choose best move.
    # When it's the MIN agent (White), we take the EXPECTED value over all moves.
    if state.turn == 'b':  # Max node
        best_val, best_move = -math.inf, None
        for m in order_moves(state, moves):
            val, _ = expectimax(state.apply_move(m), depth - 1, h)
            if val > best_val: best_val, best_move = val, m
        return best_val, best_move
    else:  # Expectation over White moves (uniform random model)
        vals = []
        for m in moves:
            val, _ = expectimax(state.apply_move(m), depth - 1, h)
            vals.append(val)
        return (sum(vals) / len(vals), None)


# ---------------- Top-level chooser ----------------
def choose_ai_move(state: CheckersState, config: EngineConfig) -> Move:
    """
    Choose the best move for the AI based on the given CheckersState and configuration.
    This function selects the algorithm to use (random, minimax, or expectimax) and
    applies the appropriate search strategy to find the best move.
    It also handles the case where no legal moves are available.
    :param state: CheckersState to evaluate, which includes the board and turn
    :param config: EngineConfig, configuration for the AI engine, including algorithm choice,
                   depth, heuristic, and whether to use a transposition table
    :return: Move -> the best move for the AI to make based on the current state and configuration
    """
    algo = config.get("algo", "minimax")
    h = HEURISTICS.get(config.get("heuristic", "material"), HEURISTICS["material"])

    # Beginner: pick a random legal move
    if algo == "random":
        legal = state.legal_moves()
        if not legal:
            raise RuntimeError("No legal moves for AI.")
        return random.choice(legal)

    # Expectimax tier
    if algo == "expectimax":
        val, mv = expectimax(state, int(config.get("depth", 3)), h)
        if mv is None:
            legal = state.legal_moves()
            if not legal:
                raise RuntimeError("No legal moves for AI.")
            return legal[0]
        return mv

    # Default: minimax + optional TT
    tt = {} if config.get("use_tt", False) else None
    val, mv = alphabeta(state, int(config.get("depth", 5)), -math.inf, math.inf, h, tt)
    if mv is None:
        legal = state.legal_moves()
        if not legal:
            raise RuntimeError("No legal moves for AI.")
        return legal[0]
    return mv
