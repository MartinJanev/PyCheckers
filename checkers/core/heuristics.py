from __future__ import annotations
from typing import Callable, Dict
from checkers.util.constants import BOARD_SIZE
from .state import CheckersState

# Heuristic signature
Heuristic = Callable[[CheckersState], float]

# ---- Lightweight per-state move cache (used only during evaluation) ----
def _get_move_cache(state: CheckersState) -> dict:
    """
    Get or create a per-state cache dictionary for storing computed values.
    This cache is used to avoid redundant calculations during heuristic evaluations.
    The cache is stored as an attribute on the state object. If the state object
    is slotted or frozen and cannot have new attributes set, a temporary dictionary
    is used instead (which will not persist across calls).
    :param state: CheckersState object to attach the cache to
    :return: dict: A dictionary for caching computed values
    """
    cache = getattr(state, "_heval_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(state, "_heval_cache", cache)
        except Exception:
            # If state is slotted/frozen, fall back to ephemeral dict (no persistence across calls)
            cache = {}
    return cache

def _cached_legal_moves(state: CheckersState, side: str):
    """
    Return legal moves for a given side, caching on the state object.
    :param state: CheckersState to evaluate
    :param side: 'b' or 'w' for which side to get moves
    :return: list of Move objects representing legal moves for the specified side
    """
    cache = _get_move_cache(state)
    key = ('moves', side)
    if key in cache:
        return cache[key]
    turn = state.turn
    state.turn = side
    try:
        moves = state.legal_moves()
    finally:
        state.turn = turn
    cache[key] = moves
    return moves



# -------------------- Base heuristics (kept, tuned) --------------------

def h_material(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the material balance in a CheckersState.
    It counts the pieces on the board, giving different weights to regular and king pieces.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Regular black piece ('b') = +1.0
    - Regular white piece ('w') = -1.0
    - King black piece ('B') = +1.8
    - King white piece ('W') = -1.8
    The final score is adjusted based on the current turn. If it's white's turn, 
    the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces.
    """""
    score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == 'b':
                score += 1.0
            elif p == 'w':
                score -= 1.0
            elif p == 'B':
                score += 1.8
            elif p == 'W':
                score -= 1.8
    return score if state.turn == 'b' else -score


def h_material_advancement(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the material balance with positional advancement in a CheckersState.
    It counts the pieces on the board, giving different weights to regular and king pieces,
    and adjusts the score based on the row position of the pieces.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Regular black piece ('b') = +1.0 + (7 - row)
    - Regular white piece ('w') = -1.0 - (row)
    - King black piece ('B') = +2.0
    - King white piece ('W') = -2.0
    The final score is adjusted based on the current turn. If it's white's turn,
    the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    with positional advancement considered.
    """
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == 'b':
                s += 1.0 + (7 - r) * 0.05
            elif p == 'w':
                s -= 1.0 + r * 0.05
            elif p == 'B':
                s += 2.0
            elif p == 'W':
                s -= 2.0
    return s if state.turn == 'b' else -s


def h_mobility(state: CheckersState) -> float:
    """
    Mobility: cached move counts per side.

    """
    turn = state.turn
    b_moves = len(_cached_legal_moves(state, 'b'))
    w_moves = len(_cached_legal_moves(state, 'w'))
    s = (b_moves - w_moves) * 0.05
    return s if turn == 'b' else -s


def h_center_control(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the control of the center squares in a CheckersState.
    It checks the pieces in the center squares (3,3), (3,4),
    (4,3), and (4,4) and assigns scores based on their presence.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Center square occupied by a regular black piece ('b') = +0.25
    - Center square occupied by a king black piece ('B') = +0.4
    - Center square occupied by a regular white piece ('w') = -0.25
    - Center square occupied by a king white piece ('W') = -0.4
    The final score is adjusted based on the current turn. If it's white's turn,
    the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on center control.
    """
    centers = {(3, 3), (3, 4), (4, 3), (4, 4)}
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if (r, c) in centers:
                if p.lower() == 'b':
                    s += 0.25 if p == 'b' else 0.4
                elif p.lower() == 'w':
                    s -= 0.25 if p == 'w' else 0.4
    return s if state.turn == 'b' else -s


def h_promotion_potential(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the promotion potential of pieces in a CheckersState.
    It checks the row positions of the pieces and assigns scores based on their proximity to the promotion
    row (the last row for each player).
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Regular black piece ('b') in row 7 = +0.21
    - Regular white piece ('w') in row 0 = -0.21
    - King black piece ('B') = +0.15
    - King white piece ('W') = -0.15
    The final score is adjusted based on the current turn. If it's white's turn,
    the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on promotion potential.
    """
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == 'b':
                s += (7 - r) * 0.03
            elif p == 'w':
                s -= r * 0.03
    return s if state.turn == 'b' else -s


def h_piece_square(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the piece-square table for a CheckersState.
    It uses a predefined table to assign scores based on the position of pieces on the board.
    The score is positive for black pieces and negative for white pieces.
    The table is defined as follows:
    - Each square on the board has a specific score based on its position.
    - The score is positive for black pieces ('b') and negative for white pieces ('w
    ').
    The final score is adjusted based on the current turn. If it's white's turn,
    the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on the piece-square table.
    """
    table = [
        [0, 0.10, 0, 0.10, 0, 0.10, 0, 0.10],
        [0.10, 0, 0.15, 0, 0.15, 0, 0.15, 0],
        [0, 0.20, 0, 0.20, 0, 0.20, 0, 0.20],
        [0.20, 0, 0.25, 0, 0.25, 0, 0.25, 0],
        [0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25],
        [0.20, 0, 0.20, 0, 0.20, 0, 0.20, 0],
        [0, 0.15, 0, 0.15, 0, 0.15, 0, 0.15],
        [0.10, 0, 0.10, 0, 0.10, 0, 0.10, 0],
    ]
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p.lower() == 'b':
                s += table[r][c]
            elif p.lower() == 'w':
                s -= table[r][c]
    return s if state.turn == 'b' else -s


def h_attack_bias(state: CheckersState) -> float:
    """
    Attack bias: cached capture counts per side.
    """
    turn = state.turn
    b_caps = sum(1 for m in _cached_legal_moves(state, 'b') if m.is_capture)
    w_caps = sum(1 for m in _cached_legal_moves(state, 'w') if m.is_capture)
    s = (b_caps - w_caps) * 0.10
    return s if turn == 'b' else -s


def h_connectivity(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the connectivity of pieces in a CheckersState.
    It checks the number of adjacent pieces of the same color for each piece on the board.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - For each black piece ('b'), count the number of adjacent black pieces and multiply by
        0.02
    - For each white piece ('w'), count the number of adjacent white pieces and multiply by
        0.02
    The final score is the sum of these counts, adjusted based on the current turn.
    If it's white's turn, the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on the connectivity of pieces.
    """
    s = 0.0
    nbrs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == '.':
                continue
            cnt = 0
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and state.board[rr][cc] != '.' and state.board[rr][
                    cc].lower() == p.lower():
                    cnt += 1
            if p.lower() == 'b':
                s += cnt * 0.02
            else:
                s -= cnt * 0.02
    return s if state.turn == 'b' else -s


def h_back_rank_guard(state: CheckersState) -> float:
    """
    Heuristic function to evaluate the back rank guard in a CheckersState.
    It checks the presence of pieces on the back rank (the last row for each player)
    and assigns scores based on their presence.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Back rank occupied by a regular black piece ('b') = +0.15
    - Back rank occupied by a king black piece ('B') = +0.15
    - Back rank occupied by a regular white piece ('w') = -0.15
    - Back rank occupied by a king white piece ('W') = -0.15
    The final score is adjusted based on the current turn. If it's white's turn,
    the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on the presence of pieces on the back rank.
    """
    s = 0.0
    for c in range(BOARD_SIZE):
        if state.board[7][c] == 'b':
            s += 0.15
        if state.board[0][c] == 'w':
            s -= 0.15
    return s if state.turn == 'b' else -s


def h_mobility_safe(state: CheckersState) -> float:
    """
    Safe mobility: count non-capture moves that do not give opponent an immediate capture.
    Uses cached base move lists and early skips for capture moves.
    """
    turn = state.turn
    s = 0.0
    for side in ('b', 'w'):
        moves = _cached_legal_moves(state, side)
        for m in moves:
            if m.is_capture:
                continue  # only consider non-captures (lighter and matches intent)
            nxt = state.apply_move(m)
            opp = 'w' if side == 'b' else 'b'
            nxt.turn = opp
            # short-circuit: we only need to know if ANY capture exists
            has_cap = any(x.is_capture for x in nxt.legal_moves())
            if not has_cap:
                s += 0.03 if side == 'b' else -0.03
    state.turn = turn
    return s if turn == 'b' else -s

def h_double_corner_control(state: CheckersState) -> float:
    black_corners = [(7, 0), (7, 2)]
    white_corners = [(0, 5), (0, 7)]
    s = 0.0
    for r, c in black_corners:
        if state.board[r][c].lower() == 'b':
            s += 0.3
    for r, c in white_corners:
        if state.board[r][c].lower() == 'w':
            s -= 0.3
    return s if state.turn == 'b' else -s

def h_piece_clustering(state: CheckersState) -> float:
    s = 0.0
    nbrs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == '.':
                continue
            cnt = 0
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and state.board[rr][cc] != '.' and state.board[rr][
                    cc].lower() == p.lower():
                    cnt += 1
            if p.lower() == 'b':
                s += cnt * 0.03
            else:
                s -= cnt * 0.03
    return s if state.turn == 'b' else -s


def h_opponent_mobility_restriction(state: CheckersState) -> float:
    """
    Heuristic to slightly favor positions that limit opponent's mobility.
    Uses cached move lists.
    0.04 per move difference.
    :param state: CheckersState to evaluate
    :return:
    """
    turn = state.turn
    opp = 'w' if turn == 'b' else 'b'
    state.turn = opp
    opp_moves = len(state.legal_moves())
    state.turn = turn
    return (-opp_moves * 0.04) if turn == 'b' else (opp_moves * 0.04)


def h_exchange_favorability(state: CheckersState) -> float:
    """
    Heuristic to slightly favor positions that have more capturing moves available.
    Uses cached move lists.
    0.1 per capture move difference.
    :param state: CheckersState to evaluate
    :return:
    """
    turn = state.turn
    state.turn = 'b'
    b_caps = sum(m.is_capture for m in state.legal_moves())
    state.turn = 'w'
    w_caps = sum(m.is_capture for m in state.legal_moves())
    state.turn = turn
    diff = b_caps - w_caps
    return diff * 0.1 if turn == 'b' else -diff * 0.1


def h_king_advancement_pressure(state: CheckersState) -> float:
    """
    Heuristic to slightly favor positions where kings are advancing towards opponent's back rank.
    0.02 per row advanced.
    7 rows max for a king.
    0.14 max per king.
    0.28 max total if 2 kings.
    0 if no kings.
    :param state: CheckersState to evaluate
    :return:
    """
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == 'B':
                s += (7 - r) * 0.02
            elif p == 'W':
                s -= r * 0.02
    return s if state.turn == 'b' else -s


# -------------------- Registry bootstrap --------------------

# 1) Base registry: ONLY raw heuristic functions here
BASE_HEURISTICS: Dict[str, Heuristic] = {
    "material": h_material,
    "material_adv": h_material_advancement,
    "mobility": h_mobility,
    "center": h_center_control,
    "promotion_potential": h_promotion_potential,
    "piece_square": h_piece_square,
    "attack_bias": h_attack_bias,
    "connectivity": h_connectivity,
    "back_rank": h_back_rank_guard,
    "mobility_safe": h_mobility_safe,
    "double_corner": h_double_corner_control,
    "clustering": h_piece_clustering,
    "opp_mobility_restrict": h_opponent_mobility_restriction,
    "exchange_favorability": h_exchange_favorability,
    "king_adv_pressure": h_king_advancement_pressure,
}

# 2) Heuristic registry: this is what the rest of the code uses
HEURISTICS: Dict[str, Heuristic] = dict(BASE_HEURISTICS)
