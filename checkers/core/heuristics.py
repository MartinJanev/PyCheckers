from __future__ import annotations
from typing import Callable, Dict
from .constants import BOARD_SIZE
from .state import CheckersState

# Heuristic signature
Heuristic = Callable[[CheckersState], float]


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
    Heuristic function to evaluate the mobility of pieces in a CheckersState.
    It counts the number of legal moves available for each player.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Count the number of legal moves for black pieces and multiply by 0.05
    - Count the number of legal moves for white pieces and multiply by 0.05
    The final score is the difference between black and white moves, adjusted based on the current turn.
    If it's white's turn, the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
    """
    turn = state.turn
    state.turn = 'b'
    b_moves = len(state.legal_moves())
    state.turn = 'w'
    w_moves = len(state.legal_moves())
    state.turn = turn
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
    Heuristic function to evaluate the attack bias in a CheckersState.
    It counts the number of legal capture moves available for each player.
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Count the number of legal capture moves for black pieces and multiply by 0.10
    - Count the number of legal capture moves for white pieces and multiply by 0.10
    The final score is the difference between black and white captures, adjusted based on the current turn
    If it's white's turn, the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on the number of legal capture moves.
    """
    turn = state.turn
    state.turn = 'b'
    bcaps = len([m for m in state.legal_moves() if m.is_capture])
    state.turn = 'w'
    wcaps = len([m for m in state.legal_moves() if m.is_capture])
    state.turn = turn
    s = (bcaps - wcaps) * 0.10
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
    Heuristic function to evaluate the mobility of pieces in a CheckersState,
    focusing on safe moves that do not lead to captures.
    It counts the number of legal moves available for each player that do not result in a capture
    The score is positive for black pieces and negative for white pieces.
    The score is calculated as follows:
    - Count the number of legal moves for black pieces that are not captures and multiply by
        0.03
    - Count the number of legal moves for white pieces that are not captures and multiply by
        0.03
    The final score is the difference between black and white safe moves, adjusted based on the current turn.
    If it's white's turn, the score is negated to reflect the perspective of the player.
    :param state: CheckersState to evaluate, which includes the board and turn
    :return: float: Positive score for black pieces, negative for white pieces,
                    based on the number of legal safe moves.
    """
    turn = state.turn
    s = 0.0
    for side in ('b', 'w'):
        state.turn = side
        for m in state.legal_moves():
            nxt = state.apply_move(m)
            opp = 'w' if side == 'b' else 'b'
            nxt.turn = opp
            if len([x for x in nxt.legal_moves() if x.is_capture]) == 0:
                s += 0.03 if side == 'b' else -0.03
    state.turn = turn
    return s if turn == 'b' else -s


# Additional heuristics based on your list

def h_king_safety(state: CheckersState) -> float:
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == 'B':
                if 1 <= r <= 6 and 1 <= c <= 6:
                    s += 0.2
                else:
                    s -= 0.15
            elif p == 'W':
                if 1 <= r <= 6 and 1 <= c <= 6:
                    s -= 0.2
                else:
                    s += 0.15
    return s if state.turn == 'b' else -s


def h_trapped_pieces(state: CheckersState) -> float:
    s = 0.0
    for side in ('b', 'w'):
        state.turn = side
        movable = {m.src for m in state.legal_moves()}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = state.board[r][c]
                if p.lower() == side and (r, c) not in movable:
                    s += -0.2 if side == 'b' else 0.2
    return s if state.turn == 'b' else -s


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


def h_tempo_advantage(state: CheckersState) -> float:
    # Reward for having capture threats while opponent doesn't
    turn = state.turn
    opp = 'w' if turn == 'b' else 'b'
    state.turn = turn
    my_caps = any(m.is_capture for m in state.legal_moves())
    state.turn = opp
    opp_caps = any(m.is_capture for m in state.legal_moves())
    state.turn = turn
    if my_caps and not opp_caps:
        return 0.25
    elif opp_caps and not my_caps:
        return -0.25
    return 0.0


def h_runaway_king_threat(state: CheckersState) -> float:
    # Check if any king can reach promotion row without being captured (simple heuristic)
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p == 'B' and r > 0:
                s += 0.15
            elif p == 'W' and r < 7:
                s -= 0.15
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
    turn = state.turn
    opp = 'w' if turn == 'b' else 'b'
    state.turn = opp
    opp_moves = len(state.legal_moves())
    state.turn = turn
    return (-opp_moves * 0.04) if turn == 'b' else (opp_moves * 0.04)


def h_exchange_favorability(state: CheckersState) -> float:
    # Rough estimate: if we have more captures available than opponent, favor it
    turn = state.turn
    state.turn = 'b'
    b_caps = sum(m.is_capture for m in state.legal_moves())
    state.turn = 'w'
    w_caps = sum(m.is_capture for m in state.legal_moves())
    state.turn = turn
    diff = b_caps - w_caps
    return diff * 0.1 if turn == 'b' else -diff * 0.1


def h_king_advancement_pressure(state: CheckersState) -> float:
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
    "king_safety": h_king_safety,
    "trapped": h_trapped_pieces,
    "double_corner": h_double_corner_control,
    "tempo_advantage": h_tempo_advantage,
    "runaway_king": h_runaway_king_threat,
    "clustering": h_piece_clustering,
    "opp_mobility_restrict": h_opponent_mobility_restriction,
    "exchange_favorability": h_exchange_favorability,
    "king_adv_pressure": h_king_advancement_pressure,
}


# 2) Combiner uses BASE_HEURISTICS by default to avoid bootstrap issues

def combine(*, registry: Dict[str, Heuristic] = BASE_HEURISTICS, **weights: float) -> Heuristic:
    """
    Combine multiple heuristics into a single heuristic function.
    This function takes named weights for each heuristic and returns a new heuristic function
    that computes a weighted sum of the specified heuristics.
    :param registry: Dict[str, Heuristic] -> a dictionary of heuristics to use
    :param weights: Named weights for each heuristic, e.g., material=1.0
    :return: Heuristic -> a new heuristic function that combines the specified heuristics
    """
    parts: list[tuple[Heuristic, float]] = []
    for name, w in weights.items():
        h = registry.get(name)
        if h is not None and w != 0:
            parts.append((h, float(w)))
        else:
            # silently ignore unknown names or zero weights (or raise if preferred)
            pass

    def _combo(state: CheckersState) -> float:
        total = 0.0
        for h, w in parts:
            total += w * h(state)
        return total

    return _combo


HEURISTICS: Dict[str, Heuristic] = dict(BASE_HEURISTICS)

HEURISTICS["mix_light"] = combine(
    material=1.0,
    piece_square=0.65,
    mobility=0.4,
    center=0.3,
    promotion_potential=0.2,
    back_rank=0.15
)
HEURISTICS["mix_strong"] = combine(
    material_adv=1.0,
    piece_square=0.8,
    mobility=0.6,
    center=0.5,
    attack_bias=0.5,
    king_safety=0.3,
    clustering=0.2,
    opp_mobility_restrict=0.2
)

HEURISTICS["pro_mix"] = HEURISTICS["mix_light"]
HEURISTICS["hard_mix"] = HEURISTICS["mix_strong"]

# Example stronger Hard:
# HEURISTICS["hard_mix"] = combine(material_adv=1.0, piece_square=0.9, mobility=0.7, center=0.5, attack_bias=0.7, king_safety=0.4, clustering=0.3, opp_mobility_restrict=0.3)
