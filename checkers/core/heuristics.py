from __future__ import annotations
from typing import Callable
from .constants import BOARD_SIZE
from .state import CheckersState

# Heuristic signature
Heuristic = Callable[[CheckersState], float]

def h_material(state: CheckersState) -> float:
    # Basic material and kings
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
    turn = state.turn
    state.turn = 'b'; b_moves = len(state.legal_moves())
    state.turn = 'w'; w_moves = len(state.legal_moves())
    state.turn = turn
    s = (b_moves - w_moves) * 0.05
    return s if turn == 'b' else -s

def h_center_control(state: CheckersState) -> float:
    centers = {(3,3),(3,4),(4,3),(4,4)}
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if (r,c) in centers:
                if p.lower()=='b': s += 0.25 if p=='b' else 0.4
                elif p.lower()=='w': s -= 0.25 if p=='w' else 0.4
    return s if state.turn=='b' else -s

def h_back_rank_guard(state: CheckersState) -> float:
    s = 0.0
    # bonus for keeping back rank men (prevents enemy promotion)
    for c in range(BOARD_SIZE):
        if state.board[7][c] == 'b': s += 0.15
        if state.board[0][c] == 'w': s -= 0.15
    return s if state.turn=='b' else -s

def h_edge_penalty(state: CheckersState) -> float:
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if c in (0,7) or r in (0,7):
                p=state.board[r][c]
                if p=='b': s -= 0.05
                elif p=='w': s += 0.05
    return s if state.turn=='b' else -s

def h_promotion_potential(state: CheckersState) -> float:
    # encourage approaching promotion
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p=='b':
                s += (7-r)*0.03
            elif p=='w':
                s -= r*0.03
    return s if state.turn=='b' else -s

def h_piece_square(state: CheckersState) -> float:
    # simple piece-square values that favor central advancement
    table = [
        [0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1],
        [0.1, 0, 0.15, 0, 0.15, 0, 0.15, 0],
        [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2],
        [0.2, 0, 0.25,0, 0.25,0, 0.25,0],
        [0, 0.25,0, 0.25,0, 0.25,0, 0.25],
        [0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0],
        [0, 0.15,0, 0.15,0, 0.15,0, 0.15],
        [0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0],
    ]
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p.lower()=='b': s += table[r][c]
            elif p.lower()=='w': s -= table[r][c]
    return s if state.turn=='b' else -s

def h_trapped_penalty(state: CheckersState) -> float:
    # penalize pieces with no legal moves individually
    s = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = state.board[r][c]
            if p=='.': continue
            if p.lower()=='b':
                state.turn='b'
                has=False
                for m in state._quiets_from((r,c))+state._captures_from((r,c)):
                    has=True; break
                if not has: s -= 0.1
            else:
                state.turn='w'
                has=False
                for m in state._quiets_from((r,c))+state._captures_from((r,c)):
                    has=True; break
                if not has: s += 0.1
    return s if state.turn=='b' else -s

def h_safe_moves(state: CheckersState) -> float:
    # moves that are not immediately recaptured (rough estimate by counting)
    turn = state.turn
    s = 0.0
    for side in ('b','w'):
        state.turn = side
        for m in state.legal_moves():
            nxt = state.apply_move(m)
            opp = 'w' if side=='b' else 'b'
            nxt.turn = opp
            if len([x for x in nxt.legal_moves() if x.is_capture])==0:
                s += 0.03 if side=='b' else -0.03
    state.turn = turn
    return s if turn=='b' else -s

def h_connectivity(state: CheckersState) -> float:
    # bonus for adjacent friendly neighbors
    s=0.0
    nbrs=[(-1,-1),(-1,1),(1,-1),(1,1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p=state.board[r][c]
            if p=='.': continue
            cnt=0
            for dr,dc in nbrs:
                rr,cc=r+dr,c+dc
                if 0<=rr<BOARD_SIZE and 0<=cc<BOARD_SIZE and state.board[rr][cc] != '.' and state.board[rr][cc].lower()==p.lower():
                    cnt+=1
            if p.lower()=='b': s+=cnt*0.02
            else: s-=cnt*0.02
    return s if state.turn=='b' else -s

def h_attack_bias(state: CheckersState) -> float:
    # prefer capture availability
    turn=state.turn
    state.turn='b'; bcaps = len([m for m in state.legal_moves() if m.is_capture])
    state.turn='w'; wcaps = len([m for m in state.legal_moves() if m.is_capture])
    state.turn=turn
    s = (bcaps - wcaps)*0.1
    return s if turn=='b' else -s

def h_back_two_rows(state: CheckersState) -> float:
    # extra bonus for maintaining two back defensive rows
    s=0.0
    for r in (6,7):
        for c in range(BOARD_SIZE):
            if state.board[r][c]=='b': s += 0.05
    for r in (0,1):
        for c in range(BOARD_SIZE):
            if state.board[r][c]=='w': s -= 0.05
    return s if state.turn=='b' else -s

def h_temporal(state: CheckersState) -> float:
    # small random-ish tempo via mobility difference and piece count
    turn=state.turn
    state.turn='b'; bm=len(state.legal_moves())
    state.turn='w'; wm=len(state.legal_moves())
    state.turn=turn
    bcnt = sum(p.lower()=='b' for row in state.board for p in row)
    wcnt = sum(p.lower()=='w' for row in state.board for p in row)
    s = 0.01*(bm-wm) + 0.02*(bcnt-wcnt)
    return s if turn=='b' else -s

def h_king_distance(state: CheckersState) -> float:
    # favor proximity to becoming king
    s=0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p=state.board[r][c]
            if p=='b': s += (7-r)*0.02
            elif p=='w': s -= r*0.02
    return s if state.turn=='b' else -s

def h_mix_strong(state: CheckersState) -> float:
    # composite of several above, tuned a bit stronger
    return (h_material_advancement(state) + h_piece_square(state) +
            0.5*h_mobility(state) + 0.5*h_center_control(state) +
            0.5*h_attack_bias(state))

# Registry of 15 heuristics
HEURISTICS: dict[str, Heuristic] = {
    "material": h_material,
    "material_adv": h_material_advancement,
    "mobility": h_mobility,
    "center": h_center_control,
    "back_rank": h_back_rank_guard,
    "edge_penalty": h_edge_penalty,
    "promotion_potential": h_promotion_potential,
    "piece_square": h_piece_square,
    "trapped_penalty": h_trapped_penalty,
    "safe_moves": h_safe_moves,
    "connectivity": h_connectivity,
    "attack_bias": h_attack_bias,
    "back_two_rows": h_back_two_rows,
    "temporal": h_temporal,
    "king_distance": h_king_distance,
    "mix_strong": h_mix_strong,
}
