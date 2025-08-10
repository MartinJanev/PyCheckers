from __future__ import annotations
import json, os
from .state import CheckersState
from .move import Move

BOOK_FILE = os.path.join(os.path.dirname(__file__), "book.json")

# Map: FEN_current -> set(FEN_next,...)
TRANSITIONS: dict[str, set[str]] = {}

def load_book():
    global TRANSITIONS
    if not os.path.exists(BOOK_FILE):
        TRANSITIONS = {}
        return
    with open(BOOK_FILE, "r") as f:
        raw = json.load(f)  # fen -> [fen_next, ...]
    TRANSITIONS = {k: set(v) for k, v in raw.items()}

def book_moves_for(state: CheckersState) -> list[Move]:
    if not TRANSITIONS:
        return []
    fen = state.to_fen()
    next_fens = TRANSITIONS.get(fen)
    if not next_fens:
        return []
    res = []
    for m in state.legal_moves():
        nxt = state.apply_move(m)
        if nxt.to_fen() in next_fens:
            res.append(m)
    return res

load_book()
