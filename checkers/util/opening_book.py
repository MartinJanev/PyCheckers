from __future__ import annotations
import json, os
from checkers.core.state import CheckersState
from checkers.core.move import Move

BOOK_FILE = os.path.join(os.path.dirname(__file__), "book.json")

# Map: FEN_current -> set(FEN_next,...)
TRANSITIONS: dict[str, set[str]] = {}


def load_book():
    """
    Load the opening book from the JSON file.
    This function reads the book file and populates the global TRANSITIONS dictionary.
    If the file does not exist, it initializes TRANSITIONS as an empty dictionary.
    This is used to determine legal moves based on the opening book.
    :return:
    """
    global TRANSITIONS
    if not os.path.exists(BOOK_FILE):
        TRANSITIONS = {}
        return
    with open(BOOK_FILE, "r") as f:
        raw = json.load(f)  # fen -> [fen_next, ...]
    TRANSITIONS = {k: set(v) for k, v in raw.items()}


def book_moves_for(state: CheckersState) -> list[Move]:
    """
    Get the legal moves for a given state based on the opening book.
    This function checks the current state against the opening book transitions.
    It returns a list of moves that lead to next states defined in the book.
    If the current state is not in the book, it returns an empty list.
    :param state: CheckersState: The current state of the game.
    :return: list[Move]: A list of legal moves that are defined in the opening book for the current state.
    """
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
