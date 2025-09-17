from __future__ import annotations
import json, os

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


load_book()
