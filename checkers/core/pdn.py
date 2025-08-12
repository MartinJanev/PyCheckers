# checkers/core/pdn.py
from __future__ import annotations
import os
import re
from typing import Iterable, List, Optional, Tuple, Dict

from .state import CheckersState
from .move import Move

# ---------- PDN SAVE ----------

INVALID_CHARS = r'[<>:"/\\|?*]'  # Windows-forbidden chars (also fine to sanitize on Unix)


def ensure_games_dir(path: str = "data/games") -> str:
    """
    Ensure the games directory exists, creating it if necessary.
    If the directory already exists, this is a no-op.
    This is useful to ensure the directory structure is ready for saving games.
    :param path: str - The path to the games directory.
    :return: str - The absolute path to the games directory.
    """
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    """
    Convert a string to a filesystem-safe name by replacing invalid characters with underscores.
    This is useful for generating filenames that won't cause issues on different filesystems.
    It also trims trailing dots and spaces, which can cause problems on Windows.
    If the resulting name is empty, it defaults to "untitled".
    :param name: str - The name to sanitize.
    :return: str - A sanitized version of the name that is safe for use as a filename.
    """
    s = re.sub(INVALID_CHARS, "_", name)
    s = s.strip().rstrip(". ")  # Windows dislikes trailing dots/spaces
    return s or "untitled"


def _result_tag(result: str) -> str:
    """
    Convert a game result string to a safe tag for filenames.
    This maps common PDN result strings to a simplified form.
    - "1-0" → "1-0"
    - "0-1" → "0-1"
    - "1/2-1/2" → "draw"
    - "*" or anything else → "ongoing"
    This is useful for generating filenames that reflect the game outcome.
    :param result: str - The game result string (e.g., "1-0", "0-1", "1/2-1/2", "*").
    :return: str - A simplified result tag suitable for filenames.
    This will be "ongoing" for any unrecognized result.
    """
    mapping = {"1-0": "1-0", "0-1": "0-1", "1/2-1/2": "draw"}
    return mapping.get(result, "ongoing")  # "*" or anything else → "ongoing"


def save_game_pdn(
        moves: List[Move],
        result: str = "*",
        white: str = "Human",
        black: str = "AI",
        out_dir: str = "data/games",
        filename: Optional[str] = None,
) -> str:
    """
    Save a game in PDN format to the specified directory.
    :param moves: List[Move] - The list of moves played in the game.
    :param result: str - The game result (e.g., "1-0", "0-1", "1/2-1/2", "*").
    :param white: str - The name of the white player (default "Human").
    :param black: str - The name of the black player (default "AI").
    :param out_dir: str - The directory where the game file will be saved (default "data/games").
    :param filename: Optional[str] - The name of the file to save the game as. If None, a timestamped name is generated.
    :return: str - The path to the saved PDN file.
    This function ensures the output directory exists, generates a filename if not provided,
    and writes the game data in PDN format.
    """
    from datetime import datetime

    ensure_games_dir(out_dir)
    date_str = datetime.now().strftime("%Y-%m-%d")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if filename is None:
        filename = f"{stamp}-{_result_tag(result)}.pdn"
    else:
        filename = _safe_name(filename)

    path = os.path.join(out_dir, filename)

    headers = [
        f'[Event "PyCheckers"]',
        f'[Date "{date_str}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        f'[Result "{result}"]',
    ]

    # Movetext like: 1. 11-15 23-19 2. ...
    move_number = 1
    parts: List[str] = []
    for i, mv in enumerate(moves):
        notation = str(mv)
        if i % 2 == 0:
            parts.append(f"{move_number}. {notation}")
        else:
            parts[-1] += f" {notation}"
            move_number += 1
    movetext = " ".join(parts)
    if result != "*":
        movetext = (movetext + f" {result}").strip()

    with open(path, "w", encoding="utf-8") as f:
        for h in headers:
            f.write(h + "\n")
        f.write("\n")
        f.write(movetext + "\n")

    return path


# ---------- PDN READ ----------

_TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.M)
_MOVE_SPLIT_RE = re.compile(r'\s*\d+\.\s*')  # split on "1. ", "2. ", ...


def parse_pdn(text: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse a PDN text into headers and move tokens.
    This function extracts the headers (like Event, Date, White, Black, Result)
    and the move tokens from the PDN text.
    The headers are returned as a dictionary, and the moves are returned as a list of strings.
    :param text: str - The PDN text to parse.
    :return: Tuple[Dict[str, str], List[str]] - A tuple containing:
        - A dictionary of headers (e.g., {'Event': 'PyCheckers', 'Date': '2023-10-01', ...}).
        - A list of move tokens (e.g., ['11-15', '23-19', '12-16', ...
    This function uses regular expressions to extract the headers and moves.
    """
    headers: Dict[str, str] = {}
    for tag, value in _TAG_RE.findall(text):
        headers[tag] = value

    # strip tags and newlines, then split on move numbers
    body = _TAG_RE.sub("", text).strip()
    chunks = [c.strip() for c in _MOVE_SPLIT_RE.split(body) if c.strip()]
    tokens: List[str] = []
    for ch in chunks:
        # chunk like: "11-15 23-19" or "11-15 23x16"
        parts = ch.split()
        for p in parts:
            # stop at result marker if present
            if p in ("1-0", "0-1", "1/2-1/2", "*"):
                break
            tokens.append(p)
    return headers, tokens


def load_pdn_file(path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Load a PDN file and parse its contents.
    This function reads a PDN file from the specified path and parses it into headers and move
    tokens.
    :param path: str - The path to the PDN file to load.
    :return: Tuple[Dict[str, str], List[str]] - A tuple containing:
        - A dictionary of headers (e.g., {'Event': 'PyCheckers', '
        Date': '2023-10-01', ...}).
        - A list of move tokens (e.g., ['11-15', '23-
        19', '12-16', ...]).
    This function reads the file in UTF-8 encoding and uses the `parse_pdn` function to extract
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_pdn(text)


# ---------- REPLAY ----------

def _match_token_to_move(token: str, legal: Iterable[Move]) -> Optional[Move]:
    """
    Match a move token to a Move object from the legal moves.
    This function checks if the provided token matches any of the legal moves.
    :param token: str - The move token to match (e.g., "11-
    15", "23x19").
    :param legal: Iterable[Move] - An iterable of legal Move objects.
    :return: Optional[Move] - The matching Move object if found, or None if
    no match is found.
    """
    for m in legal:
        if str(m) == token:
            return m
    return None


def replay_pdn(tokens: List[str], start: Optional[CheckersState] = None) -> List[CheckersState]:
    """
    Replay a sequence of moves from a PDN file.
    This function takes a list of move tokens and an optional starting state,
    and returns a list of CheckersState objects representing the game states after each move.
    :param tokens: List[str] - A list of move tokens (e.g., ['
    11-15', '23-19', '12-16', ...]).
    :param start: Optional[CheckersState] - An optional starting state.
    If not provided, it uses the initial state of CheckersState.
    :return: List[CheckersState] - A list of CheckersState objects representing
    the game states after each move.
    This function applies each move token to the starting state and returns the resulting states.
    If a token cannot be matched to a legal move, it stops gracefully.
    """
    if start is not None:
        s = start
    elif hasattr(CheckersState, "initial"):
        s = CheckersState.initial()  # type: ignore[attr-defined]
    else:
        s = CheckersState()

    states = [s]
    for t in tokens:
        mv = _match_token_to_move(t, s.legal_moves())
        if mv is None:
            # Could not match: PDN and rules out of sync; stop gracefully
            break
        s = s.apply_move(mv)
        states.append(s)
    return states


# ---------- UTILITIES ----------

def list_game_files(out_dir: str = "data/games") -> List[str]:
    """
    List all PDN game files in the specified directory, sorted by modification time.
    This function returns a list of PDN files in the specified directory,
    sorted by their last modification time in descending order.
    :param out_dir: str - The directory to search for PDN files (default "
    data/games").
    :return: List[str] - A list of paths to PDN files, sorted by
    modification time (most recent first).
    If the directory does not exist or contains no PDN files, it returns an empty list
    """
    if not os.path.isdir(out_dir):
        return []
    files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".pdn")]
    return sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
