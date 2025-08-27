# checkers/core/pdn.py (save-only)

from __future__ import annotations
import os
import re
from typing import List, Optional

from checkers.core.move import Move

# ---------- PDN SAVE ----------

INVALID_CHARS = r'[<>:"/\\|?*]'  # Windows-forbidden chars (also safe to sanitize on Unix)


def ensure_games_dir(path: str = "data/games") -> str:
    """Ensure the games directory exists (no-op if it already does)."""
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    """Filesystem-safe filename (replace invalid chars; trim trailing dot/space)."""
    s = re.sub(INVALID_CHARS, "_", name)
    s = s.strip().rstrip(". ")
    return s or "untitled"


def _result_tag(result: str) -> str:
    """Map PDN result to a short tag for filenames."""
    mapping = {"1-0": "1-0", "0-1": "0-1", "1/2-1/2": "draw"}
    return mapping.get(result, "ongoing")


def save_game_pdn(
    moves: List[Move],
    result: str = "*",
    white: str = "Human",
    black: str = "AI",
    out_dir: str = "data/games",
    filename: Optional[str] = None,
) -> str:
    """Save a game in PDN format and return the file path."""
    from datetime import datetime

    ensure_games_dir(out_dir)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y%m%d-%H%M%S")

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


