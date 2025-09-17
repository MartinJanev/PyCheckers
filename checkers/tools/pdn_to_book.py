#!/usr/bin/env python3
"""
PDN (English 8x8) -> opening book transitions (FEN_current -> [FEN_next,...])

Usage:
  python tools/pdn_to_book.py input.pdn checkers/core/book.json --plies 12 --debug
"""
import re, json, sys, argparse
from pathlib import Path

from checkers.core.state import CheckersState
from checkers.core.move import Move

HEADER_RE = re.compile(r'^\s*\[([A-Za-z0-9_-]+)\s+"([^"]*)"\]\s*$')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdn_in")
    ap.add_argument("json_out")
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--guess-captures", action="store_true", default=True)
    return ap.parse_args()

def split_games(pdn_text: str):
    """
    Split PDN text into game blocks starting with [Event "..."]
    Each block may contain multiple lines of headers and move text.
    Empty blocks are skipped.
    :param pdn_text: full PDN text
    :return:
    """
    parts = re.split(r'(?=\[Event\s*")', pdn_text.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]

def extract_move_text(block: str) -> str:
    """
    Extract move text from a PDN block, stripping headers, comments, and variations.
    :param block: PDN game block
    :return: move text as a single string
    """
    # strip headers
    lines = [ln for ln in block.splitlines() if not HEADER_RE.match(ln)]
    text = " ".join(lines)
    # strip comments and variations
    text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r';[^\n]*', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    return " ".join(text.split())

def scan_move_tokens(movetext: str):
    """
    Scan move text for tokens like "12-16" or "22x17x10".
    Ignore move numbers, results, and anything not matching the expected pattern.
    1. Split by whitespace.
    2. Ignore tokens ending with '.' (move numbers).
    3. Stop at result tokens like "1-0", "0-1", "1/2-1/2", "*".
    4. Keep tokens matching the pattern of moves.
    :param movetext: move text string
    :return:
    """
    toks = []
    for frag in movetext.split():
        if frag.endswith("."):  # "1."
            continue
        if frag in ("1-0", "0-1", "1/2-1/2", "*"):
            break
        if re.match(r'^\d+[-x]\d+(?:[-x]\d+)*$', frag):
            toks.append(frag)
    return toks

# ----- Two English 1..32 mappings -----
def n_to_rc_v1(n: int) -> tuple[int,int]:
    """
    Convert square number (1..32) to (row,col) on 8x8 board.
    Dark squares are on (r+c)%2 == 1; row 0 has dark columns 1,3,5,7.
    1->(0,1), 2->(0,3), 3->(0,5), 4->(0,7)
    5->(1,0), 6->(1,2), 7->(1,4), 8->(1,6)
    ...
    :param n: square number 1..32
    :return:
    """
    """Dark squares on (r+c)%2 == 1; row0 dark columns 1,3,5,7."""
    n -= 1
    r = n // 4
    k = n % 4
    c = (1 + 2*k) if (r % 2 == 0) else (0 + 2*k)
    return r, c

def n_to_rc_v2(n: int) -> tuple[int,int]:
    """
    Convert square number (1..32) to (row,col) on 8x8 board.
    Dark squares are on (r+c)%2 == 1; row 0 has dark
    columns 0,2,4,6.
    1->(0,0), 2->(0,2), 3
    ->(0,4), 4->(0,6)
    5->(1,1), 6->(1,3), 7
    ...
    :param n: square number 1..32
    :return:
    """
    n -= 1
    r = n // 4
    k = n % 4
    c = (0 + 2*k) if (r % 2 == 0) else (1 + 2*k)
    return r, c

def token_to_path(token: str, mapper) -> list[tuple[int,int]]:
    """
    Convert a move token like "12-16" or "22x17x10" to a list of (row,col) tuples.
    :param token: move token string
    :param mapper: function to convert square number to (row,col)
    :return: list of (row,col) tuples
    """
    return [mapper(int(x)) for x in re.split(r'[-x]', token)]

# ----- Two start orientations -----
def start_state_black_top() -> CheckersState:
    """
    Black pieces 1-12 on top (rows 0..2), White bottom, Black to move.
    This is the opposite of CheckersState() default.
    :return:
    """
    s = CheckersState()
    # swap colors from your engine's default if needed
    for r in range(8):
        for c in range(8):
            p = s.board[r][c]
            if p == 'w': s.board[r][c] = 'b'
            elif p == 'b': s.board[r][c] = 'w'
            elif p == 'W': s.board[r][c] = 'B'
            elif p == 'B': s.board[r][c] = 'W'
    s.turn = 'b'
    return s

def start_state_white_top() -> CheckersState:
    """White pieces 1-12 on top, Black bottom, White to move."""
    s = CheckersState()
    s.turn = 'w'
    return s

def find_matching_move(state: CheckersState, path, is_capture: bool, guess_ok: bool) -> Move | None:
    """
    Find a legal move in the current state matching the given path and capture flag.
    If multiple captures match the start and end squares, and guess_ok is True,
    return the longest path (most jumps).
    Otherwise, return None if no exact match is found.
    1. Exact match: is_capture and path must match exactly.
    2. For captures, if no exact match, match start and end squares; if multiple,
       return the longest if guess_ok.
    3. For non-captures, only exact matches of two-square paths are considered.
    4. Return None if no match is found.
    5. This function does not modify the state.
    6. The caller should handle the case where no match is found.
    7. This function assumes the state is valid and has legal moves.
    :param state: current CheckersState
    :param path: list of (row,col) tuples representing the move path
    :param is_capture: whether the move is a capture
    :param guess_ok: whether to guess among multiple captures
    :return: matching Move or None
    """
    legal = state.legal_moves()

    # exact path
    for m in legal:
        if m.is_capture == is_capture and m.path == path:
            return m

    if is_capture:
        start, end = path[0], path[-1]
        cand = [m for m in legal if m.is_capture and m.path[0] == start and m.path[-1] == end]
        if len(cand) == 1:
            return cand[0]
        if guess_ok and len(cand) > 1:
            cand.sort(key=lambda m: len(m.path), reverse=True)
            return cand[0]

    if not is_capture and len(path) == 2:
        start, end = path
        for m in legal:
            if not m.is_capture and m.path[0] == start and m.path[-1] == end:
                return m
    return None

def flip_fen_orientation(fen: str) -> str:
    """Flip rows and swap colors; also swap side to move."""
    board_part, turn = fen.strip().split()
    rows = board_part.split('/')[::-1]
    out_rows = []
    for row in rows:
        buf = []
        for ch in row:
            if ch == 'b': buf.append('w')
            elif ch == 'w': buf.append('b')
            elif ch == 'B': buf.append('W')
            elif ch == 'W': buf.append('B')
            else: buf.append(ch)
        out_rows.append("".join(buf))
    turn = 'w' if turn == 'b' else 'b'
    return "/".join(out_rows) + " " + turn

def try_configs_for_first_token(tok: str):
    """Return (start_state_factory, mapper, maybe_flip_fen) or None."""
    configs = [
        (start_state_black_top, n_to_rc_v1, True),
        (start_state_black_top, n_to_rc_v2, True),
        (start_state_white_top, n_to_rc_v1, False),
        (start_state_white_top, n_to_rc_v2, False),
    ]
    for start_fn, mapper, need_flip in configs:
        st = start_fn()
        path = token_to_path(tok, mapper)
        mv = find_matching_move(st, path, 'x' in tok, True)
        if mv is not None:
            return start_fn, mapper, need_flip
    return None

def main():
    args = parse_args()
    pdn_text = Path(args.pdn_in).read_text(encoding="utf-8")
    blocks = split_games(pdn_text)

    trans: dict[str,set[str]] = {}
    ok = skipped = added = 0

    for block in blocks:
        move_text = extract_move_text(block)
        tokens = scan_move_tokens(move_text)
        if not tokens:
            skipped += 1
            if args.debug: print("Skip: no tokens")
            continue

        probe = try_configs_for_first_token(tokens[0])
        if not probe:
            skipped += 1
            if args.debug: print(f"Stop game at ply 1: no legal match for token '{tokens[0]}' (all configs)")
            continue
        start_fn, mapper, need_flip = probe

        state = start_fn()
        plimit = min(args.plies, len(tokens))
        progressed = False

        for i in range(plimit):
            fen_cur = state.to_fen()
            tok = tokens[i]
            path = token_to_path(tok, mapper)
            mv = find_matching_move(state, path, 'x' in tok, args.guess_captures)
            if mv is None:
                if args.debug:
                    print(f"Stop game at ply {i+1}: no legal match for token '{tok}'")
                break
            next_state = state.apply_move(mv)
            fen_next = next_state.to_fen()

            # If we started from Black-on-top mapping, flip to engine's canonical orientation
            if need_flip:
                fen_cur = flip_fen_orientation(fen_cur)
                fen_next = flip_fen_orientation(fen_next)

            trans.setdefault(fen_cur, set()).add(fen_next)
            added += 1
            progressed = True
            state = next_state

        ok += 1 if progressed else 0
        skipped += 0 if progressed else 1

    # finalize
    out = {k: sorted(v) for k, v in trans.items()}
    Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Parsed games: ok={ok}, skipped={skipped}.")
    print(f"Saved {len(out)} positions ({sum(len(v) for v in out.values())} transitions) to {args.json_out}")

if __name__ == "__main__":
    main()
