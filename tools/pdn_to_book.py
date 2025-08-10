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
    parts = re.split(r'(?=\[Event\s*")', pdn_text.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]

def extract_movetext(block: str) -> str:
    # strip headers
    lines = [ln for ln in block.splitlines() if not HEADER_RE.match(ln)]
    text = " ".join(lines)
    # strip comments and variations
    text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r';[^\n]*', ' ', text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    return " ".join(text.split())

def scan_move_tokens(movetext: str):
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
    """Dark squares on (r+c)%2 == 1; row0 dark columns 1,3,5,7."""
    n -= 1
    r = n // 4
    k = n % 4
    c = (1 + 2*k) if (r % 2 == 0) else (0 + 2*k)
    return r, c

def n_to_rc_v2(n: int) -> tuple[int,int]:
    """Dark squares on (r+c)%2 == 0; row0 dark columns 0,2,4,6."""
    n -= 1
    r = n // 4
    k = n % 4
    c = (0 + 2*k) if (r % 2 == 0) else (1 + 2*k)
    return r, c

def token_to_path(token: str, mapper) -> list[tuple[int,int]]:
    return [mapper(int(x)) for x in re.split(r'[-x]', token)]

# ----- Two start orientations -----
def start_state_black_top() -> CheckersState:
    """Black pieces 1-12 on top (rows 0..2), White bottom, Black to move."""
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
        movetext = extract_movetext(block)
        tokens = scan_move_tokens(movetext)
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
