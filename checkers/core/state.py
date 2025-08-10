from __future__ import annotations
from typing import List, Tuple, Optional
from .constants import BLACK, WHITE, BLACK_K, WHITE_K, EMPTY, DIRS_BLACK, DIRS_WHITE, DIRS_KING, BOARD_SIZE
from .move import Move, MoveSeq, Coord
from .util import in_bounds

class CheckersState:
    def __init__(self, board: Optional[List[List[str]]] = None, turn: str = BLACK):
        if board is None:
            board = self._initial_board()
        self.board = board
        self.turn = turn  # 'b' or 'w'

    @staticmethod
    def _initial_board() -> List[List[str]]:
        b = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        # White on top (rows 0-2), Black on bottom (rows 5-7)
        for r in range(3):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    b[r][c] = WHITE
        for r in range(5, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    b[r][c] = BLACK
        return b

    def clone(self) -> 'CheckersState':
        return CheckersState([row[:] for row in self.board], self.turn)

    def piece_at(self, rc: Coord) -> str:
        r, c = rc
        return self.board[r][c]

    def set_piece(self, rc: Coord, p: str):
        r, c = rc
        self.board[r][c] = p

    def opposite(self, p: str) -> str:
        return WHITE if p.lower() == 'b' else BLACK

    def is_king(self, p: str) -> bool:
        return p in (BLACK_K, WHITE_K)

    def directions_for(self, p: str):
        if p == WHITE:
            return DIRS_WHITE
        if p == BLACK:
            return DIRS_BLACK
        return DIRS_KING

    def current_side_pieces(self):
        res = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p != EMPTY and p.lower() == self.turn:
                    res.append((r, c))
        return res

    def legal_moves(self) -> list[Move]:
        captures = []
        quiets = []
        for rc in self.current_side_pieces():
            caps = self._captures_from(rc)
            if caps:
                captures.extend(caps)
            else:
                quiets.extend(self._quiets_from(rc))
        return captures if captures else quiets

    def _quiets_from(self, rc: Coord) -> list[Move]:
        r, c = rc
        p = self.piece_at(rc)
        dirs = DIRS_KING if self.is_king(p) else self.directions_for(p)
        moves = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and self.board[nr][nc] == EMPTY:
                moves.append(Move([rc, (nr, nc)], False))
        return moves

    def _captures_from(self, rc: Coord) -> list[Move]:
        r, c = rc
        p = self.piece_at(rc)
        dirs = DIRS_KING if self.is_king(p) else self.directions_for(p)
        results: list[Move] = []

        def dfs(path: MoveSeq, board, piece: str, promoted: bool):
            nonlocal results
            r, c = path[-1]
            found = False
            cur_dirs = DIRS_KING if (promoted or piece in (BLACK_K, WHITE_K)) else (DIRS_KING if piece in (BLACK_K, WHITE_K) else self.directions_for(piece))
            for dr, dc in cur_dirs:
                mr, mc = r + dr, c + dc
                jr, jc = r + 2*dr, c + 2*dc
                if in_bounds(jr, jc) and in_bounds(mr, mc):
                    mid = board[mr][mc]
                    if mid != EMPTY and mid.lower() == self.opposite(piece) and board[jr][jc] == EMPTY:
                        found = True
                        saved_from = board[r][c]
                        saved_mid = board[mr][mc]
                        board[r][c] = EMPTY
                        board[mr][mc] = EMPTY
                        landed_piece = piece
                        became_king = False
                        # Land
                        board[jr][jc] = landed_piece
                        # Promotion when landing on back rank
                        if piece == BLACK and jr == 0:
                            board[jr][jc] = BLACK_K
                            became_king = True
                        elif piece == WHITE and jr == 7:
                            board[jr][jc] = WHITE_K
                            became_king = True

                        dfs(path + [(jr, jc)], board, board[jr][jc], promoted or became_king)

                        board[r][c] = saved_from
                        board[mr][mc] = saved_mid
                        board[jr][jc] = EMPTY
            if not found and len(path) > 1:
                results.append(Move(path, True))

        temp = [row[:] for row in self.board]
        dfs([rc], temp, p, self.is_king(p))
        return results

    def apply_move(self, mv: Move) -> 'CheckersState':
        s = self.clone()
        path = mv.path
        start = path[0]
        piece = s.piece_at(start)
        s.set_piece(start, EMPTY)
        for i in range(1, len(path)):
            r0, c0 = path[i-1]
            r1, c1 = path[i]
            if abs(r1 - r0) == 2:
                mr, mc = (r0 + r1)//2, (c0 + c1)//2
                s.set_piece((mr, mc), EMPTY)
        end = path[-1]
        er, ec = end
        if piece == BLACK and er == 0:
            piece = BLACK_K
        elif piece == WHITE and er == 7:
            piece = WHITE_K
        s.set_piece(end, piece)
        s.turn = 'w' if self.turn == 'b' else 'b'
        return s

    def terminal(self) -> Optional[str]:
        b_cnt = sum(self.board[r][c].lower() == 'b' for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        w_cnt = sum(self.board[r][c].lower() == 'w' for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        if b_cnt == 0: return 'w'
        if w_cnt == 0: return 'b'
        if not self.legal_moves():
            return 'w' if self.turn == 'b' else 'b'
        return None

    def evaluate(self) -> float:
        score = 0.0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p == 'b':
                    score += 1.0 + (7 - r) * 0.03
                elif p == 'w':
                    score -= 1.0 + r * 0.03
                elif p == 'B':
                    score += 1.8
                elif p == 'W':
                    score -= 1.8
        # Mobility
        turn_save = self.turn
        self.turn = 'b'
        b_moves = len(self.legal_moves())
        self.turn = 'w'
        w_moves = len(self.legal_moves())
        self.turn = turn_save
        score += 0.02 * (b_moves - w_moves)
        # Perspective: score is for side-to-move
        return score if self.turn == 'b' else -score

    # --- FEN I/O for 8x8 English checkers ---
    def to_fen(self) -> str:
        rows = []
        for r in range(8):
            row = ""
            run = 0
            for c in range(8):
                p = self.board[r][c]
                if p == '.':
                    run += 1
                else:
                    if run: row += str(run); run = 0
                    row += p
            if run: row += str(run)
            rows.append(row if row else "8")
        side = self.turn  # 'b' or 'w'
        return "/".join(rows) + " " + side

    @staticmethod
    def from_fen(fen: str) -> 'CheckersState':
        board_part, side = fen.strip().split()
        rows = board_part.split('/')
        board = []
        for r in range(8):
            row = []
            for ch in rows[r]:
                if ch.isdigit():
                    row.extend(['.'] * int(ch))
                else:
                    row.append(ch)
            while len(row) < 8:
                row.append('.')
            board.append(row[:8])
        return CheckersState(board=board, turn=side)
