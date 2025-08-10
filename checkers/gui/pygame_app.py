import datetime
import os
import sys


class DummyFile(object):
    def write(self, x): pass


sys.stdout = DummyFile()
import pygame

sys.stdout = sys.__stdout__
from typing import Optional, Tuple
from ..core.state import CheckersState
from ..core.constants import BOARD_SIZE, EMPTY
from ..core.move import Move
from ..core.engine import choose_ai_move, EngineConfig

# --- Layout ---
TILE = 80
BOARD_SIZE = 8
BOARD_W = TILE * BOARD_SIZE
BOARD_H = TILE * BOARD_SIZE

BOARD_MARGIN_X = 20  # around the board (left/right)
BOARD_MARGIN_Y = 20  # around the board (top/bottom)
BOARD_SIDEBAR_GAP = 20  # gap between board and sidebar

SIDEBAR_W = 400  # right panel width
SIDEBAR_PAD = 14  # inner padding for sidebar

# Total window
W = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP + SIDEBAR_W + BOARD_MARGIN_X
H = BOARD_MARGIN_Y + BOARD_H + BOARD_MARGIN_Y

FPS = 120

DARK = (118, 78, 46)
LIGHT = (238, 238, 210)
HIGHLIGHT = (186, 202, 68)
TEXT = (25, 25, 25)


def rc_to_xy(r: int, c: int) -> Tuple[int, int]:
    return BOARD_MARGIN_X + c * TILE, BOARD_MARGIN_Y + r * TILE


def pos_to_rc(x: int, y: int) -> Tuple[int, int]:
    x -= BOARD_MARGIN_X
    y -= BOARD_MARGIN_Y
    if x < 0 or y < 0:
        return -1, -1
    return y // TILE, x // TILE


from ..core.engine import choose_ai_move, EngineConfig

DIFFICULTIES = {
    "Beginner": EngineConfig(algo="random", depth=1, heuristic="material", use_tt=False),
    "Novice": EngineConfig(algo="minimax", depth=3, heuristic="material", use_tt=False),
    "Amateur": EngineConfig(algo="expectimax", depth=3, heuristic="material_adv", use_tt=False),
    "Pro": EngineConfig(algo="minimax", depth=5, heuristic="mix_strong", use_tt=False),
    "Hard": EngineConfig(algo="minimax", depth=7, heuristic="mix_strong", use_tt=True),
}
DEFAULT_DIFF = "Amateur"
ALIASES = {"Easy": "Beginner", "Medium": "Pro", "Hard": "Hard"}


class PygameUI:
    def __init__(self, difficulty: str = "Pro"):
        self.game_result = '*'
        pygame.init()

        # Safe icon load
        try:
            icon_surface = pygame.image.load("checkers/gui/icon.jpg")
            pygame.display.set_icon(icon_surface)
        except Exception:
            pass

        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("PyCheckers")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 26)
        self.big = pygame.font.SysFont(None, 36)

        self.state = CheckersState()
        self.selected: Optional[tuple[int, int]] = None
        self.legal: list[Move] = self.state.legal_moves()
        self.running = True
        self.animating = False
        self.message = ""

        # difficulty (support old names)
        req = difficulty or DEFAULT_DIFF
        name = ALIASES.get(req, req)
        if name not in DIFFICULTIES:
            name = DEFAULT_DIFF
        self.diff_name = name
        self.config = DIFFICULTIES[name]

        self.click_anim = None
        self.history: list[Move] = []
        self.states: list[CheckersState] = []
        self.redo: list[CheckersState] = []
        self.diff_rects: list[tuple[pygame.Rect, str]] = []

        self.human_color = None  # set after side selection

    # -------------------- Main loop --------------------
    def run(self):
        # Step 1: Choose side
        self.show_side_selection()

        # Step 2: Set starting message
        self.message = (
            "You are White. Click a piece, then a destination."
            if self.human_color == "w"
            else "You are Black. Click a piece, then a destination."
        )

        # Step 3: AI moves first if human is black
        if self.human_color == 'b' and self.state.turn == 'w':
            mv = choose_ai_move(self.state, self.config)
            self.apply_and_refresh(mv)
            self.message = f"AI played: {mv}"

        # Step 4: Main loop
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()

            if self.state.turn != self.human_color:
                try:
                    mv = choose_ai_move(self.state, self.config)
                except RuntimeError:
                    self.message = f"Game over — {'you win' if self.state.turn != self.human_color else 'AI wins'}"
                    self.draw();
                    pygame.time.wait(1500)
                    self.game_result = self.compute_result_from_state()
                    self.running = False
                    continue
                self.animate_move(mv)
                self.apply_and_refresh(mv)
                self.message = f"AI played: {mv}"

            self.draw()

        # Save PDN on exit
        self.save_pdn()
        pygame.quit()

    # -------------------- Events --------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: self.set_difficulty("Beginner")
                if event.key == pygame.K_2: self.set_difficulty("Novice")
                if event.key == pygame.K_3: self.set_difficulty("Amateur")
                if event.key == pygame.K_4: self.set_difficulty("Pro")
                if event.key == pygame.K_5: self.set_difficulty("Hard")
                # (keep your undo/redo here)

                # Undo (Ctrl+Z)
                if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    if self.states:
                        self.redo.append(self.state)
                        self.state = self.states.pop()
                        self.legal = self.state.legal_moves()
                        self.selected = None
                        if self.history:
                            self.history.pop()
                        self.message = "Undid last move."

                # Redo (Ctrl+Y)
                if event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    if self.redo:
                        self.states.append(self.state)
                        self.state = self.redo.pop()
                        self.legal = self.state.legal_moves()
                        self.selected = None
                        self.message = "Redid move."

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                sidebar_x = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP
                if x >= sidebar_x:
                    for rect, name in self.diff_rects:
                        if rect.collidepoint(x, y):
                            self.set_difficulty(name)
                            break
                    return  # don't treat as board click

                if self.human_color and self.state.turn == self.human_color:
                    r, c = pos_to_rc(x, y)
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        self.on_click(r, c)

    def show_side_selection(self):
        """Display a screen for the player to choose White or Black."""
        selecting = True
        font_big = pygame.font.SysFont(None, 48)

        while selecting:
            self.screen.fill((30, 30, 30))

            # Title
            title = font_big.render("Choose Your Side", True, (255, 255, 255))
            self.screen.blit(title, (W // 2 - title.get_width() // 2, H // 3))

            # Buttons
            white_rect = pygame.Rect(W // 4 - 75, H // 2, 150, 50)
            black_rect = pygame.Rect(3 * W // 4 - 75, H // 2, 150, 50)

            pygame.draw.rect(self.screen, (240, 240, 240), white_rect, border_radius=8)
            pygame.draw.rect(self.screen, (50, 50, 50), black_rect, border_radius=8)

            white_label = self.font.render("White", True, (0, 0, 0))
            black_label = self.font.render("Black", True, (255, 255, 255))
            self.screen.blit(white_label, (white_rect.centerx - white_label.get_width() // 2,
                                           white_rect.centery - white_label.get_height() // 2))
            self.screen.blit(black_label, (black_rect.centerx - black_label.get_width() // 2,
                                           black_rect.centery - black_label.get_height() // 2))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if white_rect.collidepoint(x, y):
                        self.human_color = 'w'
                        selecting = False
                    elif black_rect.collidepoint(x, y):
                        self.human_color = 'b'
                        selecting = False

    # -------------------- Game ops --------------------
    def set_difficulty(self, name: str):
        if name in DIFFICULTIES:
            self.diff_name = name
            self.config = DIFFICULTIES[name]
            self.message = f"Difficulty set to {name}."

    def on_click(self, r: int, c: int):
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return
        p = self.state.board[r][c]

        # first click: select one of *your* pieces
        if self.selected is None:
            if p != EMPTY and self.human_color and p.lower() == self.human_color:
                self.selected = (r, c)
                # (optional) start click ripple
                self.click_anim = (r, c, pygame.time.get_ticks())
            return

        # clicking another of *your* pieces switches selection
        if p != EMPTY and self.human_color and p.lower() == self.human_color:
            self.selected = (r, c)
            # (optional) restart ripple
            self.click_anim = (r, c, pygame.time.get_ticks())
            return

        # otherwise it's a destination; try to match a legal move
        path = [self.selected, (r, c)]
        mv = self.find_matching_move(path)
        if mv is None:
            self.message = "Invalid move. Try again."
            return

        # hide hints during animation
        self.selected = None
        self.animating = True
        self.animate_move(mv)
        self.animating = False

        self.apply_and_refresh(mv)

    def find_matching_move(self, path: list[tuple[int, int]]) -> Optional[Move]:
        for m in self.legal:
            if m.path == path:
                return m
        cand = [m for m in self.legal if m.path[:len(path)] == path]
        if len(cand) == 1:
            return cand[0]
        return None

    def apply_and_refresh(self, mv):
        self.states.append(self.state)
        self.history.append(mv)  # store move
        self.redo.clear()
        self.state = self.state.apply_move(mv)
        self.legal = self.state.legal_moves()

    # -------------------- Animation --------------------
    def animate_move(self, mv: Move, duration: float = 0.25):
        if not mv.path or len(mv.path) < 2:
            return
        sr, sc = mv.path[0]
        piece = self.state.board[sr][sc]

        for i in range(1, len(mv.path)):
            r0, c0 = mv.path[i - 1]
            r1, c1 = mv.path[i]
            x0, y0 = rc_to_xy(r0, c0)
            x1, y1 = rc_to_xy(r1, c1)

            frames = max(1, int(FPS * duration))
            for f in range(frames):
                t = (f + 1) / frames
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))

                # Draw full board but temporarily remove moving piece from its start square
                saved_piece = self.state.board[r0][c0]
                self.state.board[r0][c0] = EMPTY
                self.draw()
                self.state.board[r0][c0] = saved_piece

                # Draw moving piece on top
                center = (x + TILE // 2, y + TILE // 2)
                radius = TILE // 2 - 8
                fill = (30, 30, 30) if piece.lower() == 'b' else (240, 240, 240)
                outline = (230, 230, 230) if piece.lower() == 'b' else (40, 40, 40)
                pygame.draw.circle(self.screen, fill, center, radius)
                pygame.draw.circle(self.screen, outline, center, radius, 5)
                if piece in ('B', 'W'):
                    pygame.draw.circle(self.screen, outline, center, radius // 2, 5)

                pygame.display.flip()
                self.clock.tick(FPS)

    # -------------------- Rendering --------------------
    def draw(self, base_only: bool = False):
        self.screen.fill((30, 30, 30))

        # --- Board ---
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = DARK if (r + c) % 2 else LIGHT
                x, y = rc_to_xy(r, c)
                pygame.draw.rect(self.screen, color, (x, y, TILE, TILE))

        # Click ripple
        if self.click_anim:
            r_anim, c_anim, start_time = self.click_anim
            elapsed = pygame.time.get_ticks() - start_time
            if elapsed < 300:
                progress = elapsed / 300
                cx = BOARD_MARGIN_X + c_anim * TILE + TILE // 2
                cy = BOARD_MARGIN_Y + r_anim * TILE + TILE // 2
                radius = int((TILE // 2 - 6) * progress)
                alpha = max(0, 255 - int(255 * progress))
                surf = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
                pygame.draw.circle(surf, (180, 180, 180, alpha), (TILE // 2, TILE // 2), radius, 2)
                self.screen.blit(surf, (BOARD_MARGIN_X + c_anim * TILE, BOARD_MARGIN_Y + r_anim * TILE))
            else:
                self.click_anim = None

        if not base_only:
            # selection + targets
            if self.selected is not None and not self.animating:
                sr, sc = self.selected
                sx, sy = rc_to_xy(sr, sc)
                pygame.draw.rect(self.screen, HIGHLIGHT, (sx, sy, TILE, TILE), 4)
                for m in self.legal:
                    if m.path[0] == self.selected:
                        r2, c2 = m.path[1]
                        cx = BOARD_MARGIN_X + c2 * TILE + TILE // 2
                        cy = BOARD_MARGIN_Y + r2 * TILE + TILE // 2
                        pygame.draw.circle(self.screen, (180, 180, 180), (cx, cy), TILE // 4, 2)

            # pieces
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    p = self.state.board[r][c]
                    if p == EMPTY:
                        continue
                    x = BOARD_MARGIN_X + c * TILE + TILE // 2
                    y = BOARD_MARGIN_Y + r * TILE + TILE // 2
                    radius = TILE // 2 - 8
                    fill = (30, 30, 30) if p.lower() == 'b' else (240, 240, 240)
                    outline = (230, 230, 230) if p.lower() == 'b' else (40, 40, 40)
                    pygame.draw.circle(self.screen, fill, (x, y), radius)
                    pygame.draw.circle(self.screen, outline, (x, y), radius, 5)
                    if p in ('B', 'W'):
                        pygame.draw.circle(self.screen, outline, (x, y), radius // 2, 5)

        # --- Sidebar ---
        sidebar_x = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP
        pygame.draw.rect(self.screen, (245, 245, 245), (sidebar_x, 0, SIDEBAR_W, H))

        y = SIDEBAR_PAD
        # title
        turn_is_human = (self.state.turn == self.human_color)
        title = self.big.render("Turn: You" if turn_is_human else "Turn: AI", True, TEXT)
        self.screen.blit(title, (sidebar_x + SIDEBAR_PAD, y))
        y += title.get_height() + SIDEBAR_PAD

        # buttons
        self.diff_rects.clear()
        levels = ["Beginner", "Novice", "Amateur", "Pro", "Hard"]
        btn_h = 34
        for name in levels:
            rect = pygame.Rect(
                sidebar_x + SIDEBAR_PAD,
                y,
                SIDEBAR_W - 2 * SIDEBAR_PAD,
                btn_h
            )
            self.diff_rects.append((rect, name))
            color = (200, 200, 200) if name != self.diff_name else (150, 200, 150)
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            label = self.font.render(name, True, (20, 20, 20))
            self.screen.blit(label, (rect.x + 10, rect.y + 7))
            y += btn_h + 8

        y += 6
        # message
        msg = self.font.render(self.message or "", True, TEXT)
        self.screen.blit(msg, (sidebar_x + SIDEBAR_PAD, y))
        y += msg.get_height() + 10

        # history
        hist = "  ".join(str(m) for m in self.history[-8:])
        hsurf = self.font.render(f"Moves: {hist}", True, TEXT)
        self.screen.blit(hsurf, (sidebar_x + SIDEBAR_PAD, y))

        pygame.display.flip()

    def save_pdn(self, filename=None):
        """Save the current game's move history to a PDN file, appending with spacing."""
        if not self.history:
            return  # nothing to save

        if filename is None:
            filename = "data/games.pdn"

        # Make sure folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Determine PDN result
        result = self.game_result if self.game_result in {"1-0", "0-1",
                                                          "1/2-1/2"} else self.compute_result_from_state()

        headers = [
            f'[Event "Casual Game between Human and AI"]',
            f'[Date "{datetime.datetime.now().strftime("%Y-%m-%d")}"]',
            f'[White "{"Human" if self.human_color == "w" else "AI"}"]',
            f'[Black "{"Human" if self.human_color == "b" else "AI"}"]',
            f'[Result "{result}"]'
        ]

        # Build movetext: 1. 11-15 23-19 2. ...
        move_number = 1
        moves_str = []
        for i, mv in enumerate(self.history):
            notation = str(mv)
            if i % 2 == 0:
                moves_str.append(f"{move_number}. {notation}")
            else:
                moves_str[-1] += f" {notation}"
                move_number += 1
        pdn_moves = " ".join(moves_str)

        with open(filename, "a", encoding="utf-8") as f:
            for h in headers:
                f.write(h + "\n")
            f.write("\n")
            f.write(pdn_moves + f" {result}\n\n\n")  # two blank lines after each game

        print(f"Game saved to {filename}")

    def compute_result_from_state(self) -> str:
        """Return PDN result tag based on terminal state."""
        winner = self.state.terminal()  # returns 'w', 'b', or None
        if winner == 'w':
            return "1-0"
        if winner == 'b':
            return "0-1"
        if winner == 'draw':
            return "1/2-1/2"
        return "*"
