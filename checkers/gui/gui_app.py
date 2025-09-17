import math
import sys
from typing import Optional, Tuple

from ..core.state import CheckersState
from checkers.util.constants import BOARD_SIZE, EMPTY
from ..core.move import Move
from checkers.util.pdn import save_game_pdn
from ..core.engine import choose_ai_move, EngineConfig
from checkers.util.rules import DrawRulesTracker, DrawRulesConfig


class DummyFile(object):
    def write(self, x):
        pass


# Silence pygame's import-time stdout spam
sys.stdout = DummyFile()
import pygame  # noqa: E402

sys.stdout = sys.__stdout__

# --- Layout ---
TILE = 80
BOARD_W = TILE * BOARD_SIZE
BOARD_H = TILE * BOARD_SIZE

BOARD_MARGIN_X = 40
BOARD_MARGIN_Y = 40
BOARD_SIDEBAR_GAP = 40

SIDEBAR_W = 520
SIDEBAR_PAD = 14

# Total window
W = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP + SIDEBAR_W + BOARD_MARGIN_X

H = BOARD_MARGIN_Y + BOARD_H + BOARD_MARGIN_Y

FPS = 120

DARK = (118, 78, 46)
LIGHT = (238, 238, 210)
HIGHLIGHT = (186, 202, 68)
TEXT = (25, 25, 25)
TEXT_MUTED = (90, 90, 90)
ACCENT = (150, 200, 150)
BTN = (200, 200, 200)
BORDER = (210, 210, 210)


def rc_to_xy(r: int, c: int) -> Tuple[int, int]:
    """
    Convert board row/column to pixel x/y coordinates (top-left of square)
    :param r: Value of row
    :param c: Value of column
    :return: (x, y) pixel coordinates
    """
    return BOARD_MARGIN_X + c * TILE, BOARD_MARGIN_Y + r * TILE


def pos_to_rc(x: int, y: int) -> Tuple[int, int]:
    """
    Convert pixel x/y coordinates to board row/column
    0,0 is top-left of board; returns (-1,-1) if outside board area
    0 <= r,c < BOARD_SIZE if on board
    0 <= x < W, 0 <= y < H
    :param x: Value of x
    :param y: Value of y
    :return: (r, c) board coordinates
    """
    x -= BOARD_MARGIN_X
    y -= BOARD_MARGIN_Y
    if x < 0 or y < 0:
        return -1, -1
    return y // TILE, x // TILE


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """
    Simple word-wrap that breaks text into multiple lines to fit within max_width
    0-width space characters are treated as normal spaces
    1. Splits text into words by whitespace
    2. Iteratively adds words to the current line until adding another word would exceed max
    3. Starts a new line when the current line is full
    4. Returns a list of lines
    5. If a single word is longer than max_width, it will be placed on its own line (may overflow)
    6. Does not hyphenate or split words
    :param text: Input text to wrap
    :param font: Font used to measure text width
    :param max_width: Maximum width in pixels
    :return: List of lines
    """
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


DIFFICULTIES = {
    "Beginner": EngineConfig(algo="random", depth=1, heuristic="material", use_tt=False),
    "Novice": EngineConfig(algo="minimax", depth=3, heuristic="material", use_tt=False),
    "Amateur": EngineConfig(algo="expectimax", depth=3, heuristic="material_adv", use_tt=False),
    "Pro": EngineConfig(algo="minimax", depth=5, heuristic="attack_bias", use_tt=False),
    "Hard": EngineConfig(algo="minimax", depth=7, heuristic="mobility", use_tt=True),
}

DEFAULT_DIFF = "Amateur"
ALIASES = {"Easy": "Beginner", "Medium": "Pro", "Hard": "Hard"}


def _clone_state(s: CheckersState) -> CheckersState:
    """
    Clone a CheckersState object safely.
    Uses .copy() or .clone() if available; otherwise falls back to deepcopy.
    0. Check if the object has a 'copy' method and use it if available
    1. If not, check for a 'clone' method and use it if available
    :param s: CheckersState to clone
    :return: Cloned CheckersState
    """
    if hasattr(s, "copy"):
        return s.copy()
    if hasattr(s, "clone"):
        return s.clone()
    import copy as _copy
    return _copy.deepcopy(s)


class PygameUI:
    def __init__(self, difficulty: str = "Amateur"):
        # Core state / flags
        self.message: str = ""
        self.animating: bool = False
        self.running: bool = True
        self.state: CheckersState = CheckersState()
        self.legal: list[Move] = self.state.legal_moves()

        self.rules: DrawRulesTracker = DrawRulesTracker(DrawRulesConfig(
            no_capture_plies_threshold=80,  # 40 full moves = 80 plies
            repetition_threshold=3
        ))
        self.rules.start(self.state)

        self.game_result: str = '*'
        pygame.init()

        # Safe icon load
        try:
            icon_surface = pygame.image.load("checkers/gui/icon.jpg")
            pygame.display.set_icon(icon_surface)
        except Exception:
            pass

        self.screen: pygame.Surface = pygame.display.set_mode((W, H))
        pygame.display.set_caption("PyCheckers")

        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.font: pygame.font.Font = pygame.font.SysFont(None, 26)
        self.small: pygame.font.Font = pygame.font.SysFont(None, 22)
        self.big: pygame.font.Font = pygame.font.SysFont(None, 36)

        # difficulty (support old names)
        req = difficulty or DEFAULT_DIFF
        name = ALIASES.get(req, req)
        if name not in DIFFICULTIES:
            name = DEFAULT_DIFF
        self.diff_name: str = name
        self.config: EngineConfig = DIFFICULTIES[name]

        # UI state
        self.selected: Optional[tuple[int, int]] = None
        self.click_anim: Optional[tuple[int, int, int]] = None
        self.history: list[Move] = []
        self.diff_rects: list[tuple[pygame.Rect, str]] = []
        self.undo_stack: list[CheckersState] = []
        self.redo_stack: list[CheckersState] = []
        self.new_game_rect: Optional[pygame.Rect] = None  # clickable "New Game" button area

        self.human_color: Optional[str] = None  # 'w' or 'b'; set after side selection

    # -------------------- Main loop --------------------

    def run(self):
        """
        Main application loop.
        Handles events, updates game state, and renders the UI.
        1. Show side selection screen
        2. If human plays Black, let AI make the first move automatically
        3. Main loop:
           - Handle events (mouse clicks, key presses)
           - If it's AI's turn, compute and apply AI move with animation
              - Draw the current game state
        4. On exit, save the game to a PDN file and quit pygame
        :return:
        """
        self.show_side_selection()

        self.message = (
            "You are White. Click a piece, then a destination."
            if self.human_color == "w"
            else "You are Black. Click a piece, then a destination."
        )

        if self.human_color == 'b' and self.state.turn == 'w':
            mv = choose_ai_move(self.state, self.config)
            self.apply_and_refresh(mv)
            self.message = f"AI played: {mv}"

        while self.running:
            self.clock.tick(FPS)
            self.handle_events()

            if self.state.turn != self.human_color:
                try:
                    mv = choose_ai_move(self.state, self.config)
                except RuntimeError:
                    self.message = f"Game over — {'you win' if self.state.turn != self.human_color else 'AI wins'}"
                    self.draw()
                    pygame.time.wait(1500)
                    self.game_result = self.compute_result_from_state()
                    self.running = False
                    continue
                self.animate_move(mv)
                self.apply_and_refresh(mv)
                self.message = f"AI played: {mv}"

            self.draw()

        self.save_pdn()
        pygame.quit()

    # -------------------- Side selection screen --------------------

    def show_side_selection(self):
        """
        Show a simple side selection screen where the user can choose to play as White or Black.
        1. Display two buttons: "Play White" and "Play Black"
        2. Wait for the user to click one of the buttons
        3. Set self.human_color based on the selection
        :return:
        """
        import sys as _sys
        clock = pygame.time.Clock()
        font_title = pygame.font.SysFont(None, 48)
        font_btn = pygame.font.SysFont(None, 32)

        selected = None
        while selected is None:
            w, h = self.screen.get_size()
            btn_w, btn_h = 260, 64
            gap = 28
            white_rect = pygame.Rect(w // 2 - btn_w - gap // 2, h // 2, btn_w, btn_h)
            black_rect = pygame.Rect(w // 2 + gap // 2, h // 2, btn_w, btn_h)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    _sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    _sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if white_rect.collidepoint(mx, my):
                        selected = 'w'
                    elif black_rect.collidepoint(mx, my):
                        selected = 'b'

            self.screen.fill((20, 22, 30))

            title_surf = font_title.render("Choose your side", True, (235, 235, 235))
            title_rect = title_surf.get_rect(center=(w // 2, h // 2 - 100))

            pygame.draw.rect(self.screen, (240, 240, 240), white_rect, border_radius=12)
            pygame.draw.rect(self.screen, (35, 35, 35), black_rect, border_radius=12)
            pygame.draw.rect(self.screen, (40, 40, 40), white_rect, width=2, border_radius=12)
            pygame.draw.rect(self.screen, (220, 220, 220), black_rect, width=2, border_radius=12)

            white_label = font_btn.render("Play White", True, (10, 10, 10))
            black_label = font_btn.render("Play Black", True, (235, 235, 235))

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(white_label, white_label.get_rect(center=white_rect.center))
            self.screen.blit(black_label, black_label.get_rect(center=black_rect.center))

            pygame.display.flip()
            clock.tick(60)

        self.human_color = selected
        self._show_loading_overlay("Preparing board...", duration_ms=700)

    # -------------------- Events --------------------

    def handle_events(self):
        """
        Handle pygame events: mouse clicks, key presses, window close.
        1. Process all pending pygame events
        2. Handle quitting the application
        3. Handle key presses for difficulty selection and undo/redo
        4. Handle mouse clicks for board interaction and sidebar buttons
        5. If it's not the human's turn, ignore board clicks
        6. Convert mouse position to board coordinates and handle piece selection/movement
        7. Update self.selected, self.message, and apply moves as needed
        8. No mouse wheel handling needed now that the Games panel is gone
        9. Return early if a new game is started to avoid processing further events
        :return:
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.set_difficulty("Beginner")
                elif event.key == pygame.K_2:
                    self.set_difficulty("Novice")
                elif event.key == pygame.K_3:
                    self.set_difficulty("Amateur")
                elif event.key == pygame.K_4:
                    self.set_difficulty("Pro")
                elif event.key == pygame.K_5:
                    self.set_difficulty("Hard")

                elif event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    plies = 2 if self.human_color and self.state.turn == self.human_color else 1
                    self.undo(plies)

                elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    plies = 2 if self.human_color and self.state.turn == self.human_color else 1
                    self.redo(plies)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                sidebar_x = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP

                # Sidebar always clickable
                if x >= sidebar_x:
                    # New Game button
                    if self.new_game_rect and self.new_game_rect.collidepoint(x, y):
                        self._show_loading_overlay("Starting new game...", duration_ms=900)
                        self._reset_to_new_game()
                        return
                    for rect, name in self.diff_rects:
                        if rect.collidepoint(x, y):
                            self.set_difficulty(name)
                            return
                    return

                # Bail quickly if it's not the human's turn
                if not (self.human_color and self.state.turn == self.human_color):
                    return

                # Board click
                r, c = pos_to_rc(x, y)
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    self.on_click(r, c)
        # No mouse wheel handling needed now that the Games panel is gone

    # -------------------- Game ops --------------------

    def set_difficulty(self, name: str):
        """
        Set the AI difficulty level.
        1. Check if the provided name is a valid difficulty level
        2. If valid, update self.diff_name and self.config
        :param name: Difficulty level name
        :return:
        """
        if name in DIFFICULTIES:
            self.diff_name = name
            self.config = DIFFICULTIES[name]
            self.message = f"Difficulty set to {name}."

    def on_click(self, r: int, c: int):
        """
        Handle a click on the board at row r, column c.
        1. If no piece is selected, select the clicked piece if it's the human's color
        2. If a piece is already selected, check if the clicked square is a valid move target
        3. If valid, apply the move with animation and update the game state
        4. If invalid, show an error message
        5. If clicking on another piece of the human's color, change the selection
        6. Clear selection after a move is made
        :param r:
        :param c:
        :return:
        """
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return
        p = self.state.board[r][c]

        if self.selected is None:
            if p != EMPTY and self.human_color and p.lower() == self.human_color:
                self.selected = (r, c)
                self.click_anim = (r, c, pygame.time.get_ticks())
            return

        if p != EMPTY and self.human_color and p.lower() == self.human_color:
            self.selected = (r, c)
            self.click_anim = (r, c, pygame.time.get_ticks())
            return

        path = [self.selected, (r, c)]
        mv = self.find_matching_move(path)
        if mv is None:
            self.message = "Invalid move. Try again."
            return

        self.selected = None
        self.animating = True
        self.animate_move(mv)
        self.animating = False

        self.apply_and_refresh(mv)

    def find_matching_move(self, path: list[tuple[int, int]]) -> Optional[Move]:
        """
        Find a legal move that matches the given path.
        1. Check if any legal move exactly matches the provided path
        2. If no exact match, check for moves that start with the provided path
        3. If exactly one move starts with the path, return that move
        :param path:
        :return:
        """
        for m in self.legal:
            if m.path == path:
                return m
        cand = [m for m in self.legal if m.path[: len(path)] == path]
        if len(cand) == 1:
            return cand[0]
        return None

    def apply_and_refresh(self, mv: Move):
        """
        Apply a move to the game state and refresh legal moves and history.
        1. Save the current state to the undo stack and clear the redo stack
        2. Apply the move to the current state
        3. Update the list of legal moves
        4. Append the move to the history
        5. Clear the selected piece
        6. Update draw rules and check for draw conditions
        7. If a draw condition is met, set the game result and stop the game
        :param mv:
        :return:
        """
        self.undo_stack.append(_clone_state(self.state))
        self.redo_stack.clear()

        self.state = self.state.apply_move(mv)
        self.legal = self.state.legal_moves()
        self.history.append(mv)
        self.selected = None

        dr = self.rules.on_move(self.state, mv)
        if dr == 'draw':
            self.game_result = "1/2-1/2"
            self.message = "Draw by 40-move rule / threefold repetition"
            self.running = False
            return

    def _show_loading_overlay(self, text: str = "Loading...", duration_ms: int = 800):
        """
        Non-flashy loading overlay: dimmed backdrop + gentle dot spinner at 30 FPS.
        1. Draw the current game state and save it as a background
        2. For the specified duration, display a semi-transparent overlay with a card
        3. On the card, show the provided text and a dot spinner animation
        4. Allow quitting during the overlay
        """
        # Draw once and reuse as background to avoid flicker
        self.draw()
        bg = self.screen.copy()

        duration_ms = max(300, int(duration_ms))  # clamp to reasonable range
        start = pygame.time.get_ticks()
        n = 12  # number of dots
        speed_hz = 5  # steps per second (gentle)
        base_alpha = 110  # dim dots
        active_alpha = 220  # active dot

        while self.running and (pygame.time.get_ticks() - start) < duration_ms:
            # allow quitting during overlay
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

            # restore background
            self.screen.blit(bg, (0, 0))

            # semi-transparent page dimmer
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))

            # card
            box_w, box_h = 360, 180
            box = pygame.Rect((W - box_w) // 2, (H - box_h) // 2, box_w, box_h)
            pygame.draw.rect(overlay, (248, 248, 248, 255), box, border_radius=12)
            pygame.draw.rect(overlay, (220, 220, 220, 255), box, 1, border_radius=12)

            # title
            label = self.big.render(text, True, (30, 30, 30))
            overlay.blit(label, label.get_rect(center=(box.centerx, box.top + 52)))

            # dot spinner (constant luminance; only one dot brighter)
            cx, cy = box.centerx, box.centery + 18
            r_track = 24
            base_color = (80, 80, 80)
            active_color = (60, 60, 60)

            elapsed = (pygame.time.get_ticks() - start) / 1000.0
            active_idx = int(elapsed * speed_hz) % n
            for i in range(n):
                a = (2 * 3.14159265 * i) / n
                x = int(cx + r_track * math.cos(a))
                y = int(cy + r_track * math.sin(a))
                if i == active_idx:
                    pygame.draw.circle(overlay, (*active_color, active_alpha), (x, y), 5)
                else:
                    pygame.draw.circle(overlay, (*base_color, base_alpha), (x, y), 4)

            self.screen.blit(overlay, (0, 0))
            pygame.display.flip()
            self.clock.tick(30)  # gentle frame rate

    def _reset_to_new_game(self):
        """
        Reset the game state to start a new game.
        1. Reset the game state, legal moves, and draw rules
        2. Clear the UI state: history, undo/redo stacks, selection, and message
        3. If the human plays Black, let the AI make the first move automatically
        :return:
        """
        # reset board
        self.state = CheckersState()
        self.legal = self.state.legal_moves()

        self.rules = DrawRulesTracker(DrawRulesConfig(80, 3))
        self.rules.start(self.state)

        # clear UI/game history
        self.history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected = None
        self.message = "New game started."

        # If human plays Black, let AI make the first move automatically
        if self.human_color == 'b' and self.state.turn == 'w':
            mv = choose_ai_move(self.state, self.config)
            self.apply_and_refresh(mv)

    def _rebuild_rules_from_history(self):
        """
        Rebuild the draw rules tracker from the current move history.
        This is needed after undo/redo operations to keep the draw rule counters and repetition map consistent.
        :return:
        """
        # Recreate the draw-rule tracker from scratch and replay the move history
        self.rules = DrawRulesTracker(DrawRulesConfig(
            no_capture_plies_threshold=80,  # 40 full moves = 80 plies
            repetition_threshold=3
        ))
        s = CheckersState()
        self.rules.start(s)
        for m in self.history:
            s = s.apply_move(m)
            self.rules.on_move(s, m)

    def undo(self, plies: int = 1):
        """
        Undo the last 'plies' moves.
        1. For the specified number of plies, pop states from the undo stack and push them onto the redo stack
        2. Update the current state, legal moves, and history accordingly
        3. Rebuild the draw rules tracker from the updated history
        4. Update the message to indicate how many plies were undone
        :param plies:
        :return:
        """
        for _ in range(plies):
            if not self.undo_stack:
                break
            self.redo_stack.append(_clone_state(self.state))
            prev_state = self.undo_stack.pop()
            if self.history:
                self.history.pop()
            self.state = prev_state
            self.legal = self.state.legal_moves()
            self.selected = None

        # keep draw rule counters and repetition map consistent
        self._rebuild_rules_from_history()

        self.message = f"Undid {plies} ply." if plies == 1 else f"Undid {plies} plies."

    def redo(self, plies: int = 1):
        """
        Redo the last 'plies' undone moves.
        1. For the specified number of plies, pop states from the redo stack and push them onto the undo stack
        2. Update the current state, legal moves, and history accordingly
        3. Rebuild the draw rules tracker from the updated history
        4. Update the message to indicate how many plies were redone
        :param plies:
        :return:
        """
        for _ in range(plies):
            if not self.redo_stack:
                break
            self.undo_stack.append(_clone_state(self.state))
            next_state = self.redo_stack.pop()
            self.state = next_state
            self.legal = self.state.legal_moves()
            self.selected = None

        # keep draw rule counters and repetition map consistent
        self._rebuild_rules_from_history()

        self.message = f"Redid {plies} ply." if plies == 1 else f"Redid {plies} plies."

    # -------------------- Animation --------------------

    def animate_move(self, mv: Move, duration: float = 0.25):
        """
        Animate a move by smoothly moving the piece along its path.
        1. Check if the move has a valid path with at least two squares
        2. For each segment of the path, interpolate the piece's position over the specified duration
        3. During the animation, redraw the board and pieces without the moving piece
        4. Draw the moving piece on top at its interpolated position
        5. Use a high frame rate for smooth animation
        6. The moving piece is drawn as a filled circle with an outline; kings have an additional inner circle
        7. The board is redrawn each frame to reflect the current state
        ...
        :param mv: Move to animate
        :param duration: Total duration of the animation in seconds
        :return:
        """
        if not mv.path or len(mv.path) < 2:
            return

        piece = self.state.board[mv.path[0][0]][mv.path[0][1]]

        for i in range(1, len(mv.path)):
            r0, c0 = mv.path[i - 1]
            r1, c1 = mv.path[i]
            x0, y0 = rc_to_xy(r0, c0)
            x1, y1 = rc_to_xy(r1, c1)

            start_ms = pygame.time.get_ticks()
            end_ms = start_ms + int(duration * 1000)

            while True:
                now = pygame.time.get_ticks()
                if now >= end_ms:
                    break
                t = (now - start_ms) / (duration * 1000.0)
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))

                # draw board & pieces without the moving piece
                saved_piece = self.state.board[r0][c0]
                self.state.board[r0][c0] = EMPTY
                self.draw(base_only=False)
                self.state.board[r0][c0] = saved_piece

                # draw moving piece on top
                center = (x + TILE // 2, y + TILE // 2)
                radius = TILE // 2 - 8
                fill = (30, 30, 30) if piece.lower() == 'b' else (240, 240, 240)
                outline = (230, 230, 230) if piece.lower() == 'b' else (40, 40, 40)
                pygame.draw.circle(self.screen, fill, center, radius)
                pygame.draw.circle(self.screen, outline, center, radius, 8)
                if piece in ('B', 'W'):
                    pygame.draw.circle(self.screen, outline, center, radius // 2, 6)

                pygame.display.flip()
                self.clock.tick(FPS)

    # -------------------- Rendering --------------------

    def draw(self, base_only: bool = False):
        """
        Draw the entire UI: board, pieces, sidebar, highlights, animations.
        :param base_only: If True, only draw the board and pieces (no highlights or animations)
        :return:
        """
        self.screen.fill((30, 30, 30))

        # Board
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
                    pygame.draw.circle(self.screen, outline, (x, y), radius, 8)
                    if p in ('B', 'W'):
                        pygame.draw.circle(self.screen, outline, (x, y), radius // 2, 6)

        # Sidebar
        sidebar_x = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP
        pygame.draw.rect(self.screen, (245, 245, 245), (sidebar_x, 0, SIDEBAR_W, H))

        y = SIDEBAR_PAD
        turn_is_human = (self.state.turn == self.human_color)
        title = self.big.render("Turn: You" if turn_is_human else "Turn: AI", True, TEXT)
        self.screen.blit(title, (sidebar_x + SIDEBAR_PAD, y))

        # subtle animated dots when AI is thinking
        if not turn_is_human:
            dots = (pygame.time.get_ticks() // 400) % 4  # 0..3
            thinking = self.small.render("thinking" + "." * dots, True, TEXT_MUTED)
            self.screen.blit(thinking, (sidebar_x + SIDEBAR_PAD + title.get_width() + 12, y + 6))

        y += title.get_height() + SIDEBAR_PAD

        # difficulty buttons
        self.diff_rects.clear()
        levels = ["Beginner", "Novice", "Amateur", "Pro", "Hard"]
        btn_h = 34
        mx, my = pygame.mouse.get_pos()
        for name in levels:
            rect = pygame.Rect(
                sidebar_x + SIDEBAR_PAD,
                y,
                SIDEBAR_W - 2 * SIDEBAR_PAD,
                btn_h
            )
            self.diff_rects.append((rect, name))
            hovered = rect.collidepoint(mx, my)
            base = ACCENT if name == self.diff_name else BTN
            if hovered and name != self.diff_name:
                base = (max(0, base[0] - 10), max(0, base[1] - 10), max(0, base[2] - 10))
            pygame.draw.rect(self.screen, base, rect, border_radius=8)
            label = self.font.render(name, True, (20, 20, 20))
            self.screen.blit(label, (rect.x + 10, rect.y + 7))
            y += btn_h + 8

        y += 6
        msg = self.font.render(self.message or "", True, TEXT)
        self.screen.blit(msg, (sidebar_x + SIDEBAR_PAD, y))
        y += msg.get_height() + 10

        hist = " ".join(str(m) for m in self.history[-30:])  # show more but wrapped
        maxw = SIDEBAR_W - 2 * SIDEBAR_PAD
        lines = wrap_text(f"Moves: {hist}", self.font, maxw)
        for ln in lines[:4]:  # cap lines so panel stays tidy
            move_history_surface = self.font.render(ln, True, TEXT)
            self.screen.blit(move_history_surface, (sidebar_x + SIDEBAR_PAD, y))
            y += move_history_surface.get_height() + 2

        # --- Controls box + New Game button ---
        controls_h = 180
        box = pygame.Rect(
            sidebar_x + SIDEBAR_PAD,
            H - SIDEBAR_PAD - controls_h,  # ⬅️ anchor to bottom
            SIDEBAR_W - 2 * SIDEBAR_PAD,
            controls_h
        )
        pygame.draw.rect(self.screen, (255, 255, 255), box, border_radius=10)
        pygame.draw.rect(self.screen, BORDER, box, 1, border_radius=10)

        label_controls = self.font.render("Controls", True, TEXT)
        self.screen.blit(label_controls, (box.x + 12, box.y + 10))

        lines = [
            "1–5: change difficulty",
            "Ctrl+Z / Ctrl+Y: undo / redo",
            "Click piece and move to the ",
            "destination square",
        ]
        ly = box.y + 40
        for ln in lines:
            t = self.small.render(ln, True, TEXT_MUTED)
            self.screen.blit(t, (box.x + 12, ly))
            ly += t.get_height() + 4

        # New Game button (stores rect for click handling)
        btn = pygame.Rect(box.x + 12, box.bottom - 40, box.width - 24, 34)
        self.new_game_rect = btn
        pygame.draw.rect(self.screen, ACCENT, btn, border_radius=8)
        btn_label = self.font.render("New Game", True, (20, 20, 20))
        self.screen.blit(btn_label, btn_label.get_rect(center=btn.center))
        # --------------------------------------

        pygame.display.flip()

    def save_pdn(self, filename=None):
        """
        Save the current game to a PDN file.
        1. If there is no move history, do nothing
        2. Determine the game result, using self.game_result if valid, otherwise compute from state
        3. Set player names based on human color
        4. Call save_game_pdn to write the PDN file in the data/games directory
        5. Update the message to indicate the game was saved
        6. If filename is None, a timestamped filename will be generated
        :param filename: Optional filename for the PDN file
        :return:
        """
        if not self.history:
            return
        result = self.game_result if self.game_result in {"1-0", "0-1", "1/2-1/2"} else self.compute_result_from_state()
        white = "Human" if self.human_color == "w" else "AI"
        black = "Human" if self.human_color == "b" else "AI"
        path = save_game_pdn(
            moves=self.history,
            result=result,
            white=white,
            black=black,
            out_dir="data/games",
            filename=filename,
        )
        self.message = f"Game saved."

    def compute_result_from_state(self) -> str:
        """
        Compute the game result from the current state.
        1. Check if the game is over using self.state.terminal()
        2. Return "1-0" if White wins, "0-1"
        if Black wins, "1/2-1/2" if it is a draw, or "*" if the game is ongoing
        3. This is a fallback if self.game_result is not already set
        4. Note: This does not handle all draw conditions (e.g., 40-move rule)
        :return:
        """
        winner = self.state.terminal()  # 'w', 'b', 'draw' or None
        if winner == 'w':
            return "1-0"
        if winner == 'b':
            return "0-1"
        if winner == 'draw':
            return "1/2-1/2"
        return "*"
