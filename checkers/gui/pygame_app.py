import sys
import copy
import os
from typing import Optional, Tuple

from ..core.heuristics import HEURISTICS, combine, Heuristic  # (kept; may be used elsewhere)
from ..core.state import CheckersState
from ..core.constants import BOARD_SIZE, EMPTY
from ..core.move import Move
from ..core.pdn import save_game_pdn
from ..core.engine import choose_ai_move, EngineConfig
from ..core import pdn as pdn_mod  # for list/load/replay


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

BOARD_MARGIN_X = 20  # around the board (left/right)
BOARD_MARGIN_Y = 20  # around the board (top/bottom)
BOARD_SIDEBAR_GAP = 20  # gap between board and sidebar

SIDEBAR_W = 460  # widened right panel as requested
SIDEBAR_PAD = 14  # inner padding for sidebar

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
    Convert a board position (row, col) into pixel coordinates for the top-left corner of that tile.
    :param r: Board row index
    :param c: Board column index
    :return: (x, y) pixel coordinates on the screen
    """
    return BOARD_MARGIN_X + c * TILE, BOARD_MARGIN_Y + r * TILE


def pos_to_rc(x: int, y: int) -> Tuple[int, int]:
    """
    Convert pixel coordinates on the screen to board coordinates.
    :param x: X pixel coordinate
    :param y: Y pixel coordinate
    :return: (row, col) board coordinates, or (-1, -1) if outside board
    """
    x -= BOARD_MARGIN_X
    y -= BOARD_MARGIN_Y
    if x < 0 or y < 0:
        return -1, -1
    return y // TILE, x // TILE


DIFFICULTIES = {
    "Beginner": EngineConfig(algo="random", depth=1, heuristic="material", use_tt=False),
    "Novice": EngineConfig(algo="minimax", depth=3, heuristic="material", use_tt=False),
    "Amateur": EngineConfig(algo="expectimax", depth=3, heuristic="material_adv", use_tt=False),

    "Pro": EngineConfig(algo="minimax", depth=5, heuristic="pro_mix", use_tt=False),
    "Hard": EngineConfig(algo="minimax", depth=7, heuristic="hard_mix", use_tt=True),
}

DEFAULT_DIFF = "Amateur"
ALIASES = {"Easy": "Beginner", "Medium": "Pro", "Hard": "Hard"}


def _clone_state(s: CheckersState) -> CheckersState:
    """
    Clone a CheckersState for safe storage (undo/redo/replay).
    Tries copy(), then clone(), then falls back to deepcopy.
    :param s: The CheckersState to clone
    :return: A cloned CheckersState
    """
    if hasattr(s, "copy"):
        return s.copy()
    if hasattr(s, "clone"):
        return s.clone()
    return copy.deepcopy(s)


class PygameUI:
    def __init__(self, difficulty: str = "Amateur"):
        """
        Initialize the Pygame UI for PyCheckers.
        Sets up state, replay lists, sidebar data, and loads Pygame assets.
        :param difficulty: Starting AI difficulty
        """
        # Core state / flags
        self.message: str = ""
        self.animating: bool = False
        self.running: bool = True
        self.state: CheckersState = CheckersState()
        self.legal: list[Move] = self.state.legal_moves()

        # Replay management
        self.replay_states: list[CheckersState] = []
        self.replay_index: int = 0
        self.in_replay: bool = False
        self._pre_replay_state: Optional[CheckersState] = None  # live state to return to after replay

        # Sidebar game list
        self.game_files: list[str] = []
        self.game_rects: list[pygame.Rect] = []  # clickable rects aligned with game_files
        self.games_scroll: int = 0  # index offset for scroll
        self.max_game_rows: int = 10  # will be recomputed based on layout in draw

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
        self.undo_stack: list[CheckersState] = []  # snapshots before each move
        self.redo_stack: list[CheckersState] = []  # snapshots we can return to

        self.human_color: Optional[str] = None  # 'w' or 'b'; set after side selection

        # Initial game list
        self.refresh_game_list()

    # -------------------- Main loop --------------------

    def run(self):
        """
        Start the main game loop.
        Handles side selection, AI/human turns, rendering, and exit flow.
        :return: None

        Steps:
        1. Show side selection screen to choose 'w' (White) or 'b' (Black).
        2. Set starting message based on chosen side.
        3. If human is Black, let AI move first.
        4. Enter main loop:
            - Handle events (clicks, key presses).
            - If it's AI's turn, compute and animate AI move.
            - Draw the board and sidebar.
        5. On exit, save the game PDN if not in replay.
        6. Quit Pygame.
        7. Save PDN on exit if not in replay.
        8. Exit Pygame.
        """
        # Step 1: Choose side (fixes missing attribute error)
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

            # Do NOT let the AI think during replay
            if not self.in_replay and self.state.turn != self.human_color:
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

        # Save PDN on exit (skip if in replay)
        if not self.in_replay:
            self.save_pdn()
        pygame.quit()

    # -------------------- NEW: Side selection screen --------------------

    def show_side_selection(self):
        """
        Display a modal for selecting White or Black side.
        Updates self.human_color to 'w' or 'b'.
        Blocks until a side is selected or the window is closed.
        """
        import sys as _sys

        clock = pygame.time.Clock()
        font_title = pygame.font.SysFont(None, 48)
        font_btn = pygame.font.SysFont(None, 32)

        selected = None
        while selected is None:
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

            w, h = self.screen.get_size()
            self.screen.fill((20, 22, 30))

            title_surf = font_title.render("Choose your side", True, (235, 235, 235))
            title_rect = title_surf.get_rect(center=(w // 2, h // 2 - 100))

            btn_w, btn_h = 260, 64
            gap = 28
            white_rect = pygame.Rect(w // 2 - btn_w - gap // 2, h // 2, btn_w, btn_h)
            black_rect = pygame.Rect(w // 2 + gap // 2, h // 2, btn_w, btn_h)

            # Buttons
            pygame.draw.rect(self.screen, (240, 240, 240), white_rect, border_radius=12)
            pygame.draw.rect(self.screen, (35, 35, 35), black_rect, border_radius=12)
            pygame.draw.rect(self.screen, (40, 40, 40), white_rect, width=2, border_radius=12)
            pygame.draw.rect(self.screen, (220, 220, 220), black_rect, width=2, border_radius=12)

            white_label = font_btn.render("Play White (moves first)", True, (10, 10, 10))
            black_label = font_btn.render("Play Black", True, (235, 235, 235))

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(white_label, white_label.get_rect(center=white_rect.center))
            self.screen.blit(black_label, black_label.get_rect(center=black_rect.center))

            pygame.display.flip()
            clock.tick(60)

        self.human_color = selected  # 'w' or 'b'

    # -------------------- Events --------------------

    def handle_events(self):

        """
        Handle Pygame events: key presses, mouse clicks, and mouse wheel.
        Updates game state, handles user input, and manages replay navigation.
        Handles:
        - Key presses for difficulty selection, undo/redo, loading games, and replay navigation.
        - Mouse clicks for selecting pieces, moving them, and interacting with the sidebar.
        - Mouse wheel for scrolling through the game list in the sidebar.
        - Exiting the game on quit event.
        - Entering and exiting replay mode.
        - Saving/loading games from the sidebar.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # difficulty keys
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

                # Undo (Ctrl+Z): usually undo two plies (AI + Human) so you get back your move
                elif event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    plies = 2 if self.human_color and self.state.turn == self.human_color else 1
                    self.undo(plies)

                # Redo (Ctrl+Y)
                elif event.key == pygame.K_y and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    plies = 2 if self.human_color and self.state.turn == self.human_color else 1
                    self.redo(plies)

                # L: Load latest in replay
                elif event.key == pygame.K_l:
                    files = pdn_mod.list_game_files("data/games")
                    if not files:
                        self.message = "No saved games."
                    else:
                        headers, tokens = pdn_mod.load_pdn_file(files[0])
                        states = pdn_mod.replay_pdn(tokens)
                        self.enter_replay(states)

                # O: Open latest and continue from last position
                elif event.key == pygame.K_o and not self.in_replay:
                    files = pdn_mod.list_game_files("data/games")
                    if not files:
                        self.message = "No saved games."
                    else:
                        headers, tokens = pdn_mod.load_pdn_file(files[0])
                        states = pdn_mod.replay_pdn(tokens)
                        self.state = states[-1]
                        self.legal = self.state.legal_moves()
                        self.selected = None
                        self.message = "Game loaded. Continue playing."

                # R: Refresh game list
                elif event.key == pygame.K_r:
                    self.refresh_game_list()
                    self.message = "Game list refreshed."

                # Step forward/backward through replay
                elif event.key == pygame.K_RIGHT and self.in_replay:
                    if self.replay_states and self.replay_index < len(self.replay_states) - 1:
                        self.replay_index += 1
                        self.state = self.replay_states[self.replay_index]
                        self.legal = self.state.legal_moves()
                        self.selected = None
                        self.message = f"Step {self.replay_index}/{len(self.replay_states) - 1}"

                elif event.key == pygame.K_LEFT and self.in_replay:
                    if self.replay_states and self.replay_index > 0:
                        self.replay_index -= 1
                        self.state = self.replay_states[self.replay_index]
                        self.legal = self.state.legal_moves()
                        self.selected = None
                        self.message = f"Step {self.replay_index}/{len(self.replay_states) - 1}"

                # ESC: Exit replay
                elif event.key == pygame.K_ESCAPE and self.in_replay:
                    self.exit_replay()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                sidebar_x = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP
                if x >= sidebar_x:
                    # Sidebar interactions
                    # Difficulty buttons
                    for rect, name in self.diff_rects:
                        if rect.collidepoint(x, y):
                            self.set_difficulty(name)
                            return

                    # Game list clicks
                    for idx, rect in enumerate(self.game_rects):
                        if rect.collidepoint(x, y):
                            real_index = self.games_scroll + idx
                            if 0 <= real_index < len(self.game_files):
                                path = self.game_files[real_index]
                                headers, tokens = pdn_mod.load_pdn_file(path)
                                states = pdn_mod.replay_pdn(tokens)
                                if pygame.mouse.get_pressed()[2] or (
                                        pygame.key.get_mods() & pygame.KMOD_SHIFT
                                ):
                                    # Right-click or Shift+click: open and continue from last position
                                    if states:
                                        self.state = states[-1]
                                        self.legal = self.state.legal_moves()
                                        self.selected = None
                                        self.in_replay = False
                                        self.message = f"Loaded {os.path.basename(path)}. Continue playing."
                                else:
                                    # Left-click: enter replay
                                    self.enter_replay(states)
                            return

                    # Scroll wheel handled in MOUSEWHEEL
                    return

                # Board clicks
                if self.human_color and self.state.turn == self.human_color and not self.in_replay:
                    r, c = pos_to_rc(x, y)
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                        self.on_click(r, c)

            elif event.type == pygame.MOUSEWHEEL:
                # Scroll the games list (if mouse over sidebar area)
                mx, my = pygame.mouse.get_pos()
                sidebar_x = BOARD_MARGIN_X + BOARD_W + BOARD_SIDEBAR_GAP
                if mx >= sidebar_x:
                    if event.y > 0:  # scroll up
                        self.games_scroll = max(0, self.games_scroll - 1)
                    elif event.y < 0:  # scroll down
                        max_start = max(0, len(self.game_files) - self.max_game_rows)
                        self.games_scroll = min(max_start, self.games_scroll + 1)

    # -------------------- Replay helpers --------------------

    def enter_replay(self, states: list[CheckersState]):
        """
        Enter replay mode with a list of states.
        This will freeze the current game state and allow stepping through the provided states.
        If no states are provided, sets an error message.
        If states are provided, it initializes the replay with the first state and clears any previous replay data.
        :param states: List of CheckersState objects representing the replay history.
        :return:
        """
        if not states:
            self.message = "Replay failed: no states."
            return
        # Remember live state and freeze the game
        self._pre_replay_state = _clone_state(self.state)
        self.in_replay = True
        self.replay_states = states
        self.replay_index = 0
        self.state = self.replay_states[0]
        self.legal = self.state.legal_moves()
        self.selected = None
        self.message = f"Replay loaded ({len(states) - 1} plies). Use ←/→, ESC to exit."

    def exit_replay(self):
        """
        Exit replay mode and restore the live game state.
        Clears the replay states and resets the replay index.
        :return:
        """
        if not self.in_replay:
            return
        # Restore the live position exactly as it was
        self.in_replay = False
        if self._pre_replay_state is not None:
            self.state = self._pre_replay_state
            self.legal = self.state.legal_moves()
        self.replay_states = []
        self.replay_index = 0
        self._pre_replay_state = None
        self.selected = None
        self.message = "Exited replay."

    # -------------------- Game ops --------------------

    def refresh_game_list(self):
        """
        Refresh the list of saved game files from the "data/games" directory.
        :return:
        """
        self.game_files = pdn_mod.list_game_files("data/games")
        self.games_scroll = 0

    def set_difficulty(self, name: str):
        """
        Set the AI difficulty based on the provided name.
        Updates the config and message accordingly.
        If the name is not recognized, it does nothing.
        :param name: Name of the difficulty level (e.g., "Beginner", "Novice", etc.)
        :return:
        """
        if name in DIFFICULTIES:
            self.diff_name = name
            self.config = DIFFICULTIES[name]
            self.message = f"Difficulty set to {name}."

    def on_click(self, r: int, c: int):
        """
        Handle a click on the board at position (r, c).
        If the click is on a piece, it selects it; if it's on an empty square, it tries to move the selected piece there.
        :param r: Row index of the clicked tile (0-7)
        :param c: Column index of the clicked tile (0-7)
        :return:
        """
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
        """
        Find a legal move that matches the given path.
        This checks if the path corresponds to any of the legal moves available in the current state.
        If exactly one match is found, it returns that move; otherwise, it returns None.
        :param path: List of (row, col) tuples representing the move path
        :return: A Move object if a matching move is found, otherwise None.
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
        Apply a move to the current game state and update the UI.
        This method:
        1. Saves the current state to the undo stack.
        2. Clears the redo stack.
        3. Applies the move to the state.
        4. Updates the legal moves.
        5. Appends the move to the history.
        6. Clears the selected piece.
        :param mv:
        :return:
        """
        # snapshot *current* position BEFORE applying the move
        self.undo_stack.append(_clone_state(self.state))
        self.redo_stack.clear()

        # apply move -> new state
        self.state = self.state.apply_move(mv)
        self.legal = self.state.legal_moves()
        self.history.append(mv)
        self.selected = None

    def undo(self, plies: int = 1):
        """
        Undo the last move(s) by restoring the previous state from the undo stack.
        This method:
        1. Checks if we are in replay mode; if so, shows a message and returns.
        2. Iterates through the specified number of plies to undo.
        3. For each ply, it pops the last state from the undo stack and pushes the current state to the redo stack.
        4. Updates the current state to the previous state.
        5. Clears the selected piece and updates the legal moves.
        6. Sets a message indicating how many plies were undone.
        :param plies: Number of plies to undo (default is 1).
        :return:
        """
        if self.in_replay:
            self.message = "Cannot undo during replay. Press ESC to exit replay."
            return
        for _ in range(plies):
            if not self.undo_stack:
                break
            # current state goes to redo, we revert to the snapshot
            self.redo_stack.append(_clone_state(self.state))
            prev_state = self.undo_stack.pop()
            if self.history:
                self.history.pop()
            self.state = prev_state
            self.legal = self.state.legal_moves()
            self.selected = None
        self.message = f"Undid {plies} ply." if plies == 1 else f"Undid {plies} plies."

    def redo(self, plies: int = 1):
        """
        Redo the last undone move(s) by restoring the next state from the redo stack.
        This method:
        1. Checks if we are in replay mode; if so, shows a message and returns.
        2. Iterates through the specified number of plies to redo.
        3. For each ply, it pops the last state from the redo stack and pushes the current state to the undo stack.
        4. Updates the current state to the next state.
        5. Clears the selected piece and updates the legal moves.
        6. Sets a message indicating how many plies were redone.
        :param plies: Number of plies to redo (default is 1).
        :return:
        """
        if self.in_replay:
            self.message = "Cannot redo during replay. Press ESC to exit replay."
            return
        for _ in range(plies):
            if not self.redo_stack:
                break
            # current state goes back to undo, jump to the redo snapshot
            self.undo_stack.append(_clone_state(self.state))
            next_state = self.redo_stack.pop()
            self.state = next_state
            self.legal = self.state.legal_moves()
            self.selected = None
        self.message = f"Redid {plies} ply." if plies == 1 else f"Redid {plies} plies."

    # -------------------- Animation --------------------

    def animate_move(self, mv: Move, duration: float = 0.25):
        """
        Animate a move by drawing the piece moving from its start position to its end position.
        This method:
        1. Checks if the move has a valid path.
        2. If the path is valid, it iterates through each segment of the path.
        3. For each segment, it calculates the pixel coordinates for the start and end positions.
        4. It then interpolates the position over the specified duration, drawing the piece at each frame.
        5. The piece is drawn as a circle with a fill and outline color based on its type (black or white).
        6. The screen is updated at a specified FPS to create a smooth animation.
        :param mv: The Move object containing the path to animate.
        :param duration: Duration of the animation in seconds (default is 0.25).
        :return:
        """
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
                pygame.draw.circle(self.screen, outline, center, radius, 6)  # thicker outline
                if piece in ('B', 'W'):
                    pygame.draw.circle(self.screen, outline, center, radius // 2, 5)

                pygame.display.flip()
                self.clock.tick(FPS)

    # -------------------- Rendering --------------------

    def _truncate(self, text: str, max_width: int, font: pygame.font.Font) -> str:
        """
        Truncate a string to fit within a specified width, adding an ellipsis if necessary.
        This method checks if the text fits within the max width using the provided font.
        If it does not fit, it iteratively removes characters from the end until it fits,
        appending an ellipsis ("…") at the end.
        :param text: Text to truncate
        :param max_width: Maximum width in pixels that the text should fit within
        :param font: pygame.font.Font object used to measure text size
        :return: Truncated text with ellipsis if necessary
        """
        if font.size(text)[0] <= max_width:
            return text
        ell = "…"
        while text and font.size(text + ell)[0] > max_width:
            text = text[:-1]
        return text + ell

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
                    pygame.draw.circle(self.screen, outline, (x, y), radius, 6)  # thicker outline
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

        # difficulty buttons
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
            color = BTN if name != self.diff_name else ACCENT
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            label = self.font.render(name, True, (20, 20, 20))
            self.screen.blit(label, (rect.x + 10, rect.y + 7))
            y += btn_h + 8

        y += 6
        # message
        msg = self.font.render(self.message or "", True, TEXT)
        self.screen.blit(msg, (sidebar_x + SIDEBAR_PAD, y))
        y += msg.get_height() + 10

        # history (last line)
        hist = "  ".join(str(m) for m in self.history[-8:])
        move_history_surface = self.font.render(f"Moves: {hist}", True, TEXT)
        self.screen.blit(move_history_surface, (sidebar_x + SIDEBAR_PAD, y))
        y += move_history_surface.get_height() + SIDEBAR_PAD

        # --- Games list header ---
        header = self.big.render("Games", True, TEXT)
        self.screen.blit(header, (sidebar_x + SIDEBAR_PAD, y))
        # refresh hint
        hint = self.small.render("(R to refresh • L=latest replay • O=latest continue)", True, TEXT_MUTED)
        self.screen.blit(hint, (sidebar_x + SIDEBAR_PAD, y + header.get_height()))
        y += header.get_height() + hint.get_height() + 8

        # List area frame
        list_x = sidebar_x + SIDEBAR_PAD
        list_w = SIDEBAR_W - 2 * SIDEBAR_PAD
        list_h = H - y - SIDEBAR_PAD
        pygame.draw.rect(self.screen, (252, 252, 252), (list_x, y, list_w, list_h), border_radius=8)
        pygame.draw.rect(self.screen, BORDER, (list_x, y, list_w, list_h), 1, border_radius=8)

        # Compute rows and draw entries
        row_h = 30
        self.max_game_rows = max(1, (list_h - 8) // row_h)
        start = self.games_scroll
        end = min(len(self.game_files), start + self.max_game_rows)
        self.game_rects = []

        if not self.game_files:
            empty = self.small.render("No games yet.", True, TEXT_MUTED)
            self.screen.blit(empty, (list_x + 10, y + 8))
        else:
            for i, path in enumerate(self.game_files[start:end]):
                ry = y + 4 + i * row_h
                rect = pygame.Rect(list_x + 4, ry, list_w - 8, row_h - 4)
                self.game_rects.append(rect)
                base = os.path.basename(path)
                text = self._truncate(base, rect.width - 10, self.small)

                # highlight if this file is the one in replay
                is_active = False
                if self.in_replay and self.replay_states:
                    # best-effort: compare to currently loaded file name in message
                    is_active = base in self.message

                pygame.draw.rect(self.screen, (238, 241, 245) if not is_active else (220, 235, 220), rect,
                                 border_radius=6)
                pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)

                label = self.small.render(text, True, (30, 30, 30))
                self.screen.blit(label, (rect.x + 6, rect.y + 5))

            # Scroll indicators
            if start > 0:
                up = self.small.render("▲", True, TEXT_MUTED)
                self.screen.blit(up, (list_x + list_w - 20, y + 4))
            if end < len(self.game_files):
                down = self.small.render("▼", True, TEXT_MUTED)
                self.screen.blit(down, (list_x + list_w - 20, y + list_h - 20))

        pygame.display.flip()

    def save_pdn(self, filename=None):
        """
        Save the current game state to a PDN file.
        This method:
        1. Checks if we are in replay mode or if there are no moves in history
        2. If not in replay and there are moves, it computes the game result.
        3. Determines the player names based on the human color.
        4. Calls the `save_game_pdn` function to save the game to a file.
        5. Refreshes the game list to include the newly saved game.
        6. Sets a message indicating the save location.
        :param filename: Optional filename to save the game as; if None, a timestamped name is used.
        :return: None
        """
        # Skip saving while in replay or if there are no moves
        if self.in_replay or not self.history:
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
            filename=filename,  # None => auto timestamped name
        )
        # Refresh list so the newly saved game appears immediately
        self.refresh_game_list()
        self.message = f"Game saved to {os.path.basename(path)}"

    def compute_result_from_state(self) -> str:
        """
        Return PDN result tag based on terminal state.
        This method checks the current game state to determine the result of the game.
        It returns:
        - "1-0" if White wins,
        - "0-1" if Black wins,
        - "1/2-1/2" if the game is a draw,
        - "*" if the game is still ongoing (no winner yet).
        :return: A string representing the game result in PDN format.
        """
        winner = self.state.terminal()  # returns 'w', 'b', 'draw' or None
        if winner == 'w':
            return "1-0"
        if winner == 'b':
            return "0-1"
        if winner == 'draw':
            return "1/2-1/2"
        return "*"
