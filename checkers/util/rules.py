# checkers/core/rules.py
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

from checkers.core.move import Move
from checkers.core.state import CheckersState

@dataclass
class DrawRulesConfig:
    # Forty-move rule is traditionally 40 *full moves* without a capture.
    # We count in plies (half-moves), so default threshold is 80 plies.
    no_capture_plies_threshold: int = 80

    # Threefold repetition threshold (position incl. side-to-move)
    repetition_threshold: int = 3

class DrawRulesTracker:
    """
    Tracks (1) 40-move no-capture and (2) threefold repetition.
    Stateless w.r.t. search tree: use only in the actual game driver (root play).
    """
    def __init__(self, cfg: Optional[DrawRulesConfig] = None):
        self.cfg = cfg or DrawRulesConfig()
        self.no_capture_plies: int = 0
        self.pos_counts: Dict[str, int] = defaultdict(int)

    def start(self, state: CheckersState) -> None:
        """Initialize tracker with the *current* position."""
        fen = state.to_fen()               # includes side to move
        self.pos_counts[fen] += 1

    def on_move(self, state_after: CheckersState, mv: Move) -> Optional[str]:
        """
        Update counters after a move is applied.
        Return 'draw' if any draw rule is met, else None.
        """
        # 1) 40-move rule: reset on capture, else increment
        if mv.is_capture:
            self.no_capture_plies = 0
        else:
            self.no_capture_plies += 1
            if self.no_capture_plies >= self.cfg.no_capture_plies_threshold:
                return 'draw'

        # 2) Repetition rule: count positions including side-to-move
        fen = state_after.to_fen()
        self.pos_counts[fen] += 1
        if self.pos_counts[fen] >= self.cfg.repetition_threshold:
            return 'draw'

        return None
