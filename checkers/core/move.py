from dataclasses import dataclass

Coord = tuple[int, int]
MoveSeq = list[Coord]

@dataclass(frozen=True)
class Move:
    """
    Represents a move in a checkers game.
    Attributes:
        path (MoveSeq): A sequence of coordinates representing the move path.
        is_capture (bool): Indicates if the move is a capturing move.
    """
    path: MoveSeq
    is_capture: bool

    def __str__(self) -> str:
        from checkers.util.util import idx_to_coord
        sep = ':' if self.is_capture else ' '
        return sep.join(idx_to_coord(r, c) for r, c in self.path)
