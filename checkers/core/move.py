from dataclasses import dataclass
from typing import List, Tuple

Coord = tuple[int, int]
MoveSeq = list[Coord]

@dataclass(frozen=True)
class Move:
    path: MoveSeq
    is_capture: bool

    def __str__(self) -> str:
        from .util import idx_to_coord
        sep = ':' if self.is_capture else ' '
        return sep.join(idx_to_coord(r, c) for r, c in self.path)
