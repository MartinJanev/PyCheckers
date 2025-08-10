from .constants import BOARD_SIZE

def idx_to_coord(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{BOARD_SIZE - r}"

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
