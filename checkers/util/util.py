from checkers.util.constants import BOARD_SIZE

def idx_to_coord(r: int, c: int) -> str:
    """
    Convert (row, col) indices to standard board coordinates (e.g. (0,0) -> 'a8').
    0,0 is top-left corner (a8), 7,7 is bottom-right corner (h1).
    0th row is 8th rank, 0th col is 'a' file.
    0 <= r,c < BOARD_SIZE
    :param r: row index
    :param c: column index
    :return: board coordinate as string
    """
    return f"{chr(ord('a') + c)}{BOARD_SIZE - r}"

def in_bounds(r: int, c: int) -> bool:
    """
    Check if (r,c) indices are within board bounds.
    0 <= r,c < BOARD_SIZE
    0,0 is top-left corner (a8), 7,7 is bottom-right corner (h1).
    0th row is 8th rank, 0th col is 'a' file.
    0 <= r,c < BOARD_SIZE
    :param r: row index
    :param c: column index
    :return: True if in bounds, else False
    """
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
