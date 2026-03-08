"""Hex game engine. 11x11 board, connection game with swap rule."""

from collections import deque
from typing import Any, Optional

from games.base import BaseGame

BOARD_SIZE = 11
EMPTY = 0
RED = 1    # connects top to bottom
BLUE = 2   # connects left to right

# Six hex neighbors using offset coordinates (each row shifted by half a cell).
# In offset coords, neighbors depend on whether the row is even or odd,
# but for a flat hex grid stored as a simple 2D array the six neighbors are:
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


def _neighbors(r: int, c: int) -> list[tuple[int, int]]:
    """Return valid neighbor cells for hex position (r, c)."""
    result = []
    for dr, dc in HEX_NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            result.append((nr, nc))
    return result


def _check_win(board: list[list[int]], color: int) -> bool:
    """BFS to check if color connects its two target edges.

    Red (1): connects top edge (row 0) to bottom edge (row 10).
    Blue (2): connects left edge (col 0) to right edge (col 10).
    """
    if color == RED:
        # Start from all red stones on row 0
        starts = [(0, c) for c in range(BOARD_SIZE) if board[0][c] == RED]
        target_check = lambda r, c: r == BOARD_SIZE - 1
    else:
        # Start from all blue stones on col 0
        starts = [(r, 0) for r in range(BOARD_SIZE) if board[r][0] == BLUE]
        target_check = lambda r, c: c == BOARD_SIZE - 1

    if not starts:
        return False

    visited = set(starts)
    queue = deque(starts)
    while queue:
        r, c = queue.popleft()
        if target_check(r, c):
            return True
        for nr, nc in _neighbors(r, c):
            if (nr, nc) not in visited and board[nr][nc] == color:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def _largest_connected_group(board: list[list[int]], color: int) -> int:
    """Return the size of the largest connected group of stones of the given color."""
    visited = set()
    largest = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == color and (r, c) not in visited:
                # BFS from this stone
                group_size = 0
                queue = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    group_size += 1
                    for nr, nc in _neighbors(cr, cc):
                        if (nr, nc) not in visited and board[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                largest = max(largest, group_size)
    return largest


def _edge_connections(board: list[list[int]], color: int) -> int:
    """Count how many of the player's 2 target edges they touch (0, 1, or 2)."""
    count = 0
    if color == RED:
        # Top edge (row 0) and bottom edge (row BOARD_SIZE-1)
        if any(board[0][c] == color for c in range(BOARD_SIZE)):
            count += 1
        if any(board[BOARD_SIZE - 1][c] == color for c in range(BOARD_SIZE)):
            count += 1
    else:
        # Left edge (col 0) and right edge (col BOARD_SIZE-1)
        if any(board[r][0] == color for r in range(BOARD_SIZE)):
            count += 1
        if any(board[r][BOARD_SIZE - 1] == color for r in range(BOARD_SIZE)):
            count += 1
    return count


def _center_control(board: list[list[int]], color: int) -> float:
    """Fraction of the center 5x5 region owned by color."""
    start = (BOARD_SIZE - 5) // 2  # row/col 3
    total = 25
    owned = 0
    for r in range(start, start + 5):
        for c in range(start, start + 5):
            if board[r][c] == color:
                owned += 1
    return owned / total


def _bridge_threats(board: list[list[int]], color: int) -> int:
    """Count bridge patterns: two stones of the same color separated by exactly
    two empty cells that are mutual neighbors of both stones. A bridge is a
    virtual connection -- if the opponent blocks one cell, the player can
    connect through the other.

    For hex neighbors, a bridge exists between (r1,c1) and (r2,c2) when they
    share exactly 2 common empty neighbors and are distance-2 apart along
    specific hex directions.
    """
    bridges = 0
    counted = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != color:
                continue
            nbrs = _neighbors(r, c)
            for nr, nc in nbrs:
                if board[nr][nc] != EMPTY:
                    continue
                # Look at neighbors of the empty cell that are our color
                for nr2, nc2 in _neighbors(nr, nc):
                    if (nr2, nc2) == (r, c):
                        continue
                    if board[nr2][nc2] != color:
                        continue
                    # Check this is a real bridge: (r,c) and (nr2,nc2) must
                    # share exactly 2 common neighbors, both empty.
                    pair = (min((r, c), (nr2, nc2)), max((r, c), (nr2, nc2)))
                    if pair in counted:
                        continue
                    common = set(_neighbors(r, c)) & set(_neighbors(nr2, nc2))
                    empty_common = [(er, ec) for er, ec in common
                                    if board[er][ec] == EMPTY]
                    if len(empty_common) == 2:
                        counted.add(pair)
                        bridges += 1
    return bridges


class HexGame(BaseGame):
    """11x11 Hex game with swap rule."""

    def __init__(self):
        self.board: list[list[int]] = [
            [EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        self._turn: str = "red"
        self._move_count: int = 0
        self._winner: Optional[str] = None
        self._swap_available: bool = False  # True only when Blue can swap

    # ---- BaseGame properties ------------------------------------------------

    @property
    def players(self) -> tuple[str, str]:
        return ("red", "blue")

    @property
    def current_player(self) -> str:
        return self._turn

    @property
    def move_count(self) -> int:
        return self._move_count

    # ---- Core methods -------------------------------------------------------

    def copy(self) -> "HexGame":
        g = HexGame.__new__(HexGame)
        g.board = [row[:] for row in self.board]
        g._turn = self._turn
        g._move_count = self._move_count
        g._winner = self._winner
        g._swap_available = self._swap_available
        return g

    def get_legal_moves(self, player: str | None = None) -> list:
        if player is None:
            player = self._turn
        if self._winner is not None:
            return []
        moves: list[Any] = []
        # Swap is available only for blue on move 2 (after red's first move)
        if player == "blue" and self._swap_available and player == self._turn:
            moves.append("swap")
        # All empty cells
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY:
                    moves.append((r, c))
        return moves

    def apply_move(self, move: Any) -> None:
        if self._winner is not None:
            raise ValueError("Game is already over.")

        if move == "swap":
            if not self._swap_available or self._turn != "blue":
                raise ValueError("Swap is not available.")
            # Find Red's stone and claim it as Blue's
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if self.board[r][c] == RED:
                        self.board[r][c] = BLUE
            self._swap_available = False
            self._move_count += 1
            self._turn = "red"
            # Check win (extremely unlikely after swap, but be correct)
            self._check_winner()
            return

        r, c = move
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            raise ValueError(f"Position ({r},{c}) is out of bounds.")
        if self.board[r][c] != EMPTY:
            raise ValueError(f"Position ({r},{c}) is already occupied.")

        color = RED if self._turn == "red" else BLUE
        self.board[r][c] = color
        self._move_count += 1

        # After Red's first move, Blue can swap on the next turn
        if self._move_count == 1 and self._turn == "red":
            self._swap_available = True
        else:
            self._swap_available = False

        # Switch turn
        self._turn = "blue" if self._turn == "red" else "red"

        # Check for winner
        self._check_winner()

    def _check_winner(self) -> None:
        """Update self._winner if someone has won or move limit reached."""
        if _check_win(self.board, RED):
            self._winner = "red"
        elif _check_win(self.board, BLUE):
            self._winner = "blue"
        elif self._move_count >= 200:
            self._winner = "draw"

    def winner(self) -> Optional[str]:
        return self._winner

    def render(self, perspective: str | None = None) -> str:
        symbols = {EMPTY: ".", RED: "R", BLUE: "B"}
        # Column header
        col_header = "  " + " ".join(f"{c:X}" if c >= 10 else str(c)
                                      for c in range(BOARD_SIZE))
        lines = [col_header]
        for r in range(BOARD_SIZE):
            indent = " " * r
            row_label = f"{r:>2}"
            row_str = " ".join(symbols[self.board[r][c]]
                               for c in range(BOARD_SIZE))
            lines.append(f"{indent}{row_label}  {row_str}")
        return "\n".join(lines)

    def get_state(self, player: str) -> dict:
        my_color_int = RED if player == "red" else BLUE
        opp_color_int = BLUE if player == "red" else RED

        my_stones = sum(
            1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if self.board[r][c] == my_color_int
        )
        opp_stones = sum(
            1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if self.board[r][c] == opp_color_int
        )

        return {
            "board": [row[:] for row in self.board],
            "my_color": player,
            "move_number": self._move_count,
            "board_size": BOARD_SIZE,
            "my_stones": my_stones,
            "opponent_stones": opp_stones,
            "my_longest_chain": _largest_connected_group(self.board, my_color_int),
            "opponent_longest_chain": _largest_connected_group(self.board, opp_color_int),
            "my_edge_connections": _edge_connections(self.board, my_color_int),
            "opponent_edge_connections": _edge_connections(self.board, opp_color_int),
            "center_control": _center_control(self.board, my_color_int),
            "my_bridge_threats": _bridge_threats(self.board, my_color_int),
        }

    def move_to_str(self, move: Any) -> str:
        if move == "swap":
            return "swap"
        r, c = move
        return f"({r},{c})"

    def parse_move(self, text: str, player: str) -> Any:
        """Parse a move from LLM output. Accepts '(row,col)', 'row,col', or 'swap'."""
        import re

        text = text.strip().lower()

        # Check for swap
        if "swap" in text:
            legal = self.get_legal_moves(player)
            if "swap" in legal:
                return "swap"
            return None

        # Try to extract coordinates
        # Match patterns like (5,6) or 5,6 or (5, 6) etc.
        match = re.search(r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?', text)
        if not match:
            return None

        r, c = int(match.group(1)), int(match.group(2))
        move = (r, c)
        legal = self.get_legal_moves(player)
        if move in legal:
            return move
        return None
