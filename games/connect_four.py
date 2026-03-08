"""Connect Four game engine. 7 columns, 6 rows, 4-in-a-row to win."""

import re
from typing import Any, Optional

from games.base import BaseGame

ROWS = 6
COLS = 7
EMPTY = 0
RED = 1
YELLOW = 2

# All directions for checking lines: (dr, dc)
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class ConnectFourGame(BaseGame):
    """Standard Connect Four on a 7x6 board.

    Red goes first. Players drop discs into columns; discs fall to the
    lowest empty row. First to get 4 in a row wins.
    """

    def __init__(self):
        # board[r][c]: row 0 is the top, row 5 is the bottom
        self.board: list[list[int]] = [[EMPTY] * COLS for _ in range(ROWS)]
        self.turn: str = "red"
        self._move_count: int = 0

    # ── BaseGame properties ──────────────────────────────────────────

    @property
    def players(self) -> tuple[str, str]:
        return ("red", "yellow")

    @property
    def current_player(self) -> str:
        return self.turn

    @property
    def move_count(self) -> int:
        return self._move_count

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> "ConnectFourGame":
        g = ConnectFourGame.__new__(ConnectFourGame)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g._move_count = self._move_count
        return g

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _player_to_disc(player: str) -> int:
        return RED if player == "red" else YELLOW

    @staticmethod
    def _disc_to_player(disc: int) -> str:
        return "red" if disc == RED else "yellow"

    def _lowest_empty_row(self, col: int) -> Optional[int]:
        """Return the lowest empty row in `col`, or None if full."""
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                return r
        return None

    def _column_height(self, col: int) -> int:
        """Return how many discs are in `col`."""
        count = 0
        for r in range(ROWS):
            if self.board[r][col] != EMPTY:
                count += 1
        return count

    def _check_four(self) -> Optional[int]:
        """Return the disc value (RED/YELLOW) that has 4 in a row, or None."""
        for r in range(ROWS):
            for c in range(COLS):
                disc = self.board[r][c]
                if disc == EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    if self._line_of_four(r, c, dr, dc, disc):
                        return disc
        return None

    def _line_of_four(self, r: int, c: int, dr: int, dc: int, disc: int) -> bool:
        """Check whether there are 4 consecutive `disc` starting at (r,c)."""
        for i in range(4):
            nr, nc = r + dr * i, c + dc * i
            if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                return False
            if self.board[nr][nc] != disc:
                return False
        return True

    # ── Core game interface ──────────────────────────────────────────

    def get_legal_moves(self, player: str | None = None) -> list[int]:
        """Return list of column indices that are not full."""
        _ = player  # legal moves don't depend on which player
        return [c for c in range(COLS) if self.board[0][c] == EMPTY]

    def apply_move(self, move: Any) -> None:
        col = int(move)
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} out of range 0-{COLS - 1}")
        row = self._lowest_empty_row(col)
        if row is None:
            raise ValueError(f"Column {col} is full")
        self.board[row][col] = self._player_to_disc(self.turn)
        self._move_count += 1
        self.turn = "yellow" if self.turn == "red" else "red"

    def winner(self) -> Optional[str]:
        """Return 'red', 'yellow', 'draw', or None if game is ongoing."""
        four = self._check_four()
        if four is not None:
            return self._disc_to_player(four)
        # Draw if every column is full
        if all(self.board[0][c] != EMPTY for c in range(COLS)):
            return "draw"
        return None

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, perspective: str | None = None) -> str:
        symbol = {EMPTY: ".", RED: "R", YELLOW: "Y"}
        lines = []
        for r in range(ROWS):
            lines.append(" ".join(symbol[self.board[r][c]] for c in range(COLS)))
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)

    # ── Move parsing / formatting ────────────────────────────────────

    def move_to_str(self, move: Any) -> str:
        return str(move)

    def parse_move(self, text: str, player: str) -> Optional[int]:
        """Parse an LLM output into a column number.

        Accepts formats like "3", "column 3", "col 3", "drop in 3", etc.
        """
        text = text.strip()

        # Try to find a bare digit 0-6
        # First try: the entire text is just a digit
        if text.isdigit():
            col = int(text)
            if col in self.get_legal_moves(player):
                return col

        # Look for "column N" or "col N" patterns
        m = re.search(r'(?:column|col)\s*(\d+)', text, re.IGNORECASE)
        if m:
            col = int(m.group(1))
            if col in self.get_legal_moves(player):
                return col

        # Fall back: find any single digit 0-6 in the text
        digits = re.findall(r'\b([0-6])\b', text)
        if len(digits) == 1:
            col = int(digits[0])
            if col in self.get_legal_moves(player):
                return col

        # If multiple digits found, prefer one that is a legal move
        if digits:
            legal = set(self.get_legal_moves(player))
            for d in digits:
                col = int(d)
                if col in legal:
                    return col

        return None

    # ── State for strategies ─────────────────────────────────────────

    def get_state(self, player: str) -> dict:
        disc = self._player_to_disc(player)
        opp_disc = YELLOW if disc == RED else RED

        board_copy = [row[:] for row in self.board]
        column_heights = [self._column_height(c) for c in range(COLS)]

        my_threats = self._count_threats(disc)
        opp_threats = self._count_threats(opp_disc)

        center_col = 3
        my_center = sum(1 for r in range(ROWS) if self.board[r][center_col] == disc)
        opp_center = sum(1 for r in range(ROWS) if self.board[r][center_col] == opp_disc)

        my_pairs = self._count_connected_pairs(disc)
        opp_pairs = self._count_connected_pairs(opp_disc)

        return {
            "board": board_copy,
            "my_color": player,
            "move_number": self._move_count,
            "column_heights": column_heights,
            "my_threats_3": my_threats,
            "opponent_threats_3": opp_threats,
            "my_center_count": my_center,
            "opponent_center_count": opp_center,
            "my_connected_pairs": my_pairs,
            "opponent_connected_pairs": opp_pairs,
        }

    def _count_threats(self, disc: int) -> int:
        """Count windows of 4 where `disc` occupies exactly 3 and the 4th is
        empty AND playable (the empty cell is either on the bottom row or has
        a non-empty cell directly below it)."""
        count = 0
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in DIRECTIONS:
                    cells = []
                    valid = True
                    for i in range(4):
                        nr, nc = r + dr * i, c + dc * i
                        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                            valid = False
                            break
                        cells.append((nr, nc, self.board[nr][nc]))
                    if not valid or len(cells) != 4:
                        continue
                    discs_in_window = sum(1 for _, _, v in cells if v == disc)
                    empties_in_window = [(nr, nc) for nr, nc, v in cells if v == EMPTY]
                    if discs_in_window == 3 and len(empties_in_window) == 1:
                        er, ec = empties_in_window[0]
                        # The empty spot must be playable (supported from below)
                        if er == ROWS - 1 or self.board[er + 1][ec] != EMPTY:
                            count += 1
        return count

    def _count_connected_pairs(self, disc: int) -> int:
        """Count the number of adjacent same-color pairs (horizontal, vertical,
        diagonal). Each pair is counted once."""
        count = 0
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] != disc:
                    continue
                # Check right, down, down-right, down-left to avoid double-counting
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and self.board[nr][nc] == disc:
                        count += 1
        return count
