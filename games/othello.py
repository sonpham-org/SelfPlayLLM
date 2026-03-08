"""Othello/Reversi game engine. 8x8 board, black goes first."""

import re
from typing import Any, Optional

from games.base import BaseGame

BOARD_SIZE = 8

# Cell values
EMPTY = 0
BLACK = 1
WHITE = 2

# All eight directions: (dr, dc)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

# Corner positions
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# Edge positions (all cells on the border)
EDGES = set()
for i in range(BOARD_SIZE):
    EDGES.add((0, i))
    EDGES.add((7, i))
    EDGES.add((i, 0))
    EDGES.add((i, 7))


def _opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _color_name(color: int) -> str:
    return "black" if color == BLACK else "white"


def _color_from_name(name: str) -> int:
    return BLACK if name == "black" else WHITE


class OthelloGame(BaseGame):
    """Complete Othello/Reversi implementation."""

    def __init__(self):
        self.board: list[list[int]] = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn: str = "black"  # black goes first
        self._move_count: int = 0
        self._consecutive_passes: int = 0
        self._setup_board()

    def _setup_board(self):
        """Place the four starting discs in the center."""
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE

    # ── BaseGame properties ──────────────────────────────────────────

    @property
    def players(self) -> tuple[str, str]:
        return ("black", "white")

    @property
    def current_player(self) -> str:
        return self.turn

    @property
    def move_count(self) -> int:
        return self._move_count

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> "OthelloGame":
        g = OthelloGame.__new__(OthelloGame)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g._move_count = self._move_count
        g._consecutive_passes = self._consecutive_passes
        return g

    # ── Core logic ───────────────────────────────────────────────────

    def _flips_in_direction(self, row: int, col: int, dr: int, dc: int,
                            color: int) -> list[tuple[int, int]]:
        """Return list of opponent positions flipped by placing `color` at (row, col)
        along direction (dr, dc). Empty list if no flips."""
        opp = _opponent(color)
        flips: list[tuple[int, int]] = []
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self.board[r][c] == opp:
                flips.append((r, c))
            elif self.board[r][c] == color:
                return flips  # bracketed by own disc – valid
            else:
                return []  # hit empty – no flip
            r += dr
            c += dc
        return []  # ran off board – no flip

    def _all_flips(self, row: int, col: int, color: int) -> list[tuple[int, int]]:
        """Return all positions flipped by placing `color` at (row, col)."""
        if self.board[row][col] != EMPTY:
            return []
        flips: list[tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            flips.extend(self._flips_in_direction(row, col, dr, dc, color))
        return flips

    def _is_valid_move(self, row: int, col: int, color: int) -> bool:
        if self.board[row][col] != EMPTY:
            return False
        for dr, dc in DIRECTIONS:
            if self._flips_in_direction(row, col, dr, dc, color):
                return True
        return False

    # ── Legal moves ──────────────────────────────────────────────────

    def get_legal_moves(self, player: str | None = None) -> list[tuple[int, int]]:
        if player is None:
            player = self.turn
        color = _color_from_name(player)
        moves: list[tuple[int, int]] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._is_valid_move(r, c, color):
                    moves.append((r, c))
        return moves

    # ── Apply move ───────────────────────────────────────────────────

    def apply_move(self, move: Any) -> None:
        """Apply a move. `move` is (row, col) or None for a pass.

        Passes are handled automatically by the game when a player has no
        legal moves; callers should only pass None when get_legal_moves
        returns an empty list.
        """
        if move is None:
            # Pass – no legal moves for current player
            self._consecutive_passes += 1
            self._move_count += 1
            self.turn = "white" if self.turn == "black" else "black"
            return

        row, col = move
        color = _color_from_name(self.turn)
        flips = self._all_flips(row, col, color)
        if not flips:
            raise ValueError(f"Illegal move ({row},{col}) for {self.turn}")

        # Place disc
        self.board[row][col] = color
        # Flip captured discs
        for fr, fc in flips:
            self.board[fr][fc] = color

        self._consecutive_passes = 0
        self._move_count += 1
        self.turn = "white" if self.turn == "black" else "black"

    # ── Winner / end detection ───────────────────────────────────────

    def winner(self) -> Optional[str]:
        """Return 'black', 'white', 'draw', or None if game is ongoing."""
        # Safety cap
        if self._move_count >= 200:
            return "draw"

        # Two consecutive passes → game over
        if self._consecutive_passes >= 2:
            return self._disc_winner()

        # Board full → game over
        black_count, white_count, empty_count = self._disc_counts()
        if empty_count == 0:
            return self._disc_winner()

        # If current player has no moves, they must pass; if opponent also
        # has none the game is over – but we don't auto-advance here,
        # we let the caller handle pass moves. However we check if BOTH
        # players have no moves (game should end even without passes).
        if not self.get_legal_moves("black") and not self.get_legal_moves("white"):
            return self._disc_winner()

        return None

    def _disc_counts(self) -> tuple[int, int, int]:
        black_count = 0
        white_count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == BLACK:
                    black_count += 1
                elif self.board[r][c] == WHITE:
                    white_count += 1
        return black_count, white_count, BOARD_SIZE * BOARD_SIZE - black_count - white_count

    def _disc_winner(self) -> str:
        black_count, white_count, _ = self._disc_counts()
        if black_count > white_count:
            return "black"
        elif white_count > black_count:
            return "white"
        return "draw"

    # ── State for strategies ─────────────────────────────────────────

    def get_state(self, player: str) -> dict:
        color = _color_from_name(player)
        opp = _opponent(color)

        black_count, white_count, empty_count = self._disc_counts()
        my_discs = black_count if color == BLACK else white_count
        opp_discs = white_count if color == BLACK else black_count

        my_corners = sum(1 for r, c in CORNERS if self.board[r][c] == color)
        opp_corners = sum(1 for r, c in CORNERS if self.board[r][c] == opp)

        my_edge = sum(1 for r, c in EDGES if self.board[r][c] == color)
        opp_edge = sum(1 for r, c in EDGES if self.board[r][c] == opp)

        my_mobility = len(self.get_legal_moves(player))
        opp_name = "white" if player == "black" else "black"
        opp_mobility = len(self.get_legal_moves(opp_name))

        my_stable = self._count_stable_discs(color)
        frontier = self._count_frontier_discs(color)

        return {
            "board": [row[:] for row in self.board],
            "my_color": player,
            "move_number": self._move_count,
            "my_discs": my_discs,
            "opponent_discs": opp_discs,
            "empty_squares": empty_count,
            "my_corners": my_corners,
            "opponent_corners": opp_corners,
            "my_edge_count": my_edge,
            "opponent_edge_count": opp_edge,
            "my_mobility": my_mobility,
            "opponent_mobility": opp_mobility,
            "my_stable_discs": my_stable,
            "frontier_discs": frontier,
        }

    # ── Stable disc computation ──────────────────────────────────────

    def _count_stable_discs(self, color: int) -> int:
        """Count discs that can never be flipped.

        A disc is stable if, for every one of the four axis pairs
        (horizontal, vertical, and both diagonals), the line through
        that disc in that axis is fully filled OR the disc is anchored
        to the edge/corner in that axis direction.

        We use a conservative flood-fill from corners: a disc is
        stable if it is the player's color and for every axis, it is
        either on the board edge in that direction or adjacent (in
        that direction) to a disc already known to be stable.
        """
        stable = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        # Four axis pairs: each pair is two opposite directions
        axes = [
            (( 0,  1), ( 0, -1)),  # horizontal
            (( 1,  0), (-1,  0)),  # vertical
            (( 1,  1), (-1, -1)),  # diagonal ↘↖
            (( 1, -1), (-1,  1)),  # diagonal ↙↗
        ]
        changed = True
        while changed:
            changed = False
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if stable[r][c] or self.board[r][c] != color:
                        continue
                    # Check if stable along all four axes
                    is_stable = True
                    for (d1r, d1c), (d2r, d2c) in axes:
                        # Disc is stable along this axis if:
                        #   direction 1 is edge or stable neighbor, AND
                        #   direction 2 is edge or stable neighbor
                        # OR the entire line in this axis is filled (no empties)
                        n1r, n1c = r + d1r, c + d1c
                        n2r, n2c = r + d2r, c + d2c

                        dir1_ok = (not (0 <= n1r < BOARD_SIZE and 0 <= n1c < BOARD_SIZE)
                                   or (self.board[n1r][n1c] == color and stable[n1r][n1c]))
                        dir2_ok = (not (0 <= n2r < BOARD_SIZE and 0 <= n2c < BOARD_SIZE)
                                   or (self.board[n2r][n2c] == color and stable[n2r][n2c]))

                        if not (dir1_ok and dir2_ok):
                            # Fallback: check if entire line along this axis is filled
                            line_full = True
                            for dr, dc in [(d1r, d1c), (d2r, d2c)]:
                                tr, tc = r + dr, c + dc
                                while 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE:
                                    if self.board[tr][tc] == EMPTY:
                                        line_full = False
                                        break
                                    tr += dr
                                    tc += dc
                                if not line_full:
                                    break
                            if not line_full:
                                is_stable = False
                                break
                    if is_stable:
                        stable[r][c] = True
                        changed = True

        return sum(stable[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))

    def _count_frontier_discs(self, color: int) -> int:
        """Count player's discs adjacent to at least one empty square."""
        count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != color:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and self.board[nr][nc] == EMPTY):
                        count += 1
                        break
        return count

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, perspective: str | None = None) -> str:
        symbols = {EMPTY: ".", BLACK: "B", WHITE: "W"}
        black_count, white_count, _ = self._disc_counts()
        lines = [f"  0 1 2 3 4 5 6 7    Black(B): {black_count}  White(W): {white_count}"]
        for r in range(BOARD_SIZE):
            row_str = f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(BOARD_SIZE))
            lines.append(row_str)
        return "\n".join(lines)

    # ── Move string conversion ───────────────────────────────────────

    def move_to_str(self, move: Any) -> str:
        if move is None:
            return "pass"
        r, c = move
        return f"({r},{c})"

    def parse_move(self, text: str, player: str) -> Optional[tuple[int, int]]:
        """Parse a move string from LLM output and match to a legal move.

        Accepts formats like:
          "(3,4)"  "3,4"  "(3, 4)"  "row 3, col 4"  "3 4"  "pass"
        """
        text = text.strip().lower()

        # Handle pass
        if "pass" in text:
            legal = self.get_legal_moves(player)
            if not legal:
                return None  # None signals a pass
            # Player has legal moves, cannot pass
            return None

        # Try to extract two integers
        # Pattern 1: (row, col) or row,col
        match = re.search(r'\(?\s*(\d)\s*[,\s]\s*(\d)\s*\)?', text)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            legal = self.get_legal_moves(player)
            if (r, c) in legal:
                return (r, c)

        # Try all two-digit pairs in the text
        digits = re.findall(r'\d', text)
        if len(digits) >= 2:
            r, c = int(digits[0]), int(digits[1])
            legal = self.get_legal_moves(player)
            if (r, c) in legal:
                return (r, c)

        return None
