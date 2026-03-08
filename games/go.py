"""Go game engine. 9x9 board, Chinese/area scoring, komi 6.5."""

import re
from collections import deque
from typing import Any, Optional

from games.base import BaseGame

BOARD_SIZE = 9
TOTAL_POINTS = BOARD_SIZE * BOARD_SIZE
KOMI = 6.5
MAX_MOVES = 200

# Cell values
EMPTY = 0
BLACK = 1
WHITE = 2

# Four cardinal directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _color_name(color: int) -> str:
    return "black" if color == BLACK else "white"


def _color_from_name(name: str) -> int:
    return BLACK if name == "black" else WHITE


def _board_key(board: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Return a hashable representation of the board."""
    return tuple(tuple(row) for row in board)


class GoGame(BaseGame):
    """Complete 9x9 Go implementation with Chinese scoring."""

    def __init__(self):
        self.board: list[list[int]] = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn: str = "black"  # black goes first
        self._move_count: int = 0
        self._consecutive_passes: int = 0
        self._prev_board_key: Optional[tuple[tuple[int, ...], ...]] = None
        self._captured_by_black: int = 0  # stones black has captured from white
        self._captured_by_white: int = 0  # stones white has captured from black

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

    def copy(self) -> "GoGame":
        g = GoGame.__new__(GoGame)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g._move_count = self._move_count
        g._consecutive_passes = self._consecutive_passes
        g._prev_board_key = self._prev_board_key
        g._captured_by_black = self._captured_by_black
        g._captured_by_white = self._captured_by_white
        return g

    # ── Group and liberty logic ──────────────────────────────────────

    def _get_group(self, row: int, col: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """BFS to find the connected group containing (row, col) and its liberties.

        Returns (group_stones, liberties) where both are sets of (r, c) tuples.
        Returns empty sets if the cell is empty.
        """
        color = self.board[row][col]
        if color == EMPTY:
            return set(), set()

        group: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        queue = deque([(row, col)])
        group.add((row, col))

        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if (nr, nc) in group:
                    continue
                cell = self.board[nr][nc]
                if cell == EMPTY:
                    liberties.add((nr, nc))
                elif cell == color:
                    group.add((nr, nc))
                    queue.append((nr, nc))

        return group, liberties

    def _remove_group(self, group: set[tuple[int, int]]) -> int:
        """Remove all stones in the group from the board. Returns count removed."""
        for r, c in group:
            self.board[r][c] = EMPTY
        return len(group)

    def _is_legal_move(self, row: int, col: int, color: int) -> bool:
        """Check if placing `color` at (row, col) is legal.

        A move is illegal if:
        1. The cell is not empty.
        2. It results in suicide (own group has 0 liberties after captures).
        3. It violates the simple ko rule (recreates the previous board state).

        Uses a temporary board save/restore for correctness.
        """
        if self.board[row][col] != EMPTY:
            return False

        # Save board state
        saved_board = [r[:] for r in self.board]

        # Place stone
        self.board[row][col] = color
        opp = _opponent(color)

        # Remove captured opponent groups
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == opp:
                group, liberties = self._get_group(nr, nc)
                if not liberties:
                    self._remove_group(group)

        # Check suicide
        _, own_liberties = self._get_group(row, col)
        if not own_liberties:
            self.board = saved_board
            return False

        # Check simple ko
        new_key = _board_key(self.board)
        if self._prev_board_key is not None and new_key == self._prev_board_key:
            self.board = saved_board
            return False

        # Legal move - restore board (actual placement happens in apply_move)
        self.board = saved_board
        return True

    # ── Legal moves ──────────────────────────────────────────────────

    def get_legal_moves(self, player: str | None = None) -> list:
        """Return list of legal moves: list of (row, col) tuples + "pass".

        Pass is always legal.
        """
        if player is None:
            player = self.turn
        color = _color_from_name(player)

        moves: list = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._is_legal_move(r, c, color):
                    moves.append((r, c))
        moves.append("pass")
        return moves

    # ── Apply move ───────────────────────────────────────────────────

    def apply_move(self, move: Any) -> None:
        """Apply a move. `move` is (row, col) or "pass"."""
        if move == "pass":
            self._consecutive_passes += 1
            self._prev_board_key = _board_key(self.board)
            self._move_count += 1
            self.turn = "white" if self.turn == "black" else "black"
            return

        row, col = move
        color = _color_from_name(self.turn)
        opp = _opponent(color)

        if self.board[row][col] != EMPTY:
            raise ValueError(f"Illegal move ({row},{col}) for {self.turn}: cell not empty")

        # Save previous board state for ko
        prev_key = _board_key(self.board)

        # Place stone
        self.board[row][col] = color

        # Remove captured opponent groups
        total_captured = 0
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == opp:
                group, liberties = self._get_group(nr, nc)
                if not liberties:
                    total_captured += self._remove_group(group)

        # Check suicide (should not happen if move was validated)
        _, own_liberties = self._get_group(row, col)
        if not own_liberties:
            raise ValueError(f"Suicide move ({row},{col}) for {self.turn}")

        # Check ko (should not happen if move was validated)
        new_key = _board_key(self.board)
        if self._prev_board_key is not None and new_key == self._prev_board_key:
            raise ValueError(f"Ko violation at ({row},{col}) for {self.turn}")

        # Update capture counts
        if color == BLACK:
            self._captured_by_black += total_captured
        else:
            self._captured_by_white += total_captured

        self._prev_board_key = prev_key
        self._consecutive_passes = 0
        self._move_count += 1
        self.turn = "white" if self.turn == "black" else "black"

    # ── Scoring ──────────────────────────────────────────────────────

    def _count_stones(self) -> tuple[int, int]:
        """Return (black_stones, white_stones) on the board."""
        black_count = 0
        white_count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == BLACK:
                    black_count += 1
                elif self.board[r][c] == WHITE:
                    white_count += 1
        return black_count, white_count

    def _compute_territory(self) -> tuple[int, int]:
        """BFS from empty points to determine territory.

        An empty region belongs to a color if it is bordered only by stones
        of that color. Returns (black_territory, white_territory).
        """
        visited = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        black_territory = 0
        white_territory = 0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != EMPTY or visited[r][c]:
                    continue

                # BFS to find connected empty region
                region: list[tuple[int, int]] = []
                border_colors: set[int] = set()
                queue = deque([(r, c)])
                visited[r][c] = True

                while queue:
                    cr, cc = queue.popleft()
                    region.append((cr, cc))

                    for dr, dc in DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                            continue
                        cell = self.board[nr][nc]
                        if cell == EMPTY:
                            if not visited[nr][nc]:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                        else:
                            border_colors.add(cell)

                # Assign territory
                if border_colors == {BLACK}:
                    black_territory += len(region)
                elif border_colors == {WHITE}:
                    white_territory += len(region)
                # Mixed or no border: neutral, counts for neither

        return black_territory, white_territory

    def _compute_score(self) -> tuple[float, float]:
        """Chinese scoring: stones on board + territory.

        Returns (black_score, white_score) where white_score includes komi.
        """
        black_stones, white_stones = self._count_stones()
        black_territory, white_territory = self._compute_territory()
        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + KOMI
        return black_score, white_score

    # ── Group statistics ─────────────────────────────────────────────

    def _compute_group_stats(self, color: int) -> tuple[int, int, int]:
        """Compute group statistics for a color.

        Returns (num_groups, total_liberties, total_stones).
        """
        visited = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        num_groups = 0
        total_liberties = 0
        total_stones = 0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == color and not visited[r][c]:
                    group, liberties = self._get_group(r, c)
                    for gr, gc in group:
                        visited[gr][gc] = True
                    num_groups += 1
                    total_liberties += len(liberties)
                    total_stones += len(group)

        return num_groups, total_liberties, total_stones

    # ── Winner / end detection ───────────────────────────────────────

    def winner(self) -> Optional[str]:
        """Return 'black', 'white', 'draw', or None if game is ongoing."""
        # Safety cap
        if self._move_count >= MAX_MOVES:
            return "draw"

        # Two consecutive passes -> game over
        if self._consecutive_passes >= 2:
            black_score, white_score = self._compute_score()
            if black_score > white_score:
                return "black"
            elif white_score > black_score:
                return "white"
            return "draw"

        return None

    # ── State for strategies ─────────────────────────────────────────

    def get_state(self, player: str) -> dict:
        color = _color_from_name(player)
        opp = _opponent(color)

        black_stones, white_stones = self._count_stones()
        black_territory, white_territory = self._compute_territory()

        my_stones = black_stones if color == BLACK else white_stones
        opp_stones = white_stones if color == BLACK else black_stones
        my_territory = black_territory if color == BLACK else white_territory
        opp_territory = white_territory if color == BLACK else black_territory

        my_groups, my_liberties_total, _ = self._compute_group_stats(color)
        opp_groups, opp_liberties_total, _ = self._compute_group_stats(opp)

        my_captured = self._captured_by_black if color == BLACK else self._captured_by_white
        opp_captured = self._captured_by_white if color == BLACK else self._captured_by_black

        my_influence = (my_stones + my_territory) / TOTAL_POINTS

        # Estimated score from this player's perspective
        my_total = my_stones + my_territory
        opp_total = opp_stones + opp_territory
        # Add komi to opponent's score if player is black (white gets komi)
        if color == BLACK:
            estimated_score = float(my_total) - float(opp_total + KOMI)
        else:
            estimated_score = float(my_total + KOMI) - float(opp_total)

        return {
            "board": [row[:] for row in self.board],
            "my_color": player,
            "move_number": self._move_count,
            "board_size": BOARD_SIZE,
            "my_stones": my_stones,
            "opponent_stones": opp_stones,
            "my_territory": my_territory,
            "opponent_territory": opp_territory,
            "my_liberties_total": my_liberties_total,
            "opponent_liberties_total": opp_liberties_total,
            "my_groups": my_groups,
            "opponent_groups": opp_groups,
            "my_captured": my_captured,
            "opponent_captured": opp_captured,
            "my_influence": my_influence,
            "komi": KOMI,
            "estimated_score": estimated_score,
        }

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, perspective: str | None = None) -> str:
        symbols = {EMPTY: ".", BLACK: "B", WHITE: "W"}
        black_stones, white_stones = self._count_stones()

        header = "  " + " ".join(str(i) for i in range(BOARD_SIZE))
        info = f"    Black(B): {black_stones}  White(W): {white_stones}"
        info += f"  Captures: B={self._captured_by_black} W={self._captured_by_white}"
        lines = [header + info]
        for r in range(BOARD_SIZE):
            row_str = f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(BOARD_SIZE))
            lines.append(row_str)
        return "\n".join(lines)

    # ── Move string conversion ───────────────────────────────────────

    def move_to_str(self, move: Any) -> str:
        if move == "pass":
            return "pass"
        r, c = move
        return f"({r},{c})"

    def parse_move(self, text: str, player: str) -> Any:
        """Parse a move string from LLM output and match to a legal move.

        Accepts formats like:
          "(3,4)"  "3,4"  "(3, 4)"  "pass"
        """
        text = text.strip().lower()

        # Handle pass
        if "pass" in text:
            return "pass"

        # Try to extract two integers for (row, col)
        match = re.search(r'\(?\s*(\d)\s*[,\s]\s*(\d)\s*\)?', text)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                legal = self.get_legal_moves(player)
                if (r, c) in legal:
                    return (r, c)

        # Try all two-digit pairs in the text
        digits = re.findall(r'\d', text)
        if len(digits) >= 2:
            r, c = int(digits[0]), int(digits[1])
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                legal = self.get_legal_moves(player)
                if (r, c) in legal:
                    return (r, c)

        return None
