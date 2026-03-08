"""Dots and Boxes game engine. 4x4 dot grid (3x3 boxes), two players, extra turn on box completion."""

import re
from typing import Any, Optional

from games.base import BaseGame


class DotsAndBoxesGame(BaseGame):
    """4x4 Dots and Boxes: 9 boxes, 24 lines, extra turns for completing boxes.

    Dots are on a 4x4 grid (rows 0-3, cols 0-3).
    Horizontal lines connect (r,c)-(r,c+1): 4 rows x 3 cols = 12 lines.
    Vertical lines connect (r,c)-(r+1,c): 3 rows x 4 cols = 12 lines.
    Total: 24 lines.

    A box at position (r,c) (0-indexed, 3x3) is bounded by:
        top:    h_lines[r][c]
        bottom: h_lines[r+1][c]
        left:   v_lines[r][c]
        right:  v_lines[r][c+1]
    """

    ROWS = 4  # dot grid rows
    COLS = 4  # dot grid cols
    BOX_ROWS = 3  # ROWS - 1
    BOX_COLS = 3  # COLS - 1
    TOTAL_LINES = 24  # 2 * ROWS * (COLS - 1) ... well, 4*3 + 3*4 = 24

    def __init__(self):
        # Horizontal lines: h_lines[r][c] = line between dot (r,c) and (r,c+1)
        self.h_lines = [[False] * self.BOX_COLS for _ in range(self.ROWS)]
        # Vertical lines: v_lines[r][c] = line between dot (r,c) and (r+1,c)
        self.v_lines = [[False] * self.COLS for _ in range(self.BOX_ROWS)]
        # Box owners: box_owners[r][c] = "player1", "player2", or None
        self.box_owners = [[None] * self.BOX_COLS for _ in range(self.BOX_ROWS)]
        self._current_player = "player1"
        self._move_count = 0
        self._scores = {"player1": 0, "player2": 0}
        self._lines_placed = 0

    # ---- BaseGame properties ----

    @property
    def players(self) -> tuple[str, str]:
        return ("player1", "player2")

    @property
    def current_player(self) -> str:
        return self._current_player

    @property
    def move_count(self) -> int:
        return self._move_count

    # ---- Core helpers ----

    def _other_player(self, player: str) -> str:
        return "player2" if player == "player1" else "player1"

    def _box_sides(self, r: int, c: int) -> int:
        """Count how many sides of box (r,c) are drawn."""
        count = 0
        if self.h_lines[r][c]:      # top
            count += 1
        if self.h_lines[r + 1][c]:  # bottom
            count += 1
        if self.v_lines[r][c]:      # left
            count += 1
        if self.v_lines[r][c + 1]:  # right
            count += 1
        return count

    def _box_complete(self, r: int, c: int) -> bool:
        return self._box_sides(r, c) == 4

    def _boxes_completed_by_line(self, line: tuple) -> list[tuple[int, int]]:
        """Given a line (as a canonical tuple), return which box positions it would complete."""
        r1, c1, r2, c2 = line
        completed = []
        if r1 == r2:
            # Horizontal line between (r1,c1) and (r1,c2) where c2 = c1+1
            c = min(c1, c2)
            r = r1
            # Box above: box (r-1, c)
            if r - 1 >= 0 and r - 1 < self.BOX_ROWS and c < self.BOX_COLS:
                if self._box_sides(r - 1, c) == 3:
                    # This line is the bottom of that box; check the other 3
                    # Actually we need to check if adding THIS line completes it
                    # _box_sides counts current state; the line isn't placed yet
                    # so if it currently has 3, placing this makes 4
                    completed.append((r - 1, c))
            # Box below: box (r, c)
            if r < self.BOX_ROWS and c < self.BOX_COLS:
                if self._box_sides(r, c) == 3:
                    completed.append((r, c))
        else:
            # Vertical line between (r1,c1) and (r2,c1) where r2 = r1+1
            r = min(r1, r2)
            c = c1
            # Box to the left: box (r, c-1)
            if c - 1 >= 0 and c - 1 < self.BOX_COLS and r < self.BOX_ROWS:
                if self._box_sides(r, c - 1) == 3:
                    completed.append((r, c - 1))
            # Box to the right: box (r, c)
            if c < self.BOX_COLS and r < self.BOX_ROWS:
                if self._box_sides(r, c) == 3:
                    completed.append((r, c))
        return completed

    def _line_exists(self, line: tuple) -> bool:
        """Check if a line (r1,c1,r2,c2) is already placed."""
        r1, c1, r2, c2 = line
        if r1 == r2:
            c = min(c1, c2)
            return self.h_lines[r1][c]
        else:
            r = min(r1, r2)
            return self.v_lines[r][c1]

    def _place_line(self, line: tuple) -> None:
        """Place a line on the board."""
        r1, c1, r2, c2 = line
        if r1 == r2:
            c = min(c1, c2)
            self.h_lines[r1][c] = True
        else:
            r = min(r1, r2)
            self.v_lines[r][c1] = True
        self._lines_placed += 1

    @staticmethod
    def _canonicalize(r1: int, c1: int, r2: int, c2: int) -> tuple:
        """Normalize a line so the smaller dot comes first."""
        if (r1, c1) > (r2, c2):
            r1, c1, r2, c2 = r2, c2, r1, c1
        return (r1, c1, r2, c2)

    @staticmethod
    def _is_adjacent(r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if two dots are horizontally or vertically adjacent."""
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

    @staticmethod
    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < DotsAndBoxesGame.ROWS and 0 <= c < DotsAndBoxesGame.COLS

    # ---- BaseGame abstract methods ----

    def copy(self) -> "DotsAndBoxesGame":
        g = DotsAndBoxesGame.__new__(DotsAndBoxesGame)
        g.h_lines = [row[:] for row in self.h_lines]
        g.v_lines = [row[:] for row in self.v_lines]
        g.box_owners = [row[:] for row in self.box_owners]
        g._current_player = self._current_player
        g._move_count = self._move_count
        g._scores = dict(self._scores)
        g._lines_placed = self._lines_placed
        return g

    def get_legal_moves(self, player: str | None = None) -> list:
        """Return all unplaced lines as canonical tuples (r1,c1,r2,c2)."""
        moves = []
        # Horizontal lines
        for r in range(self.ROWS):
            for c in range(self.BOX_COLS):
                if not self.h_lines[r][c]:
                    moves.append((r, c, r, c + 1))
        # Vertical lines
        for r in range(self.BOX_ROWS):
            for c in range(self.COLS):
                if not self.v_lines[r][c]:
                    moves.append((r, c, r + 1, c))
        return moves

    def apply_move(self, move: Any) -> None:
        """Apply a line placement. If it completes box(es), current player scores and keeps turn."""
        r1, c1, r2, c2 = move
        line = self._canonicalize(r1, c1, r2, c2)

        if self._line_exists(line):
            raise ValueError(f"Line {line} already exists")
        if not self._is_adjacent(r1, c1, r2, c2):
            raise ValueError(f"Dots ({r1},{c1}) and ({r2},{c2}) are not adjacent")
        if not (self._in_bounds(r1, c1) and self._in_bounds(r2, c2)):
            raise ValueError(f"Dots out of bounds")

        # Check which boxes this line will complete (before placing)
        completed = self._boxes_completed_by_line(line)

        # Place the line
        self._place_line(line)
        self._move_count += 1

        # Score completed boxes
        for br, bc in completed:
            self.box_owners[br][bc] = self._current_player
            self._scores[self._current_player] += 1

        # If no box was completed, switch turns
        if not completed:
            self._current_player = self._other_player(self._current_player)
        # Otherwise current player gets another turn (stays the same)

    def winner(self) -> Optional[str]:
        """Return winner, 'draw', or None if game ongoing."""
        if self._lines_placed < self.TOTAL_LINES:
            return None
        s1 = self._scores["player1"]
        s2 = self._scores["player2"]
        if s1 > s2:
            return "player1"
        elif s2 > s1:
            return "player2"
        else:
            return "draw"

    def render(self, perspective: str | None = None) -> str:
        """ASCII art rendering of the board.

        Example:
            *---*   *   *
            |   |
            * P1*---*   *
                    |
            *   *   *---*
            |
            *---*   *   *
        """
        lines_out = []
        for r in range(self.ROWS):
            # Dot row: dots and horizontal lines
            row_chars = []
            for c in range(self.COLS):
                row_chars.append("*")
                if c < self.BOX_COLS:
                    if self.h_lines[r][c]:
                        row_chars.append("---")
                    else:
                        row_chars.append("   ")
            lines_out.append("".join(row_chars))

            # Vertical line / box owner row (between dot rows)
            if r < self.BOX_ROWS:
                row_chars = []
                for c in range(self.COLS):
                    if self.v_lines[r][c]:
                        row_chars.append("|")
                    else:
                        row_chars.append(" ")
                    if c < self.BOX_COLS:
                        owner = self.box_owners[r][c]
                        if owner == "player1":
                            row_chars.append(" P1")
                        elif owner == "player2":
                            row_chars.append(" P2")
                        else:
                            row_chars.append("   ")
                lines_out.append("".join(row_chars))

        return "\n".join(lines_out)

    def get_state(self, player: str) -> dict:
        """Return full game state dict for strategy code."""
        opponent = self._other_player(player)

        # Compute chains: sequences of connected boxes each with exactly 3 sides
        chains, chain_lengths = self._compute_chains()

        # Count safe moves: lines that don't give opponent a box
        # (i.e., don't create a box with 3 sides that the opponent could then complete)
        safe_moves = self._count_safe_moves()

        # Sacrifice opportunities: boxes that have 3 sides drawn (current player
        # could complete them but might choose to sacrifice)
        sacrifice_opportunities = self._count_sacrifice_opportunities()

        return {
            "board": {
                "h_lines": [row[:] for row in self.h_lines],
                "v_lines": [row[:] for row in self.v_lines],
                "box_owners": [row[:] for row in self.box_owners],
            },
            "my_color": player,
            "move_number": self._move_count,
            "my_boxes": self._scores[player],
            "opponent_boxes": self._scores[opponent],
            "lines_remaining": self.TOTAL_LINES - self._lines_placed,
            "total_lines": self.TOTAL_LINES,
            "chains": chains,
            "chain_lengths": chain_lengths,
            "safe_moves": safe_moves,
            "sacrifice_opportunities": sacrifice_opportunities,
        }

    def _compute_chains(self) -> tuple[int, list[int]]:
        """Find chains: connected groups of boxes that each have exactly 3 sides drawn.

        A chain is a maximal connected sequence of boxes with 3 sides. Boxes are
        connected if they share an undrawn edge (which is the edge that would
        complete both of them in sequence).
        """
        # Find all boxes with exactly 3 sides
        three_side_boxes = set()
        for r in range(self.BOX_ROWS):
            for c in range(self.BOX_COLS):
                if self.box_owners[r][c] is None and self._box_sides(r, c) == 3:
                    three_side_boxes.add((r, c))

        if not three_side_boxes:
            return 0, []

        # BFS/DFS to find connected components among 3-side boxes
        # Two 3-side boxes are connected if they are adjacent and share an undrawn edge
        visited = set()
        chain_lengths = []

        for box in three_side_boxes:
            if box in visited:
                continue
            # BFS from this box
            queue = [box]
            visited.add(box)
            count = 0
            while queue:
                br, bc = queue.pop()
                count += 1
                # Check 4 neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = br + dr, bc + dc
                    if (nr, nc) in three_side_boxes and (nr, nc) not in visited:
                        # Check if they share an undrawn edge
                        if self._share_undrawn_edge(br, bc, nr, nc):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            chain_lengths.append(count)

        return len(chain_lengths), sorted(chain_lengths, reverse=True)

    def _share_undrawn_edge(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if two adjacent boxes share an undrawn edge between them."""
        if r1 == r2:
            # Horizontally adjacent
            c_left = min(c1, c2)
            # The edge between them is v_lines[r1][c_left + 1]
            return not self.v_lines[r1][c_left + 1]
        elif c1 == c2:
            # Vertically adjacent
            r_top = min(r1, r2)
            # The edge between them is h_lines[r_top + 1][c1]
            return not self.h_lines[r_top + 1][c1]
        return False

    def _count_safe_moves(self) -> int:
        """Count moves that don't give the opponent any box to complete next turn.

        A move is 'safe' if placing it does not create any box with exactly 3 sides
        (which the opponent could then complete). Note: if the move itself completes
        a box (the current box already has 3 sides), that's actually good for the
        current player, so we only count it as unsafe if it brings a box to exactly
        3 sides without completing it.
        """
        count = 0
        for move in self.get_legal_moves():
            if self._is_safe_move(move):
                count += 1
        return count

    def _is_safe_move(self, move: tuple) -> bool:
        """Check if placing this line is safe (doesn't give opponent a 3-side box)."""
        r1, c1, r2, c2 = self._canonicalize(*move)
        # Find which boxes this line borders
        adjacent_boxes = self._line_adjacent_boxes(r1, c1, r2, c2)
        for br, bc in adjacent_boxes:
            if self.box_owners[br][bc] is not None:
                continue
            sides = self._box_sides(br, bc)
            # This line will add one side. If currently 2, it becomes 3 (unsafe).
            # If currently 3, it becomes 4 (completed by us, that's fine).
            if sides == 2:
                return False
        return True

    def _line_adjacent_boxes(self, r1: int, c1: int, r2: int, c2: int) -> list[tuple[int, int]]:
        """Return box positions adjacent to a given line."""
        boxes = []
        if r1 == r2:
            # Horizontal line
            c = min(c1, c2)
            r = r1
            if r - 1 >= 0 and r - 1 < self.BOX_ROWS and c < self.BOX_COLS:
                boxes.append((r - 1, c))
            if r < self.BOX_ROWS and c < self.BOX_COLS:
                boxes.append((r, c))
        else:
            # Vertical line
            r = min(r1, r2)
            c = c1
            if c - 1 >= 0 and c - 1 < self.BOX_COLS and r < self.BOX_ROWS:
                boxes.append((r, c - 1))
            if c < self.BOX_COLS and r < self.BOX_ROWS:
                boxes.append((r, c))
        return boxes

    def _count_sacrifice_opportunities(self) -> int:
        """Count boxes with exactly 3 sides drawn (available to complete right now)."""
        count = 0
        for r in range(self.BOX_ROWS):
            for c in range(self.BOX_COLS):
                if self.box_owners[r][c] is None and self._box_sides(r, c) == 3:
                    count += 1
        return count

    def move_to_str(self, move: Any) -> str:
        r1, c1, r2, c2 = move
        return f"({r1},{c1})-({r2},{c2})"

    def parse_move(self, text: str, player: str) -> Any:
        """Parse move string like '(0,0)-(0,1)' and match to a legal move.

        Tolerant parsing: finds coordinate pairs in the text and tries to match.
        """
        # Try to find pattern like (r1,c1)-(r2,c2)
        pattern = r'\((\d+)\s*,\s*(\d+)\)\s*-\s*\((\d+)\s*,\s*(\d+)\)'
        match = re.search(pattern, text)
        if not match:
            return None

        r1, c1, r2, c2 = int(match.group(1)), int(match.group(2)), \
                          int(match.group(3)), int(match.group(4))

        if not (self._in_bounds(r1, c1) and self._in_bounds(r2, c2)):
            return None
        if not self._is_adjacent(r1, c1, r2, c2):
            return None

        canonical = self._canonicalize(r1, c1, r2, c2)
        legal = self.get_legal_moves()
        for m in legal:
            if self._canonicalize(*m) == canonical:
                return m
        return None
