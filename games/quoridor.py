"""Quoridor game engine. 9x9 board, pawn movement, wall placement, path validation."""

import re
from collections import deque
from typing import Any, Optional

from games.base import BaseGame

BOARD_SIZE = 9
MAX_MOVES = 200
WALLS_PER_PLAYER = 10


class QuoridorGame(BaseGame):
    """Standard Quoridor on a 9x9 board.

    Player1 starts at row 8 (bottom), wins by reaching row 0 (top).
    Player2 starts at row 0 (top), wins by reaching row 8 (bottom).
    Player1 goes first. Each turn a player either moves their pawn one
    step in a cardinal direction or places a wall (horizontal or vertical)
    that spans two cell-edges. Walls must not overlap and must not block
    all paths to the goal for either player.
    """

    def __init__(self):
        self.pos: dict[str, tuple[int, int]] = {
            "player1": (8, 4),
            "player2": (0, 4),
        }
        self.walls_left: dict[str, int] = {
            "player1": WALLS_PER_PLAYER,
            "player2": WALLS_PER_PLAYER,
        }
        # Placed walls stored as set of (orientation, row, col) tuples.
        # 'h' wall at (r, c): blocks movement between rows r and r+1
        #   at columns c and c+1.
        # 'v' wall at (r, c): blocks movement between columns c and c+1
        #   at rows r and r+1.
        self.walls: set[tuple[str, int, int]] = set()
        self.turn: str = "player1"
        self._move_count: int = 0

    # ── BaseGame properties ──────────────────────────────────────────

    @property
    def players(self) -> tuple[str, str]:
        return ("player1", "player2")

    @property
    def current_player(self) -> str:
        return self.turn

    @property
    def move_count(self) -> int:
        return self._move_count

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> "QuoridorGame":
        g = QuoridorGame.__new__(QuoridorGame)
        g.pos = dict(self.pos)
        g.walls_left = dict(self.walls_left)
        g.walls = set(self.walls)
        g.turn = self.turn
        g._move_count = self._move_count
        return g

    # ── Wall / movement helpers ──────────────────────────────────────

    def _is_blocked(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Return True if a wall blocks movement from (r1,c1) to (r2,c2).

        Assumes (r1,c1) and (r2,c2) are orthogonally adjacent.
        """
        if r2 == r1 - 1 and c2 == c1:
            # Moving up: blocked by horizontal wall at (r1-1, c1) or (r1-1, c1-1)
            return (("h", r1 - 1, c1) in self.walls or
                    ("h", r1 - 1, c1 - 1) in self.walls)
        if r2 == r1 + 1 and c2 == c1:
            # Moving down: blocked by horizontal wall at (r1, c1) or (r1, c1-1)
            return (("h", r1, c1) in self.walls or
                    ("h", r1, c1 - 1) in self.walls)
        if c2 == c1 - 1 and r2 == r1:
            # Moving left: blocked by vertical wall at (r1, c1-1) or (r1-1, c1-1)
            return (("v", r1, c1 - 1) in self.walls or
                    ("v", r1 - 1, c1 - 1) in self.walls)
        if c2 == c1 + 1 and r2 == r1:
            # Moving right: blocked by vertical wall at (r1, c1) or (r1-1, c1)
            return (("v", r1, c1) in self.walls or
                    ("v", r1 - 1, c1) in self.walls)
        return False

    def _can_move(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if a pawn can step from (r1,c1) to (r2,c2) — in bounds
        and not blocked by a wall."""
        if not (0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE):
            return False
        return not self._is_blocked(r1, c1, r2, c2)

    def _bfs_shortest_path(self, start: tuple[int, int],
                           goal_row: int,
                           occupied: set[tuple[int, int]] | None = None) -> int:
        """BFS from start to any cell in goal_row, ignoring pawns.

        Returns the number of steps, or -1 if unreachable.
        `occupied` is ignored for path-existence checks (walls only block).
        """
        if start[0] == goal_row:
            return 0
        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if self._is_blocked(r, c, nr, nc):
                    continue
                if nr == goal_row:
                    return dist + 1
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
        return -1

    def _both_players_can_reach_goal(self) -> bool:
        """Return True if both players still have a path to their goal row."""
        p1_dist = self._bfs_shortest_path(self.pos["player1"], 0)
        if p1_dist == -1:
            return False
        p2_dist = self._bfs_shortest_path(self.pos["player2"], 8)
        return p2_dist != -1

    def _wall_overlaps(self, orientation: str, r: int, c: int) -> bool:
        """Check whether placing wall (orientation, r, c) overlaps any
        existing wall."""
        # Exact duplicate
        if (orientation, r, c) in self.walls:
            return True
        # Same-orientation overlap: two walls of the same type that share
        # a segment.
        if orientation == "h":
            if ("h", r, c - 1) in self.walls or ("h", r, c + 1) in self.walls:
                return True
        else:
            if ("v", r - 1, c) in self.walls or ("v", r + 1, c) in self.walls:
                return True
        # Cross overlap: an h-wall and a v-wall that share the same
        # center point.  h(r,c) occupies center (r, c+0.5); v(r,c)
        # occupies center (r+0.5, c).  They cross when h_center ==
        # v_center, i.e., the h-wall at (r,c) crosses with v-wall at
        # (r, c) (since h-center = (r, c+0.5) and v-center = (r+0.5, c)
        # — that's NOT the same). The actual crossing condition:
        # h(rh, ch) center is at gap (rh, ch) to (rh, ch+1)
        # v(rv, cv) center is at gap (rv, cv) to (rv+1, cv)
        # They cross when rh == rv and ch == cv.
        if orientation == "h":
            if ("v", r, c) in self.walls:
                return True
        else:
            if ("h", r, c) in self.walls:
                return True
        return False

    # ── Pawn move generation ─────────────────────────────────────────

    def _get_pawn_moves(self, player: str) -> list[str]:
        """Return list of legal pawn-move strings for `player`."""
        my_pos = self.pos[player]
        opp = "player2" if player == "player1" else "player1"
        opp_pos = self.pos[opp]
        r, c = my_pos
        moves: list[str] = []

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue
            if self._is_blocked(r, c, nr, nc):
                continue
            if (nr, nc) != opp_pos:
                # Normal step
                moves.append(f"move ({nr},{nc})")
            else:
                # Opponent is adjacent — try to jump straight over
                jr, jc = nr + dr, nc + dc
                if (0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE
                        and not self._is_blocked(nr, nc, jr, jc)):
                    moves.append(f"move ({jr},{jc})")
                else:
                    # Can't jump straight — try diagonal jumps
                    for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        if (ddr, ddc) == (dr, dc) or (ddr, ddc) == (-dr, -dc):
                            continue
                        sr, sc = nr + ddr, nc + ddc
                        if (0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE
                                and not self._is_blocked(nr, nc, sr, sc)
                                and (sr, sc) != opp_pos):
                            moves.append(f"move ({sr},{sc})")
        return moves

    # ── Wall move generation ─────────────────────────────────────────

    def _get_wall_moves(self, player: str) -> list[str]:
        """Return list of legal wall-placement strings for `player`."""
        if self.walls_left[player] <= 0:
            return []
        moves: list[str] = []
        for r in range(BOARD_SIZE - 1):
            for c in range(BOARD_SIZE - 1):
                for orientation in ("h", "v"):
                    if self._wall_overlaps(orientation, r, c):
                        continue
                    # Tentatively place and check paths
                    self.walls.add((orientation, r, c))
                    if self._both_players_can_reach_goal():
                        moves.append(f"wall {orientation} ({r},{c})")
                    self.walls.remove((orientation, r, c))
        return moves

    # ── Core game interface ──────────────────────────────────────────

    def get_legal_moves(self, player: str | None = None) -> list[str]:
        """Return all legal moves (pawn moves + wall placements)."""
        if player is None:
            player = self.turn
        pawn_moves = self._get_pawn_moves(player)
        wall_moves = self._get_wall_moves(player)
        return pawn_moves + wall_moves

    def apply_move(self, move: Any) -> None:
        """Apply a move string, update state, switch turns."""
        move_str = str(move).strip()
        player = self.turn
        opp = "player2" if player == "player1" else "player1"

        if move_str.startswith("move"):
            m = re.search(r'\((\d+)\s*,\s*(\d+)\)', move_str)
            if m is None:
                raise ValueError(f"Cannot parse pawn move: {move_str}")
            nr, nc = int(m.group(1)), int(m.group(2))
            self.pos[player] = (nr, nc)
        elif move_str.startswith("wall"):
            m = re.match(r'wall\s+([hv])\s*\((\d+)\s*,\s*(\d+)\)', move_str)
            if m is None:
                raise ValueError(f"Cannot parse wall move: {move_str}")
            orientation = m.group(1)
            wr, wc = int(m.group(2)), int(m.group(3))
            self.walls.add((orientation, wr, wc))
            self.walls_left[player] -= 1
        else:
            raise ValueError(f"Unknown move format: {move_str}")

        self._move_count += 1
        self.turn = opp

    def winner(self) -> Optional[str]:
        """Return 'player1', 'player2', 'draw', or None if game ongoing."""
        if self.pos["player1"][0] == 0:
            return "player1"
        if self.pos["player2"][0] == 8:
            return "player2"
        if self._move_count >= MAX_MOVES:
            return "draw"
        return None

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, perspective: str | None = None) -> str:
        """ASCII board with pawns (1, 2), walls, and grid.

        The rendered board interleaves cell rows with gap rows.
        Cell rows show pawns and horizontal spaces; gap rows show
        horizontal walls (═) and vertical walls (║).
        """
        # Build a (2*BOARD_SIZE - 1) x (2*BOARD_SIZE - 1) character grid.
        H = 2 * BOARD_SIZE - 1
        W = 2 * BOARD_SIZE - 1
        grid: list[list[str]] = [[" "] * W for _ in range(H)]

        # Fill cell positions with dots
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                grid[2 * r][2 * c] = "."

        # Place pawns
        pr1, pc1 = self.pos["player1"]
        grid[2 * pr1][2 * pc1] = "1"
        pr2, pc2 = self.pos["player2"]
        grid[2 * pr2][2 * pc2] = "2"

        # Place walls
        for orientation, wr, wc in self.walls:
            if orientation == "h":
                # Horizontal wall at gap row (2*wr + 1), spanning
                # columns 2*wc through 2*(wc+1).
                gr = 2 * wr + 1
                for gc in range(2 * wc, 2 * (wc + 1) + 1):
                    if 0 <= gc < W:
                        grid[gr][gc] = "\u2550"  # ═
            else:
                # Vertical wall at gap column (2*wc + 1), spanning
                # rows 2*wr through 2*(wr+1).
                gc_wall = 2 * wc + 1
                for gr in range(2 * wr, 2 * (wr + 1) + 1):
                    if 0 <= gr < H:
                        grid[gr][gc_wall] = "\u2551"  # ║

        # Column header
        header = "    " + "   ".join(str(c) for c in range(BOARD_SIZE))
        lines = [header]
        for gr in range(H):
            if gr % 2 == 0:
                row_label = f" {gr // 2}  "
            else:
                row_label = "    "
            line = row_label + "  ".join(grid[gr][gc] for gc in range(W))
            lines.append(line)
        return "\n".join(lines)

    # ── Move parsing / formatting ────────────────────────────────────

    def move_to_str(self, move: Any) -> str:
        return str(move)

    def parse_move(self, text: str, player: str) -> Optional[str]:
        """Parse an LLM output into a legal move string.

        Accepts formats like:
          move (7,4)
          wall h (3,4)
          wall v (2,5)
        """
        text = text.strip()
        legal = self.get_legal_moves(player)
        legal_set = set(legal)

        # Try direct match after normalization
        normalized = re.sub(r'\s+', ' ', text)
        if normalized in legal_set:
            return normalized

        # Try to extract a pawn move
        m = re.search(r'move\s*\((\d+)\s*,\s*(\d+)\)', text, re.IGNORECASE)
        if m:
            candidate = f"move ({m.group(1)},{m.group(2)})"
            if candidate in legal_set:
                return candidate

        # Try to extract a wall move
        m = re.search(r'wall\s+([hHvV])\s*\((\d+)\s*,\s*(\d+)\)', text, re.IGNORECASE)
        if m:
            orientation = m.group(1).lower()
            candidate = f"wall {orientation} ({m.group(2)},{m.group(3)})"
            if candidate in legal_set:
                return candidate

        return None

    # ── State for strategies ─────────────────────────────────────────

    def get_state(self, player: str) -> dict:
        opp = "player2" if player == "player1" else "player1"
        my_pos = self.pos[player]
        opp_pos = self.pos[opp]

        my_goal_row = 0 if player == "player1" else 8
        opp_goal_row = 0 if opp == "player1" else 8

        my_shortest = self._bfs_shortest_path(my_pos, my_goal_row)
        opp_shortest = self._bfs_shortest_path(opp_pos, opp_goal_row)

        my_manhattan = abs(my_pos[0] - my_goal_row)
        opp_manhattan = abs(opp_pos[0] - opp_goal_row)

        return {
            "board": {
                "my_pos": my_pos,
                "opponent_pos": opp_pos,
                "walls": sorted(self.walls),
            },
            "my_color": player,
            "move_number": self._move_count,
            "my_walls_left": self.walls_left[player],
            "opponent_walls_left": self.walls_left[opp],
            "my_shortest_path": my_shortest,
            "opponent_shortest_path": opp_shortest,
            "my_distance_to_goal": my_manhattan,
            "opponent_distance_to_goal": opp_manhattan,
            "total_walls_placed": (WALLS_PER_PLAYER - self.walls_left["player1"]
                                   + WALLS_PER_PLAYER - self.walls_left["player2"]),
        }
