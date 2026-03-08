"""Stratego game engine. 10x10 board, hidden information, red vs blue."""

import re
import random
from dataclasses import dataclass
from typing import Any, Optional

from games.base import BaseGame

BOARD_SIZE = 10

# Water squares (impassable): two 2x2 blocks in the center
WATER_SQUARES: set[tuple[int, int]] = set()
for r in (4, 5):
    for c in (2, 3):
        WATER_SQUARES.add((r, c))
    for c in (6, 7):
        WATER_SQUARES.add((r, c))

# Piece rank definitions
# Rank value -> (name, count per player)
PIECE_DEFS: dict[str, tuple[str, int]] = {
    "10": ("Marshal", 1),
    "9":  ("General", 1),
    "8":  ("Colonel", 2),
    "7":  ("Major", 3),
    "6":  ("Captain", 4),
    "5":  ("Lieutenant", 4),
    "4":  ("Sergeant", 4),
    "3":  ("Miner", 5),
    "2":  ("Scout", 8),
    "1":  ("Spy", 1),
    "B":  ("Bomb", 6),
    "F":  ("Flag", 1),
}

RANK_NAMES: dict[str, str] = {k: v[0] for k, v in PIECE_DEFS.items()}

# Numeric rank for combat comparison (higher wins). B and F are special.
COMBAT_RANKS: dict[str, int] = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}

# Immovable pieces
IMMOVABLE_RANKS = {"B", "F"}

# All ranks in display order
ALL_RANKS = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "B", "F"]


@dataclass
class Piece:
    """A single Stratego piece."""
    rank: str       # "1"-"10", "B", "F"
    color: str      # "red" or "blue"
    revealed: bool = False  # Has this piece's rank been seen by the opponent?

    @property
    def name(self) -> str:
        return RANK_NAMES[self.rank]

    @property
    def is_movable(self) -> bool:
        return self.rank not in IMMOVABLE_RANKS

    def display(self, viewer: Optional[str] = None) -> str:
        """Return display string for this piece.

        If viewer is the same color, show rank. If opponent, show '?' unless
        revealed. If viewer is None, show everything (god mode).
        """
        if viewer is None or viewer == self.color:
            prefix = "r" if self.color == "red" else "b"
            return f"{prefix}{self.rank}"
        else:
            if self.revealed:
                prefix = "r" if self.color == "red" else "b"
                return f"{prefix}{self.rank}"
            else:
                return "r?" if self.color == "red" else "b?"


class StrategoGame(BaseGame):
    """Complete Stratego implementation with hidden information.

    10x10 board. Red occupies rows 6-9 (bottom), Blue occupies rows 0-3 (top).
    Red moves first.
    """

    def __init__(self, seed: Optional[int] = None):
        self.board: list[list[Optional[Piece]]] = [
            [None] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        self.turn: str = "red"
        self._move_count: int = 0
        self._max_moves: int = 400
        self._attack_history: list[str] = []
        self._seed = seed
        self._setup_board(seed)

    # ------------------------------------------------------------------
    # BaseGame interface
    # ------------------------------------------------------------------

    @property
    def players(self) -> tuple[str, str]:
        return ("red", "blue")

    @property
    def current_player(self) -> str:
        return self.turn

    @property
    def move_count(self) -> int:
        return self._move_count

    def copy(self) -> "StrategoGame":
        """Return a deep copy of the game state."""
        g = StrategoGame.__new__(StrategoGame)
        g.board = [
            [
                Piece(rank=p.rank, color=p.color, revealed=p.revealed) if p else None
                for p in row
            ]
            for row in self.board
        ]
        g.turn = self.turn
        g._move_count = self._move_count
        g._max_moves = self._max_moves
        g._attack_history = list(self._attack_history)
        g._seed = self._seed
        return g

    def get_legal_moves(self, player: Optional[str] = None) -> list[str]:
        """Return list of legal move strings for the given player."""
        if player is None:
            player = self.turn
        moves: list[str] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]
                if piece is None or piece.color != player or not piece.is_movable:
                    continue
                if piece.rank == "2":
                    # Scout: can move any number of squares in a straight line
                    moves.extend(self._get_scout_moves(r, c, player))
                else:
                    # Normal piece: one square orthogonally
                    moves.extend(self._get_normal_moves(r, c, player))
        return moves

    def apply_move(self, move: Any) -> None:
        """Apply a move string like '(r1,c1) -> (r2,c2)'."""
        if isinstance(move, str):
            parsed = self._parse_coords(move)
            if parsed is None or len(parsed) < 2:
                raise ValueError(f"Invalid move format: {move}")
            (r1, c1), (r2, c2) = parsed[0], parsed[-1]
        else:
            raise ValueError(f"Move must be a string, got {type(move)}")

        piece = self.board[r1][c1]
        if piece is None:
            raise ValueError(f"No piece at ({r1},{c1})")

        target = self.board[r2][c2]

        if target is None:
            # Simple move
            self.board[r2][c2] = piece
            self.board[r1][c1] = None
        else:
            # Attack
            self._resolve_combat(r1, c1, r2, c2)

        self._move_count += 1
        self.turn = "blue" if self.turn == "red" else "red"

    def winner(self) -> Optional[str]:
        """Return 'red', 'blue', 'draw', or None if game is ongoing."""
        if self._move_count >= self._max_moves:
            return "draw"

        red_flag = False
        blue_flag = False
        red_movable = False
        blue_movable = False

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]
                if piece is None:
                    continue
                if piece.color == "red":
                    if piece.rank == "F":
                        red_flag = True
                    if piece.is_movable:
                        red_movable = True
                else:
                    if piece.rank == "F":
                        blue_flag = True
                    if piece.is_movable:
                        blue_movable = True

        # Flag captured
        if not red_flag:
            return "blue"
        if not blue_flag:
            return "red"

        # Current player loses if they have no legal moves (no movable pieces
        # or all movable pieces are completely blocked)
        if not self.get_legal_moves(self.turn):
            return "blue" if self.turn == "red" else "red"

        return None

    def render(self, perspective: Optional[str] = None) -> str:
        """Return ASCII representation of the board.

        perspective: 'red', 'blue', or None (god mode shows all).
        """
        lines = ["   " + " ".join(f"{c:>3}" for c in range(BOARD_SIZE))]
        for r in range(BOARD_SIZE):
            row_cells = []
            for c in range(BOARD_SIZE):
                if (r, c) in WATER_SQUARES:
                    row_cells.append("~~~")
                elif self.board[r][c] is None:
                    row_cells.append("  .")
                else:
                    row_cells.append(
                        f"{self.board[r][c].display(perspective):>3}"
                    )
            lines.append(f"{r:>2} " + " ".join(row_cells))
        return "\n".join(lines)

    def get_state(self, player: str) -> dict:
        """Return game state from the perspective of the given player."""
        board_view: list[list[str]] = []
        my_pieces_remaining = 0
        opp_pieces_remaining = 0
        my_pieces_by_rank: dict[str, int] = {}
        known_opponent: dict[str, int] = {}
        unknown_opponent = 0
        my_movable = 0

        for r in range(BOARD_SIZE):
            row: list[str] = []
            for c in range(BOARD_SIZE):
                if (r, c) in WATER_SQUARES:
                    row.append("~~~")
                elif self.board[r][c] is None:
                    row.append(".")
                else:
                    piece = self.board[r][c]
                    if piece.color == player:
                        my_pieces_remaining += 1
                        my_pieces_by_rank[piece.rank] = (
                            my_pieces_by_rank.get(piece.rank, 0) + 1
                        )
                        if piece.is_movable:
                            my_movable += 1
                        row.append(piece.display(player))
                    else:
                        opp_pieces_remaining += 1
                        if piece.revealed:
                            known_opponent[piece.rank] = (
                                known_opponent.get(piece.rank, 0) + 1
                            )
                            row.append(piece.display(player))
                        else:
                            unknown_opponent += 1
                            row.append(piece.display(player))
            board_view.append(row)

        return {
            "board": board_view,
            "my_color": player,
            "move_number": self._move_count,
            "my_pieces_remaining": my_pieces_remaining,
            "opponent_pieces_remaining": opp_pieces_remaining,
            "my_pieces_by_rank": my_pieces_by_rank,
            "known_opponent_pieces": known_opponent,
            "unknown_opponent_pieces": unknown_opponent,
            "my_movable_pieces": my_movable,
            "attack_history": list(self._attack_history),
        }

    def move_to_str(self, move: Any) -> str:
        """Convert a move to its string representation (identity for strings)."""
        return str(move)

    def parse_move(self, text: str, player: str) -> Optional[str]:
        """Parse a move string from LLM output and match to a legal move.

        Accepts formats like '(r1,c1) -> (r2,c2)' or 'r1,c1 -> r2,c2'.
        """
        coords = self._parse_coords(text)
        if coords is None or len(coords) < 2:
            return None

        (r1, c1), (r2, c2) = coords[0], coords[-1]
        candidate = f"({r1},{c1}) -> ({r2},{c2})"

        legal = self.get_legal_moves(player)
        if candidate in legal:
            return candidate
        return None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_board(self, seed: Optional[int] = None) -> None:
        """Randomly place pieces for both players."""
        rng = random.Random(seed)
        self._place_army("red", rng)
        self._place_army("blue", rng)

    def _place_army(self, color: str, rng: random.Random) -> None:
        """Place all pieces for one player randomly in their starting rows."""
        # Red: rows 6-9, Blue: rows 0-3
        if color == "red":
            rows = [6, 7, 8, 9]
            back_row = 9
        else:
            rows = [0, 1, 2, 3]
            back_row = 0

        # Collect all available squares (exclude water)
        available: list[tuple[int, int]] = []
        back_row_squares: list[tuple[int, int]] = []
        for r in rows:
            for c in range(BOARD_SIZE):
                if (r, c) not in WATER_SQUARES:
                    available.append((r, c))
                    if r == back_row:
                        back_row_squares.append((r, c))

        # Build piece list
        pieces: list[str] = []
        for rank, (name, count) in PIECE_DEFS.items():
            pieces.extend([rank] * count)

        # Place Flag first in back row
        flag_pos = rng.choice(back_row_squares)
        self.board[flag_pos[0]][flag_pos[1]] = Piece(rank="F", color=color)
        available.remove(flag_pos)
        back_row_squares.remove(flag_pos)
        pieces.remove("F")

        # Place Bombs adjacent to flag (orthogonally adjacent squares that are
        # in the player's territory)
        bomb_count = PIECE_DEFS["B"][1]  # 6 bombs
        adj_squares = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = flag_pos[0] + dr, flag_pos[1] + dc
            if (nr, nc) in available:
                adj_squares.append((nr, nc))

        # Place as many bombs adjacent to flag as possible
        bombs_placed = 0
        rng.shuffle(adj_squares)
        for sq in adj_squares:
            if bombs_placed >= bomb_count:
                break
            self.board[sq[0]][sq[1]] = Piece(rank="B", color=color)
            available.remove(sq)
            if sq in back_row_squares:
                back_row_squares.remove(sq)
            bombs_placed += 1
            pieces.remove("B")

        # Place remaining bombs randomly in available squares
        remaining_bombs = pieces.count("B")
        if remaining_bombs > 0:
            bomb_positions = rng.sample(available, remaining_bombs)
            for pos in bomb_positions:
                self.board[pos[0]][pos[1]] = Piece(rank="B", color=color)
                available.remove(pos)
                if pos in back_row_squares:
                    back_row_squares.remove(pos)
            for _ in range(remaining_bombs):
                pieces.remove("B")

        # Place remaining pieces randomly
        rng.shuffle(pieces)
        positions = rng.sample(available, len(pieces))
        for piece_rank, pos in zip(pieces, positions):
            self.board[pos[0]][pos[1]] = Piece(rank=piece_rank, color=color)

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def _get_normal_moves(
        self, r: int, c: int, player: str
    ) -> list[str]:
        """Get single-step orthogonal moves for a non-Scout piece."""
        moves: list[str] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc):
                continue
            if (nr, nc) in WATER_SQUARES:
                continue
            target = self.board[nr][nc]
            if target is None:
                moves.append(f"({r},{c}) -> ({nr},{nc})")
            elif target.color != player:
                moves.append(f"({r},{c}) -> ({nr},{nc})")
        return moves

    def _get_scout_moves(
        self, r: int, c: int, player: str
    ) -> list[str]:
        """Get Scout moves: any number of squares in a straight line, no jumping."""
        moves: list[str] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            while self._in_bounds(nr, nc) and (nr, nc) not in WATER_SQUARES:
                target = self.board[nr][nc]
                if target is None:
                    moves.append(f"({r},{c}) -> ({nr},{nc})")
                    nr += dr
                    nc += dc
                elif target.color != player:
                    # Can attack but cannot go further
                    moves.append(f"({r},{c}) -> ({nr},{nc})")
                    break
                else:
                    # Own piece blocks
                    break
        return moves

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _resolve_combat(
        self, r1: int, c1: int, r2: int, c2: int
    ) -> None:
        """Resolve an attack from (r1,c1) onto (r2,c2)."""
        attacker = self.board[r1][c1]
        defender = self.board[r2][c2]
        assert attacker is not None and defender is not None

        atk_rank = attacker.rank
        def_rank = defender.rank

        # Both pieces are now revealed to each other's side
        attacker.revealed = True
        defender.revealed = True

        result: str

        # Special case: attacking a Bomb
        if def_rank == "B":
            if atk_rank == "3":
                # Miner defuses bomb
                result = "win"
                log = (
                    f"Move {self._move_count + 1}: "
                    f"{attacker.color} Miner(3) defused {defender.color} Bomb "
                    f"at ({r2},{c2})"
                )
            else:
                # Everything else loses to bomb
                result = "lose"
                log = (
                    f"Move {self._move_count + 1}: "
                    f"{attacker.color} {attacker.name}({atk_rank}) destroyed by "
                    f"{defender.color} Bomb at ({r2},{c2})"
                )
        # Special case: Spy attacks Marshal
        elif atk_rank == "1" and def_rank == "10":
            result = "win"
            log = (
                f"Move {self._move_count + 1}: "
                f"{attacker.color} Spy(1) captured {defender.color} Marshal(10) "
                f"at ({r2},{c2})"
            )
        # Special case: attacking a Flag
        elif def_rank == "F":
            result = "win"
            log = (
                f"Move {self._move_count + 1}: "
                f"{attacker.color} {attacker.name}({atk_rank}) captured "
                f"{defender.color} Flag at ({r2},{c2})!"
            )
        # Normal combat: compare numeric ranks
        else:
            atk_val = COMBAT_RANKS.get(atk_rank, 0)
            def_val = COMBAT_RANKS.get(def_rank, 0)
            if atk_val > def_val:
                result = "win"
                log = (
                    f"Move {self._move_count + 1}: "
                    f"{attacker.color} {attacker.name}({atk_rank}) defeated "
                    f"{defender.color} {defender.name}({def_rank}) at ({r2},{c2})"
                )
            elif atk_val < def_val:
                result = "lose"
                log = (
                    f"Move {self._move_count + 1}: "
                    f"{attacker.color} {attacker.name}({atk_rank}) lost to "
                    f"{defender.color} {defender.name}({def_rank}) at ({r2},{c2})"
                )
            else:
                result = "tie"
                log = (
                    f"Move {self._move_count + 1}: "
                    f"{attacker.color} {attacker.name}({atk_rank}) and "
                    f"{defender.color} {defender.name}({def_rank}) both destroyed "
                    f"at ({r2},{c2})"
                )

        self._attack_history.append(log)

        # Apply result
        if result == "win":
            self.board[r1][c1] = None
            self.board[r2][c2] = attacker
        elif result == "lose":
            self.board[r1][c1] = None
            # Defender stays
        else:
            # Tie: both removed
            self.board[r1][c1] = None
            self.board[r2][c2] = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    @staticmethod
    def _parse_coords(text: str) -> Optional[list[tuple[int, int]]]:
        """Extract coordinate pairs from a move string."""
        coords = re.findall(r'\((\d+)\s*,\s*(\d+)\)', text)
        if len(coords) < 2:
            return None
        return [(int(r), int(c)) for r, c in coords]
