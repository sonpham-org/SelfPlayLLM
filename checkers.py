"""Minimal checkers engine. 8x8 board, mandatory captures, kings, multi-jumps."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

class Piece(IntEnum):
    EMPTY = 0
    RED = 1       # moves "up" (row index decreasing)
    RED_KING = 2
    BLACK = 3     # moves "down" (row index increasing)
    BLACK_KING = 4

def is_red(p): return p in (Piece.RED, Piece.RED_KING)
def is_black(p): return p in (Piece.BLACK, Piece.BLACK_KING)
def is_king(p): return p in (Piece.RED_KING, Piece.BLACK_KING)
def same_side(p, color): return (is_red(p) and color == "red") or (is_black(p) and color == "black")
def opponent_side(p, color): return (is_red(p) and color == "black") or (is_black(p) and color == "red")

BOARD_SIZE = 8

@dataclass
class Move:
    path: list[tuple[int, int]]  # sequence of (row, col) positions
    captures: list[tuple[int, int]] = field(default_factory=list)

    def __str__(self):
        return " -> ".join(f"({r},{c})" for r, c in self.path)

class CheckersGame:
    def __init__(self):
        self.board = [[Piece.EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn = "red"  # red goes first
        self.move_count = 0
        self.no_capture_count = 0
        self._setup_board()

    def _setup_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    if r < 3:
                        self.board[r][c] = Piece.BLACK
                    elif r > 4:
                        self.board[r][c] = Piece.RED

    def copy(self):
        g = CheckersGame.__new__(CheckersGame)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g.move_count = self.move_count
        g.no_capture_count = self.no_capture_count
        return g

    def get_piece(self, r, c) -> Piece:
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return self.board[r][c]
        return Piece.EMPTY

    def _get_simple_moves(self, r, c) -> list[Move]:
        p = self.board[r][c]
        if p == Piece.EMPTY:
            return []
        dirs = []
        if is_red(p) or is_king(p):
            dirs += [(-1, -1), (-1, 1)]
        if is_black(p) or is_king(p):
            dirs += [(1, -1), (1, 1)]
        moves = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] == Piece.EMPTY:
                moves.append(Move(path=[(r, c), (nr, nc)]))
        return moves

    def _get_jumps(self, r, c, board=None) -> list[Move]:
        if board is None:
            board = self.board
        p = board[r][c]
        if p == Piece.EMPTY:
            return []
        color = "red" if is_red(p) else "black"
        dirs = []
        if is_red(p) or is_king(p):
            dirs += [(-1, -1), (-1, 1)]
        if is_black(p) or is_king(p):
            dirs += [(1, -1), (1, 1)]

        jumps = []
        for dr, dc in dirs:
            mr, mc = r + dr, c + dc  # middle (captured) square
            lr, lc = r + 2*dr, c + 2*dc  # landing square
            if (0 <= lr < BOARD_SIZE and 0 <= lc < BOARD_SIZE
                    and board[mr][mc] != Piece.EMPTY
                    and opponent_side(board[mr][mc], color)
                    and board[lr][lc] == Piece.EMPTY):
                # Try multi-jump from landing
                new_board = [row[:] for row in board]
                new_board[r][c] = Piece.EMPTY
                new_board[mr][mc] = Piece.EMPTY
                new_board[lr][lc] = p
                # Promote if reaching back row mid-chain
                if (is_red(p) and not is_king(p) and lr == 0) or \
                   (is_black(p) and not is_king(p) and lr == BOARD_SIZE - 1):
                    new_board[lr][lc] = Piece.RED_KING if is_red(p) else Piece.BLACK_KING

                further = self._get_jumps(lr, lc, new_board)
                if further:
                    for fj in further:
                        combined = Move(
                            path=[(r, c)] + fj.path,
                            captures=[(mr, mc)] + fj.captures,
                        )
                        jumps.append(combined)
                else:
                    jumps.append(Move(path=[(r, c), (lr, lc)], captures=[(mr, mc)]))
        return jumps

    def get_legal_moves(self, color: Optional[str] = None) -> list[Move]:
        if color is None:
            color = self.turn
        all_jumps = []
        all_simple = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if same_side(self.board[r][c], color):
                    all_jumps.extend(self._get_jumps(r, c))
                    all_simple.extend(self._get_simple_moves(r, c))
        # Mandatory capture rule
        if all_jumps:
            return all_jumps
        return all_simple

    def apply_move(self, move: Move):
        r0, c0 = move.path[0]
        rf, cf = move.path[-1]
        p = self.board[r0][c0]
        self.board[r0][c0] = Piece.EMPTY
        for cr, cc in move.captures:
            self.board[cr][cc] = Piece.EMPTY
        # King promotion
        if is_red(p) and rf == 0:
            p = Piece.RED_KING
        elif is_black(p) and rf == BOARD_SIZE - 1:
            p = Piece.BLACK_KING
        self.board[rf][cf] = p
        # Update counters
        if move.captures:
            self.no_capture_count = 0
        else:
            self.no_capture_count += 1
        self.move_count += 1
        self.turn = "black" if self.turn == "red" else "red"

    def winner(self) -> Optional[str]:
        """Returns 'red', 'black', 'draw', or None if game ongoing."""
        if self.no_capture_count >= 80:
            return "draw"
        if self.move_count >= 200:
            return "draw"
        red_moves = self.get_legal_moves("red")
        black_moves = self.get_legal_moves("black")
        red_pieces = any(is_red(self.board[r][c]) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        black_pieces = any(is_black(self.board[r][c]) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        if not red_pieces or not red_moves:
            return "black"
        if not black_pieces or not black_moves:
            return "red"
        return None

    def render(self, perspective: str = "red") -> str:
        symbols = {
            Piece.EMPTY: ".", Piece.RED: "r", Piece.RED_KING: "R",
            Piece.BLACK: "b", Piece.BLACK_KING: "B",
        }
        lines = ["  0 1 2 3 4 5 6 7"]
        for r in range(BOARD_SIZE):
            row_str = f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(BOARD_SIZE))
            lines.append(row_str)
        return "\n".join(lines)

    def move_to_str(self, move: Move) -> str:
        return " -> ".join(f"({r},{c})" for r, c in move.path)

    def parse_move(self, text: str, color: str) -> Optional[Move]:
        """Parse a move string like '(5,2) -> (4,3)' and match to legal moves."""
        import re
        coords = re.findall(r'\((\d+)\s*,\s*(\d+)\)', text)
        if len(coords) < 2:
            return None
        path = [(int(r), int(c)) for r, c in coords]
        legal = self.get_legal_moves(color)
        for m in legal:
            if m.path == path:
                return m
        return None
