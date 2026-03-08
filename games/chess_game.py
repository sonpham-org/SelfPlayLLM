"""Complete chess engine. Full standard rules including castling, en passant,
promotion, check/checkmate/stalemate, and draw conditions."""

import re
from typing import Any, Optional

from games.base import BaseGame

# ── Piece characters ──────────────────────────────────────────────────
# Upper = white, lower = black, '.' = empty
EMPTY = "."

# Material values
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

# Starting position (row 0 = rank 8, row 7 = rank 1)
INITIAL_BOARD = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
]

# Center squares (e4, d4, e5, d5) as (row, col)
CENTER_SQUARES = [(3, 3), (3, 4), (4, 3), (4, 4)]


# ── Helpers ───────────────────────────────────────────────────────────

def _is_white(piece: str) -> bool:
    return piece != EMPTY and piece.isupper()


def _is_black(piece: str) -> bool:
    return piece != EMPTY and piece.islower()


def _color_of(piece: str) -> Optional[str]:
    if piece == EMPTY:
        return None
    return "white" if piece.isupper() else "black"


def _is_friendly(piece: str, color: str) -> bool:
    if piece == EMPTY:
        return False
    return (color == "white" and piece.isupper()) or (color == "black" and piece.islower())


def _is_enemy(piece: str, color: str) -> bool:
    if piece == EMPTY:
        return False
    return not _is_friendly(piece, color)


def _opponent(color: str) -> str:
    return "black" if color == "white" else "white"


def _rc_to_alg(row: int, col: int) -> str:
    """Convert (row, col) to algebraic like 'e2'."""
    return chr(ord("a") + col) + str(8 - row)


def _alg_to_rc(alg: str) -> tuple[int, int]:
    """Convert algebraic like 'e2' to (row, col)."""
    col = ord(alg[0]) - ord("a")
    row = 8 - int(alg[1])
    return row, col


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def _board_hash(board: list[list[str]], turn: str,
                castling_rights: dict, ep_target: Optional[tuple[int, int]]) -> int:
    """Compute a hash for threefold-repetition detection."""
    parts = []
    for row in board:
        parts.append("".join(row))
    parts.append(turn)
    parts.append(str(castling_rights))
    parts.append(str(ep_target))
    return hash("".join(parts))


# ── Move representation ──────────────────────────────────────────────

class ChessMove:
    """Represents a chess move."""

    __slots__ = ("fr", "fc", "tr", "tc", "promotion")

    def __init__(self, fr: int, fc: int, tr: int, tc: int,
                 promotion: Optional[str] = None):
        self.fr = fr
        self.fc = fc
        self.tr = tr
        self.tc = tc
        self.promotion = promotion  # 'q','r','b','n' (lowercase)

    def to_algebraic(self) -> str:
        s = _rc_to_alg(self.fr, self.fc) + _rc_to_alg(self.tr, self.tc)
        if self.promotion:
            s += self.promotion.lower()
        return s

    def __eq__(self, other):
        if not isinstance(other, ChessMove):
            return False
        return (self.fr == other.fr and self.fc == other.fc
                and self.tr == other.tr and self.tc == other.tc
                and self.promotion == other.promotion)

    def __hash__(self):
        return hash((self.fr, self.fc, self.tr, self.tc, self.promotion))

    def __repr__(self):
        return f"ChessMove({self.to_algebraic()})"


# ── Main engine ──────────────────────────────────────────────────────

class ChessGame(BaseGame):
    """Full standard chess implementation.

    White goes first. Board is stored as an 8x8 list of single-character
    strings: upper-case = white, lower-case = black, '.' = empty.
    Row 0 is rank 8 (black's back rank), row 7 is rank 1 (white's back rank).
    """

    def __init__(self):
        self.board: list[list[str]] = [row[:] for row in INITIAL_BOARD]
        self.turn: str = "white"
        self._move_count: int = 0
        self._halfmove_clock: int = 0  # 50-move rule counter
        # Castling rights: True means that rook/king haven't moved
        self._castling: dict[str, bool] = {
            "K": True,  # white kingside
            "Q": True,  # white queenside
            "k": True,  # black kingside
            "q": True,  # black queenside
        }
        # En passant target square (row, col) or None
        self._ep_target: Optional[tuple[int, int]] = None
        # Position history for threefold repetition
        self._position_history: list[int] = []
        self._record_position()

    # ── BaseGame properties ──────────────────────────────────────────

    @property
    def players(self) -> tuple[str, str]:
        return ("white", "black")

    @property
    def current_player(self) -> str:
        return self.turn

    @property
    def move_count(self) -> int:
        return self._move_count

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> "ChessGame":
        g = ChessGame.__new__(ChessGame)
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g._move_count = self._move_count
        g._halfmove_clock = self._halfmove_clock
        g._castling = dict(self._castling)
        g._ep_target = self._ep_target
        g._position_history = list(self._position_history)
        return g

    # ── Position recording ───────────────────────────────────────────

    def _record_position(self):
        h = _board_hash(self.board, self.turn, self._castling, self._ep_target)
        self._position_history.append(h)

    def _is_threefold(self) -> bool:
        if len(self._position_history) < 3:
            return False
        current = self._position_history[-1]
        return self._position_history.count(current) >= 3

    # ── Square attack detection ──────────────────────────────────────

    def _is_attacked_by(self, row: int, col: int, attacker_color: str) -> bool:
        """Check if square (row, col) is attacked by any piece of attacker_color."""
        # Pawn attacks
        if attacker_color == "white":
            # White pawns attack diagonally upward (from white's perspective,
            # i.e. to lower row indices)
            for dc in (-1, 1):
                pr, pc = row + 1, col + dc
                if _in_bounds(pr, pc) and self.board[pr][pc] == "P":
                    return True
        else:
            for dc in (-1, 1):
                pr, pc = row - 1, col + dc
                if _in_bounds(pr, pc) and self.board[pr][pc] == "p":
                    return True

        # Knight attacks
        knight = "N" if attacker_color == "white" else "n"
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nc = row + dr, col + dc
            if _in_bounds(nr, nc) and self.board[nr][nc] == knight:
                return True

        # King attacks
        king = "K" if attacker_color == "white" else "k"
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if _in_bounds(nr, nc) and self.board[nr][nc] == king:
                    return True

        # Rook/Queen attacks (straight lines)
        rook = "R" if attacker_color == "white" else "r"
        queen = "Q" if attacker_color == "white" else "q"
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            while _in_bounds(nr, nc):
                piece = self.board[nr][nc]
                if piece != EMPTY:
                    if piece == rook or piece == queen:
                        return True
                    break  # blocked by another piece
                nr += dr
                nc += dc

        # Bishop/Queen attacks (diagonals)
        bishop = "B" if attacker_color == "white" else "b"
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            while _in_bounds(nr, nc):
                piece = self.board[nr][nc]
                if piece != EMPTY:
                    if piece == bishop or piece == queen:
                        return True
                    break
                nr += dr
                nc += dc

        return False

    def _find_king(self, color: str) -> tuple[int, int]:
        """Find the king's position for the given color."""
        king = "K" if color == "white" else "k"
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == king:
                    return r, c
        raise ValueError(f"No {color} king found on board")

    def _in_check(self, color: str) -> bool:
        """Check if the given color's king is in check."""
        kr, kc = self._find_king(color)
        return self._is_attacked_by(kr, kc, _opponent(color))

    # ── Pseudo-legal move generation ─────────────────────────────────

    def _pseudo_legal_moves(self, color: str) -> list[ChessMove]:
        """Generate all pseudo-legal moves (may leave own king in check)."""
        moves: list[ChessMove] = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if not _is_friendly(piece, color):
                    continue
                pt = piece.lower()
                if pt == "p":
                    self._pawn_moves(r, c, color, moves)
                elif pt == "n":
                    self._knight_moves(r, c, color, moves)
                elif pt == "b":
                    self._sliding_moves(r, c, color, moves,
                                        [(-1, -1), (-1, 1), (1, -1), (1, 1)])
                elif pt == "r":
                    self._sliding_moves(r, c, color, moves,
                                        [(-1, 0), (1, 0), (0, -1), (0, 1)])
                elif pt == "q":
                    self._sliding_moves(r, c, color, moves,
                                        [(-1, -1), (-1, 1), (1, -1), (1, 1),
                                         (-1, 0), (1, 0), (0, -1), (0, 1)])
                elif pt == "k":
                    self._king_moves(r, c, color, moves)
        return moves

    def _add_pawn_move(self, fr: int, fc: int, tr: int, tc: int,
                       color: str, moves: list[ChessMove]):
        """Add pawn move, expanding into promotions if reaching back rank."""
        promo_rank = 0 if color == "white" else 7
        if tr == promo_rank:
            for promo in ["q", "r", "b", "n"]:
                moves.append(ChessMove(fr, fc, tr, tc, promo))
        else:
            moves.append(ChessMove(fr, fc, tr, tc))

    def _pawn_moves(self, r: int, c: int, color: str,
                    moves: list[ChessMove]):
        direction = -1 if color == "white" else 1
        start_rank = 6 if color == "white" else 1

        # Single push
        nr = r + direction
        if _in_bounds(nr, c) and self.board[nr][c] == EMPTY:
            self._add_pawn_move(r, c, nr, c, color, moves)
            # Double push from starting rank
            nnr = nr + direction
            if r == start_rank and _in_bounds(nnr, c) and self.board[nnr][c] == EMPTY:
                moves.append(ChessMove(r, c, nnr, c))

        # Captures
        for dc in (-1, 1):
            nc = c + dc
            if not _in_bounds(nr, nc):
                continue
            target = self.board[nr][nc]
            if _is_enemy(target, color):
                self._add_pawn_move(r, c, nr, nc, color, moves)
            # En passant
            elif self._ep_target and (nr, nc) == self._ep_target:
                moves.append(ChessMove(r, c, nr, nc))

    def _knight_moves(self, r: int, c: int, color: str,
                      moves: list[ChessMove]):
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc) and not _is_friendly(self.board[nr][nc], color):
                moves.append(ChessMove(r, c, nr, nc))

    def _sliding_moves(self, r: int, c: int, color: str,
                       moves: list[ChessMove],
                       directions: list[tuple[int, int]]):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while _in_bounds(nr, nc):
                target = self.board[nr][nc]
                if target == EMPTY:
                    moves.append(ChessMove(r, c, nr, nc))
                elif _is_enemy(target, color):
                    moves.append(ChessMove(r, c, nr, nc))
                    break
                else:
                    break  # friendly piece blocks
                nr += dr
                nc += dc

    def _king_moves(self, r: int, c: int, color: str,
                    moves: list[ChessMove]):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc) and not _is_friendly(self.board[nr][nc], color):
                    moves.append(ChessMove(r, c, nr, nc))

        # Castling
        opp = _opponent(color)
        if color == "white":
            # Kingside: e1-g1
            if (self._castling["K"]
                    and r == 7 and c == 4
                    and self.board[7][5] == EMPTY
                    and self.board[7][6] == EMPTY
                    and not self._is_attacked_by(7, 4, opp)
                    and not self._is_attacked_by(7, 5, opp)
                    and not self._is_attacked_by(7, 6, opp)):
                moves.append(ChessMove(7, 4, 7, 6))
            # Queenside: e1-c1
            if (self._castling["Q"]
                    and r == 7 and c == 4
                    and self.board[7][3] == EMPTY
                    and self.board[7][2] == EMPTY
                    and self.board[7][1] == EMPTY
                    and not self._is_attacked_by(7, 4, opp)
                    and not self._is_attacked_by(7, 3, opp)
                    and not self._is_attacked_by(7, 2, opp)):
                moves.append(ChessMove(7, 4, 7, 2))
        else:
            # Kingside: e8-g8
            if (self._castling["k"]
                    and r == 0 and c == 4
                    and self.board[0][5] == EMPTY
                    and self.board[0][6] == EMPTY
                    and not self._is_attacked_by(0, 4, opp)
                    and not self._is_attacked_by(0, 5, opp)
                    and not self._is_attacked_by(0, 6, opp)):
                moves.append(ChessMove(0, 4, 0, 6))
            # Queenside: e8-c8
            if (self._castling["q"]
                    and r == 0 and c == 4
                    and self.board[0][3] == EMPTY
                    and self.board[0][2] == EMPTY
                    and self.board[0][1] == EMPTY
                    and not self._is_attacked_by(0, 4, opp)
                    and not self._is_attacked_by(0, 3, opp)
                    and not self._is_attacked_by(0, 2, opp)):
                moves.append(ChessMove(0, 4, 0, 2))

    # ── Legal move generation (filters out self-check) ───────────────

    def get_legal_moves(self, player: str | None = None) -> list[ChessMove]:
        if player is None:
            player = self.turn
        pseudo = self._pseudo_legal_moves(player)
        legal: list[ChessMove] = []
        for move in pseudo:
            if self._is_legal(move, player):
                legal.append(move)
        return legal

    def _is_legal(self, move: ChessMove, color: str) -> bool:
        """Check if a pseudo-legal move is truly legal (doesn't leave king in check)."""
        # Make the move on a temporary board
        saved_board = [row[:] for row in self.board]
        saved_ep = self._ep_target
        saved_castling = dict(self._castling)

        self._apply_move_raw(move, color)
        in_check = self._in_check(color)

        # Undo
        self.board = saved_board
        self._ep_target = saved_ep
        self._castling = saved_castling
        return not in_check

    def _apply_move_raw(self, move: ChessMove, color: str):
        """Apply a move to self.board without updating turn/counters.
        Used for legality checks."""
        piece = self.board[move.fr][move.fc]
        captured = self.board[move.tr][move.tc]
        pt = piece.lower()

        # Move piece
        self.board[move.fr][move.fc] = EMPTY
        self.board[move.tr][move.tc] = piece

        # En passant capture
        if pt == "p" and (move.tr, move.tc) == self._ep_target:
            # Remove the captured pawn
            cap_row = move.fr  # pawn being captured is on the same rank as the moving pawn
            self.board[cap_row][move.tc] = EMPTY

        # Castling: move the rook too
        if pt == "k" and abs(move.fc - move.tc) == 2:
            if move.tc == 6:  # kingside
                self.board[move.tr][5] = self.board[move.tr][7]
                self.board[move.tr][7] = EMPTY
            elif move.tc == 2:  # queenside
                self.board[move.tr][3] = self.board[move.tr][0]
                self.board[move.tr][0] = EMPTY

        # Promotion
        if move.promotion:
            promo_piece = move.promotion if color == "black" else move.promotion.upper()
            self.board[move.tr][move.tc] = promo_piece

    # ── Apply move (full, with state updates) ────────────────────────

    def apply_move(self, move: Any) -> None:
        if not isinstance(move, ChessMove):
            raise TypeError(f"Expected ChessMove, got {type(move)}")

        piece = self.board[move.fr][move.fc]
        pt = piece.lower()
        captured = self.board[move.tr][move.tc]
        color = self.turn

        # Determine if this is a pawn move or capture (for 50-move rule)
        is_pawn_move = (pt == "p")
        is_capture = (captured != EMPTY)
        # En passant is also a capture
        if pt == "p" and self._ep_target and (move.tr, move.tc) == self._ep_target:
            is_capture = True

        # Apply on board
        self.board[move.fr][move.fc] = EMPTY
        self.board[move.tr][move.tc] = piece

        # En passant capture
        if pt == "p" and self._ep_target and (move.tr, move.tc) == self._ep_target:
            self.board[move.fr][move.tc] = EMPTY

        # Castling: move rook
        if pt == "k" and abs(move.fc - move.tc) == 2:
            if move.tc == 6:  # kingside
                self.board[move.tr][5] = self.board[move.tr][7]
                self.board[move.tr][7] = EMPTY
            elif move.tc == 2:  # queenside
                self.board[move.tr][3] = self.board[move.tr][0]
                self.board[move.tr][0] = EMPTY

        # Promotion
        if move.promotion:
            promo_piece = move.promotion if color == "black" else move.promotion.upper()
            self.board[move.tr][move.tc] = promo_piece

        # Update en passant target
        if pt == "p" and abs(move.tr - move.fr) == 2:
            ep_row = (move.fr + move.tr) // 2
            self._ep_target = (ep_row, move.fc)
        else:
            self._ep_target = None

        # Update castling rights
        # King moved
        if pt == "k":
            if color == "white":
                self._castling["K"] = False
                self._castling["Q"] = False
            else:
                self._castling["k"] = False
                self._castling["q"] = False
        # Rook moved or captured
        if move.fr == 7 and move.fc == 0:
            self._castling["Q"] = False
        if move.fr == 7 and move.fc == 7:
            self._castling["K"] = False
        if move.fr == 0 and move.fc == 0:
            self._castling["q"] = False
        if move.fr == 0 and move.fc == 7:
            self._castling["k"] = False
        # Rook captured on its home square
        if move.tr == 7 and move.tc == 0:
            self._castling["Q"] = False
        if move.tr == 7 and move.tc == 7:
            self._castling["K"] = False
        if move.tr == 0 and move.tc == 0:
            self._castling["q"] = False
        if move.tr == 0 and move.tc == 7:
            self._castling["k"] = False

        # Update counters
        if is_pawn_move or is_capture:
            self._halfmove_clock = 0
        else:
            self._halfmove_clock += 1
        self._move_count += 1
        self.turn = _opponent(color)

        # Record position for threefold repetition
        self._record_position()

    # ── Winner / end detection ───────────────────────────────────────

    def winner(self) -> Optional[str]:
        """Return 'white', 'black', 'draw', or None if game is ongoing."""
        # Safety cap
        if self._move_count >= 300:
            return "draw"

        # 50-move rule
        if self._halfmove_clock >= 100:  # 100 half-moves = 50 full moves
            return "draw"

        # Threefold repetition
        if self._is_threefold():
            return "draw"

        # Insufficient material
        if self._insufficient_material():
            return "draw"

        # Check for checkmate or stalemate
        current = self.turn
        legal = self.get_legal_moves(current)
        if not legal:
            if self._in_check(current):
                # Checkmate: the other player wins
                return _opponent(current)
            else:
                # Stalemate
                return "draw"

        return None

    def _insufficient_material(self) -> bool:
        """Check for insufficient material draws:
        K vs K, K+B vs K, K+N vs K."""
        white_pieces: list[str] = []
        black_pieces: list[str] = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p == EMPTY:
                    continue
                if _is_white(p):
                    white_pieces.append(p.lower())
                else:
                    black_pieces.append(p.lower())

        white_pieces.sort()
        black_pieces.sort()

        # K vs K
        if white_pieces == ["k"] and black_pieces == ["k"]:
            return True
        # K+B vs K
        if white_pieces == ["b", "k"] and black_pieces == ["k"]:
            return True
        if white_pieces == ["k"] and black_pieces == ["b", "k"]:
            return True
        # K+N vs K
        if white_pieces == ["k", "n"] and black_pieces == ["k"]:
            return True
        if white_pieces == ["k"] and black_pieces == ["k", "n"]:
            return True

        return False

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, perspective: str | None = None) -> str:
        lines = ["  a b c d e f g h"]
        for r in range(8):
            rank = 8 - r
            row_str = f"{rank} " + " ".join(self.board[r][c] for c in range(8))
            lines.append(row_str)
        lines.append("  a b c d e f g h")
        return "\n".join(lines)

    # ── Move string conversion ───────────────────────────────────────

    def move_to_str(self, move: Any) -> str:
        if isinstance(move, ChessMove):
            return move.to_algebraic()
        return str(move)

    def parse_move(self, text: str, player: str) -> Optional[ChessMove]:
        """Parse a move string from LLM output and match to a legal move.

        Accepted formats:
          - Coordinate notation: "e2e4", "e7e8q" (promotion), "e1g1" (castling)
          - Spaced: "e2 e4", "e2-e4", "e2 -> e4"
          - Coordinate tuple: "(6,4) -> (4,4)" (row,col)
        """
        text = text.strip()
        legal = self.get_legal_moves(player)
        if not legal:
            return None

        # Try coordinate-tuple format: (r1,c1) -> (r2,c2)
        coord_match = re.search(
            r'\((\d)\s*,\s*(\d)\)\s*(?:->|to)?\s*\((\d)\s*,\s*(\d)\)', text)
        if coord_match:
            fr = int(coord_match.group(1))
            fc = int(coord_match.group(2))
            tr = int(coord_match.group(3))
            tc = int(coord_match.group(4))
            # Check for promotion suffix after the coordinates
            promo = None
            remainder = text[coord_match.end():].strip().lower()
            if remainder and remainder[0] in "qrbn":
                promo = remainder[0]
            return self._match_move(fr, fc, tr, tc, promo, legal)

        # Try algebraic coordinate notation: e2e4, e2-e4, e2 e4, e2->e4, e7e8q
        alg_match = re.search(
            r'([a-h])([1-8])\s*[->\s]*([a-h])([1-8])\s*([qrbnQRBN])?', text)
        if alg_match:
            fr, fc = _alg_to_rc(alg_match.group(1) + alg_match.group(2))
            tr, tc = _alg_to_rc(alg_match.group(3) + alg_match.group(4))
            promo = alg_match.group(5)
            if promo:
                promo = promo.lower()
            return self._match_move(fr, fc, tr, tc, promo, legal)

        return None

    def _match_move(self, fr: int, fc: int, tr: int, tc: int,
                    promo: Optional[str],
                    legal: list[ChessMove]) -> Optional[ChessMove]:
        """Match parsed coordinates against legal moves."""
        # If promotion not specified but move requires it, default to queen
        candidates = [m for m in legal
                      if m.fr == fr and m.fc == fc and m.tr == tr and m.tc == tc]
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates means promotion options
        if promo:
            for m in candidates:
                if m.promotion == promo:
                    return m
        # Default to queen promotion
        for m in candidates:
            if m.promotion == "q":
                return m
        return candidates[0]

    # ── State for strategies ─────────────────────────────────────────

    def get_state(self, player: str) -> dict:
        opp = _opponent(player)

        board_copy = [row[:] for row in self.board]

        my_pieces = self._count_pieces(player)
        opp_pieces = self._count_pieces(opp)

        my_material = self._material_value(player)
        opp_material = self._material_value(opp)

        in_check = self._in_check(player)

        can_castle_ks, can_castle_qs = self._can_castle(player)

        center = self._center_control(player)

        king_safety = self._king_safety(player)

        pawn_struct = self._pawn_structure(player)

        return {
            "board": board_copy,
            "my_color": player,
            "move_number": self._move_count,
            "my_pieces": my_pieces,
            "opponent_pieces": opp_pieces,
            "material_balance": my_material - opp_material,
            "in_check": in_check,
            "can_castle_kingside": can_castle_ks,
            "can_castle_queenside": can_castle_qs,
            "center_control": center,
            "king_safety": king_safety,
            "pawn_structure": pawn_struct,
        }

    def _count_pieces(self, color: str) -> dict:
        counts = {"pawns": 0, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0}
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if not _is_friendly(p, color):
                    continue
                pt = p.lower()
                if pt == "p":
                    counts["pawns"] += 1
                elif pt == "n":
                    counts["knights"] += 1
                elif pt == "b":
                    counts["bishops"] += 1
                elif pt == "r":
                    counts["rooks"] += 1
                elif pt == "q":
                    counts["queens"] += 1
        return counts

    def _material_value(self, color: str) -> int:
        total = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if _is_friendly(p, color):
                    total += PIECE_VALUES.get(p.lower(), 0)
        return total

    def _can_castle(self, color: str) -> tuple[bool, bool]:
        """Return (can_castle_kingside, can_castle_queenside) based on
        whether the castling move is currently legal (not just rights)."""
        legal = self.get_legal_moves(color)
        if color == "white":
            ks = any(m.fr == 7 and m.fc == 4 and m.tr == 7 and m.tc == 6 for m in legal)
            qs = any(m.fr == 7 and m.fc == 4 and m.tr == 7 and m.tc == 2 for m in legal)
        else:
            ks = any(m.fr == 0 and m.fc == 4 and m.tr == 0 and m.tc == 6 for m in legal)
            qs = any(m.fr == 0 and m.fc == 4 and m.tr == 0 and m.tc == 2 for m in legal)
        return ks, qs

    def _center_control(self, color: str) -> int:
        """Count pieces/pawns attacking or occupying the four center squares."""
        count = 0
        for r, c in CENTER_SQUARES:
            # Occupying the center
            if _is_friendly(self.board[r][c], color):
                count += 1
            # Attacking the center
            if self._is_attacked_by(r, c, color):
                count += 1
        return count

    def _king_safety(self, color: str) -> int:
        """Simple heuristic: count friendly pawns within 2 squares of king."""
        kr, kc = self._find_king(color)
        pawn = "P" if color == "white" else "p"
        count = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = kr + dr, kc + dc
                if _in_bounds(nr, nc) and self.board[nr][nc] == pawn:
                    count += 1
        return count

    def _pawn_structure(self, color: str) -> dict:
        """Analyze pawn structure: isolated, doubled, passed pawns."""
        pawn = "P" if color == "white" else "p"
        opp_pawn = "p" if color == "white" else "P"
        direction = -1 if color == "white" else 1  # forward direction

        # Collect pawn positions by file
        my_pawns_by_file: dict[int, list[int]] = {}
        opp_pawns_by_file: dict[int, list[int]] = {}
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == pawn:
                    my_pawns_by_file.setdefault(c, []).append(r)
                elif self.board[r][c] == opp_pawn:
                    opp_pawns_by_file.setdefault(c, []).append(r)

        isolated = 0
        doubled = 0
        passed = 0

        for file, rows in my_pawns_by_file.items():
            # Doubled: more than one pawn on the same file
            if len(rows) > 1:
                doubled += len(rows) - 1

            # Isolated: no friendly pawns on adjacent files
            has_neighbor = False
            for adj in (file - 1, file + 1):
                if adj in my_pawns_by_file:
                    has_neighbor = True
                    break
            if not has_neighbor:
                isolated += len(rows)

            # Passed: no opponent pawns ahead on same or adjacent files
            for pawn_row in rows:
                is_passed = True
                for check_file in (file - 1, file, file + 1):
                    if check_file not in opp_pawns_by_file:
                        continue
                    for opp_row in opp_pawns_by_file[check_file]:
                        if color == "white":
                            # White pawns move up (lower row numbers)
                            if opp_row < pawn_row:
                                is_passed = False
                                break
                        else:
                            # Black pawns move down (higher row numbers)
                            if opp_row > pawn_row:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    passed += 1

        return {"isolated": isolated, "doubled": doubled, "passed": passed}
