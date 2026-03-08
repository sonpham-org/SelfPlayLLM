"""Checkers adapter — wraps the original CheckersGame to implement BaseGame."""

import sys
import os

# Add project root to path so we can import the original checkers module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkers import CheckersGame as _OriginalCheckers, Piece, is_red, is_black, is_king, BOARD_SIZE
from games.base import BaseGame


class CheckersGame(BaseGame):
    """Checkers game implementing the BaseGame interface."""

    def __init__(self):
        self._game = _OriginalCheckers()

    @property
    def players(self) -> tuple[str, str]:
        return ("red", "black")

    @property
    def current_player(self) -> str:
        return self._game.turn

    @property
    def move_count(self) -> int:
        return self._game.move_count

    def copy(self) -> "CheckersGame":
        g = CheckersGame.__new__(CheckersGame)
        g._game = self._game.copy()
        return g

    def get_legal_moves(self, player=None):
        return self._game.get_legal_moves(player)

    def apply_move(self, move):
        self._game.apply_move(move)

    def winner(self):
        return self._game.winner()

    def render(self, perspective=None):
        return self._game.render()

    def move_to_str(self, move):
        return self._game.move_to_str(move)

    def parse_move(self, text, player):
        return self._game.parse_move(text, player)

    def get_state(self, player: str) -> dict:
        board = self._game.board
        my_pieces = 0
        my_kings = 0
        opp_pieces = 0
        opp_kings = 0
        center_control = 0
        my_edge = 0
        opp_edge = 0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = board[r][c]
                if p == Piece.EMPTY:
                    continue
                is_mine = (is_red(p) and player == "red") or (is_black(p) and player == "black")
                is_k = is_king(p)
                on_edge = r == 0 or r == 7 or c == 0 or c == 7
                in_center = 2 <= r <= 5 and 2 <= c <= 5

                if is_mine:
                    my_pieces += 1
                    if is_k:
                        my_kings += 1
                    if in_center:
                        center_control += 1
                    if on_edge:
                        my_edge += 1
                else:
                    opp_pieces += 1
                    if is_k:
                        opp_kings += 1
                    if on_edge:
                        opp_edge += 1

        # Check back row
        back_row = 7 if player == "red" else 0
        back_row_intact = all(
            (is_red(board[back_row][c]) if player == "red" else is_black(board[back_row][c]))
            or board[back_row][c] == Piece.EMPTY
            for c in range(BOARD_SIZE)
            if (back_row + c) % 2 == 1
        )

        my_mobility = len(self._game.get_legal_moves(player))
        opp_color = "black" if player == "red" else "red"
        opp_mobility = len(self._game.get_legal_moves(opp_color))

        # Check for jump opportunities
        from checkers import same_side
        jumps_available = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if same_side(board[r][c], player):
                    jumps_available += len(self._game._get_jumps(r, c))

        return {
            "board": [[int(board[r][c]) for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)],
            "my_color": player,
            "move_number": self._game.move_count,
            "my_pieces": my_pieces,
            "my_kings": my_kings,
            "opponent_pieces": opp_pieces,
            "opponent_kings": opp_kings,
            "material_advantage": my_pieces - opp_pieces,
            "center_control": center_control,
            "my_mobility": my_mobility,
            "opponent_mobility": opp_mobility,
            "back_row_intact": back_row_intact,
            "my_edge_pieces": my_edge,
            "opponent_edge_pieces": opp_edge,
            "jumps_available": jumps_available,
            "no_capture_count": self._game.no_capture_count,
        }
