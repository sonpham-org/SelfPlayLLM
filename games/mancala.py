"""Mancala (Kalah variant) game engine.

Standard Kalah rules with 6 pits per player, 4 seeds each, and stores.
Counter-clockwise sowing, extra turns, and capture mechanics.
"""

import re
from typing import Any, Optional

from games.base import BaseGame

# Board layout (internal representation):
#   Indices 0-5:   Player1's pits (left to right from P1's perspective)
#   Index 6:       Player1's store
#   Indices 7-12:  Player2's pits (left to right from P2's perspective)
#   Index 13:      Player2's store
#
# Counter-clockwise sowing order from P1's perspective:
#   P1 pit0 -> pit1 -> ... -> pit5 -> P1 store -> P2 pit0 -> ... -> P2 pit5 -> (skip P2 store) -> P1 pit0 ...
#
# Visual board:
#        [P2 pit5] [P2 pit4] [P2 pit3] [P2 pit2] [P2 pit1] [P2 pit0]
# [P2 store]                                                            [P1 store]
#        [P1 pit0] [P1 pit1] [P1 pit2] [P1 pit3] [P1 pit4] [P1 pit5]

PITS_PER_PLAYER = 6
INITIAL_SEEDS = 4
TOTAL_SEEDS = INITIAL_SEEDS * PITS_PER_PLAYER * 2  # 48

P1_PITS = list(range(0, 6))       # indices 0-5
P1_STORE = 6
P2_PITS = list(range(7, 13))      # indices 7-12
P2_STORE = 13

# Opposite pit mapping: P1 pit i <-> P2 pit (5 - i)
# P1 pit 0 (index 0) is opposite P2 pit 5 (index 12)
# P1 pit 1 (index 1) is opposite P2 pit 4 (index 11)
# ...
OPPOSITE = {}
for i in range(6):
    OPPOSITE[P1_PITS[i]] = P2_PITS[5 - i]
    OPPOSITE[P2_PITS[i]] = P1_PITS[5 - i]


class MancalaGame(BaseGame):
    """Kalah variant of Mancala for two players."""

    def __init__(self):
        # Internal board: 14 slots (6 pits + 1 store per player)
        self._board = [INITIAL_SEEDS] * 14
        self._board[P1_STORE] = 0
        self._board[P2_STORE] = 0
        self._current_player = "player1"
        self._move_count = 0

    @property
    def players(self) -> tuple[str, str]:
        return ("player1", "player2")

    @property
    def current_player(self) -> str:
        return self._current_player

    @property
    def move_count(self) -> int:
        return self._move_count

    def copy(self) -> "MancalaGame":
        g = MancalaGame.__new__(MancalaGame)
        g._board = self._board[:]
        g._current_player = self._current_player
        g._move_count = self._move_count
        return g

    def _player_pits(self, player: str) -> list[int]:
        """Return board indices for a player's pits."""
        return P1_PITS if player == "player1" else P2_PITS

    def _player_store(self, player: str) -> int:
        """Return board index for a player's store."""
        return P1_STORE if player == "player1" else P2_STORE

    def _opponent(self, player: str) -> str:
        return "player2" if player == "player1" else "player1"

    def get_legal_moves(self, player: str | None = None) -> list[int]:
        """Return list of legal pit indices (0-5) for the player."""
        if player is None:
            player = self._current_player
        pits = self._player_pits(player)
        return [i for i in range(PITS_PER_PLAYER) if self._board[pits[i]] > 0]

    def _simulate_sow(self, pit_index: int, player: str) -> tuple[int, bool, bool]:
        """Simulate sowing without modifying state.

        Returns (last_index, lands_in_own_store, lands_in_empty_own_pit_with_capture).
        """
        pits = self._player_pits(player)
        store = self._player_store(player)
        opp_store = self._player_store(self._opponent(player))
        board_idx = pits[pit_index]
        seeds = self._board[board_idx]
        if seeds == 0:
            return -1, False, False

        pos = board_idx
        for _ in range(seeds):
            pos = (pos + 1) % 14
            if pos == opp_store:
                pos = (pos + 1) % 14

        lands_in_store = (pos == store)

        # Check capture: last seed lands in an empty own pit with non-empty opposite
        own_pits = set(pits)
        lands_in_empty_own = False
        if pos in own_pits:
            # After sowing, the pit at pos will have 1 seed IF it was empty before
            # (since we haven't modified the board, check current count)
            # But we need to account for seeds that may have been added during sowing
            # For accurate simulation, do a full sow on a copy
            temp_board = self._board[:]
            temp_board[board_idx] = 0
            cur = board_idx
            for _ in range(seeds):
                cur = (cur + 1) % 14
                if cur == opp_store:
                    cur = (cur + 1) % 14
                temp_board[cur] += 1
            if temp_board[pos] == 1 and temp_board[OPPOSITE[pos]] > 0:
                lands_in_empty_own = True

        return pos, lands_in_store, lands_in_empty_own

    def apply_move(self, move: Any) -> None:
        """Apply a move (pit index 0-5) for the current player."""
        pit_index = move
        player = self._current_player
        pits = self._player_pits(player)
        store = self._player_store(player)
        opp_store = self._player_store(self._opponent(player))
        board_idx = pits[pit_index]

        if self._board[board_idx] == 0:
            raise ValueError(f"Pit {pit_index} is empty")

        # Pick up seeds
        seeds = self._board[board_idx]
        self._board[board_idx] = 0

        # Sow counter-clockwise
        pos = board_idx
        for _ in range(seeds):
            pos = (pos + 1) % 14
            if pos == opp_store:
                pos = (pos + 1) % 14
            self._board[pos] += 1

        # Check capture: last seed in an empty own pit (now has exactly 1)
        own_pits_set = set(pits)
        if pos in own_pits_set and self._board[pos] == 1:
            opp_idx = OPPOSITE[pos]
            if self._board[opp_idx] > 0:
                # Capture: move both captured seed and opposite seeds to store
                self._board[store] += self._board[pos] + self._board[opp_idx]
                self._board[pos] = 0
                self._board[opp_idx] = 0

        self._move_count += 1

        # Check if game is over (one side is empty)
        p1_empty = all(self._board[i] == 0 for i in P1_PITS)
        p2_empty = all(self._board[i] == 0 for i in P2_PITS)

        if p1_empty or p2_empty:
            # Sweep remaining seeds to respective stores
            for i in P1_PITS:
                self._board[P1_STORE] += self._board[i]
                self._board[i] = 0
            for i in P2_PITS:
                self._board[P2_STORE] += self._board[i]
                self._board[i] = 0
            # Don't switch turns; game is over
            return

        # Extra turn if last seed landed in own store
        if pos == store:
            # Current player gets another turn
            return

        # Switch turns
        self._current_player = self._opponent(player)

    def winner(self) -> Optional[str]:
        """Return 'player1', 'player2', 'draw', or None if game ongoing."""
        p1_empty = all(self._board[i] == 0 for i in P1_PITS)
        p2_empty = all(self._board[i] == 0 for i in P2_PITS)

        if not p1_empty and not p2_empty:
            return None

        # Game over: compute final scores including any remaining pit seeds
        # (apply_move sweeps these, but handle the case where winner() is
        # called on a state that wasn't reached via apply_move)
        p1_score = self._board[P1_STORE] + sum(self._board[i] for i in P1_PITS)
        p2_score = self._board[P2_STORE] + sum(self._board[i] for i in P2_PITS)

        if p1_score > p2_score:
            return "player1"
        elif p2_score > p1_score:
            return "player2"
        else:
            return "draw"

    def render(self, perspective: str | None = None) -> str:
        """Return ASCII art of the board.

        Layout:
                [P2 pit5] [P2 pit4] [P2 pit3] [P2 pit2] [P2 pit1] [P2 pit0]
        [P2 store]                                                            [P1 store]
                [P1 pit0] [P1 pit1] [P1 pit2] [P1 pit3] [P1 pit4] [P1 pit5]
        """
        b = self._board

        # Top row: P2's pits in reverse order (pit5 on left, pit0 on right)
        p2_pits_display = [b[P2_PITS[5 - i]] for i in range(6)]
        # Bottom row: P1's pits in order (pit0 on left, pit5 on right)
        p1_pits_display = [b[P1_PITS[i]] for i in range(6)]

        p1_store = b[P1_STORE]
        p2_store = b[P2_STORE]

        # Format with consistent width
        pit_width = 4  # width per pit cell

        def fmt_pit(n: int) -> str:
            return f"{n:>{pit_width}}"

        top_row = "     " + " ".join(fmt_pit(v) for v in p2_pits_display)
        mid_row = f" {p2_store:<3}" + " " * (pit_width * 6 + 5) + f"  {p1_store:>3}"
        bot_row = "     " + " ".join(fmt_pit(v) for v in p1_pits_display)

        # Pit labels (each label is right-justified in pit_width chars)
        p2_labels = "     " + " ".join(f"{'p'+str(5-i):>{pit_width}}" for i in range(6))
        p1_labels = "     " + " ".join(f"{'p'+str(i):>{pit_width}}" for i in range(6))

        lines = [
            "         Player 2",
            p2_labels,
            top_row,
            mid_row,
            bot_row,
            p1_labels,
            "         Player 1",
            "",
            f"  Stores:  P1={p1_store}  P2={p2_store}",
            f"  Turn: {self._current_player}  |  Move: {self._move_count}",
        ]

        return "\n".join(lines)

    def get_state(self, player: str) -> dict:
        """Return game state dict from the perspective of the given player."""
        opp = self._opponent(player)
        my_pits_idx = self._player_pits(player)
        opp_pits_idx = self._player_pits(opp)
        my_store_idx = self._player_store(player)
        opp_store_idx = self._player_store(opp)

        my_pits = [self._board[i] for i in my_pits_idx]
        opp_pits = [self._board[i] for i in opp_pits_idx]
        my_store = self._board[my_store_idx]
        opp_store = self._board[opp_store_idx]

        my_total = sum(my_pits) + my_store
        opp_total = sum(opp_pits) + opp_store

        # Compute capture opportunities and extra turn moves
        capture_opportunities = 0
        extra_turn_moves = []

        for pit_i in range(PITS_PER_PLAYER):
            if self._board[my_pits_idx[pit_i]] == 0:
                continue
            _, lands_in_store, lands_in_capture = self._simulate_sow(pit_i, player)
            if lands_in_capture:
                capture_opportunities += 1
            if lands_in_store:
                extra_turn_moves.append(pit_i)

        return {
            "board": {
                "my_pits": my_pits,
                "opponent_pits": opp_pits,
                "my_store": my_store,
                "opponent_store": opp_store,
            },
            "my_color": player,
            "move_number": self._move_count,
            "total_seeds": TOTAL_SEEDS,
            "my_total_seeds": my_total,
            "opponent_total_seeds": opp_total,
            "capture_opportunities": capture_opportunities,
            "extra_turn_moves": extra_turn_moves,
            "seeds_to_win": (TOTAL_SEEDS // 2) + 1,  # 25
        }

    def move_to_str(self, move: Any) -> str:
        """Convert a pit index to a human-readable string."""
        return str(move)

    def parse_move(self, text: str, player: str) -> Any:
        """Parse a move string from LLM output.

        Accepts formats like '3', 'pit 3', 'Pit 3', 'p3', 'pit3', etc.
        Returns the matched legal move (int) or None.
        """
        text = text.strip()

        # Try to extract a single digit 0-5
        # Match patterns: "3", "pit 3", "pit3", "p3", "Pit 3", etc.
        match = re.search(r'(?:pit\s*)?(\d)', text, re.IGNORECASE)
        if match:
            pit = int(match.group(1))
            legal = self.get_legal_moves(player)
            if pit in legal:
                return pit

        return None
