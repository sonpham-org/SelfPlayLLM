"""Battleship game engine. 10x10 grids, hidden information, sink all ships to win."""

import random
import re
from typing import Any, Optional

from games.base import BaseGame

GRID_SIZE = 10

# Ship definitions: (name, length)
SHIPS = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]

MAX_MOVES = 200

# Cell markers for internal grids
WATER = "."
SHIP = "S"
HIT = "H"
MISS = "M"


def _place_ships(rng: random.Random) -> tuple[list[list[str]], dict[str, list[tuple[int, int]]]]:
    """Randomly place all ships on a 10x10 grid.

    Returns:
        grid: 10x10 list of '.' or 'S'
        ship_cells: dict mapping ship name -> list of (row, col) cells
    """
    grid = [[WATER] * GRID_SIZE for _ in range(GRID_SIZE)]
    ship_cells: dict[str, list[tuple[int, int]]] = {}

    for ship_name, ship_len in SHIPS:
        placed = False
        for _ in range(1000):  # safety limit
            orientation = rng.choice(["H", "V"])
            if orientation == "H":
                r = rng.randint(0, GRID_SIZE - 1)
                c = rng.randint(0, GRID_SIZE - ship_len)
                cells = [(r, c + i) for i in range(ship_len)]
            else:
                r = rng.randint(0, GRID_SIZE - ship_len)
                c = rng.randint(0, GRID_SIZE - 1)
                cells = [(r + i, c) for i in range(ship_len)]

            if all(grid[cr][cc] == WATER for cr, cc in cells):
                for cr, cc in cells:
                    grid[cr][cc] = SHIP
                ship_cells[ship_name] = cells
                placed = True
                break

        if not placed:
            raise RuntimeError(f"Failed to place {ship_name} after 1000 attempts")

    return grid, ship_cells


class BattleshipGame(BaseGame):
    """Standard Battleship on two 10x10 grids.

    Player1 goes first. Ships are placed randomly at creation time.
    Each turn a player calls a shot on the opponent's grid.
    First to sink all opponent ships wins. Max 200 total moves -> draw.
    """

    def __init__(self, seed: int | None = None):
        rng = random.Random(seed)

        # Each player has:
        #   fleet_grid: their own grid showing ship positions + incoming hits/misses
        #   attack_grid: what they see of the opponent (hits/misses they've fired)
        #   ship_cells: dict of ship_name -> list of (r,c) that ship occupies
        #   hits_received: set of (r,c) hit on this player's fleet
        self._grids: dict[str, list[list[str]]] = {}
        self._attack_grids: dict[str, list[list[str]]] = {}
        self._ship_cells: dict[str, dict[str, list[tuple[int, int]]]] = {}
        self._hits_received: dict[str, set[tuple[int, int]]] = {}
        self._shots_fired: dict[str, set[tuple[int, int]]] = {}
        self._ships_sunk_by: dict[str, list[str]] = {}
        self._ships_sunk_on: dict[str, list[str]] = {}

        for player in ("player1", "player2"):
            # Use separate random draws per player so seed is deterministic
            grid, ship_cells = _place_ships(rng)
            self._grids[player] = grid
            self._attack_grids[player] = [[WATER] * GRID_SIZE for _ in range(GRID_SIZE)]
            self._ship_cells[player] = ship_cells
            self._hits_received[player] = set()
            self._shots_fired[player] = set()
            self._ships_sunk_by[player] = []
            self._ships_sunk_on[player] = []

        self._turn: str = "player1"
        self._move_count: int = 0

    # -- BaseGame properties -----------------------------------------------

    @property
    def players(self) -> tuple[str, str]:
        return ("player1", "player2")

    @property
    def current_player(self) -> str:
        return self._turn

    @property
    def move_count(self) -> int:
        return self._move_count

    # -- Copy --------------------------------------------------------------

    def copy(self) -> "BattleshipGame":
        g = BattleshipGame.__new__(BattleshipGame)
        g._grids = {p: [row[:] for row in grid] for p, grid in self._grids.items()}
        g._attack_grids = {p: [row[:] for row in grid] for p, grid in self._attack_grids.items()}
        g._ship_cells = {
            p: {name: list(cells) for name, cells in ships.items()}
            for p, ships in self._ship_cells.items()
        }
        g._hits_received = {p: set(s) for p, s in self._hits_received.items()}
        g._shots_fired = {p: set(s) for p, s in self._shots_fired.items()}
        g._ships_sunk_by = {p: list(v) for p, v in self._ships_sunk_by.items()}
        g._ships_sunk_on = {p: list(v) for p, v in self._ships_sunk_on.items()}
        g._turn = self._turn
        g._move_count = self._move_count
        return g

    # -- Helpers -----------------------------------------------------------

    def _opponent(self, player: str) -> str:
        return "player2" if player == "player1" else "player1"

    def _is_ship_sunk(self, owner: str, ship_name: str) -> bool:
        """Check if all cells of a ship belonging to `owner` have been hit."""
        cells = self._ship_cells[owner][ship_name]
        return all((r, c) in self._hits_received[owner] for r, c in cells)

    def _ships_remaining(self, player: str) -> int:
        """Count how many of `player`'s ships are not yet fully sunk."""
        count = 0
        for ship_name in self._ship_cells[player]:
            if not self._is_ship_sunk(player, ship_name):
                count += 1
        return count

    def _all_ships_sunk(self, player: str) -> bool:
        """Check if all of `player`'s ships have been sunk."""
        return self._ships_remaining(player) == 0

    # -- Core game interface -----------------------------------------------

    def get_legal_moves(self, player: str | None = None) -> list[tuple[int, int]]:
        """Return list of (row, col) cells the player has not yet shot at."""
        if player is None:
            player = self._turn
        fired = self._shots_fired[player]
        moves = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) not in fired:
                    moves.append((r, c))
        return moves

    def apply_move(self, move: Any) -> None:
        """Apply a shot at (row, col) on the opponent's grid."""
        row, col = move
        player = self._turn
        opponent = self._opponent(player)

        if (row, col) in self._shots_fired[player]:
            raise ValueError(f"Cell ({row},{col}) already shot by {player}")
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            raise ValueError(f"Cell ({row},{col}) out of bounds")

        self._shots_fired[player].add((row, col))

        # Check hit or miss on opponent's fleet
        if self._grids[opponent][row][col] == SHIP:
            # Hit
            self._grids[opponent][row][col] = HIT
            self._attack_grids[player][row][col] = HIT
            self._hits_received[opponent].add((row, col))

            # Check if this hit sinks a ship
            for ship_name, cells in self._ship_cells[opponent].items():
                if (row, col) in cells and self._is_ship_sunk(opponent, ship_name):
                    if ship_name not in self._ships_sunk_by[player]:
                        self._ships_sunk_by[player].append(ship_name)
                        self._ships_sunk_on[opponent].append(ship_name)
        else:
            # Miss (could be water or already-missed water)
            self._grids[opponent][row][col] = MISS
            self._attack_grids[player][row][col] = MISS

        self._move_count += 1
        self._turn = opponent

    def winner(self) -> Optional[str]:
        """Return 'player1', 'player2', 'draw', or None if game is ongoing."""
        if self._all_ships_sunk("player2"):
            return "player1"
        if self._all_ships_sunk("player1"):
            return "player2"
        if self._move_count >= MAX_MOVES:
            return "draw"
        return None

    # -- Rendering ---------------------------------------------------------

    def render(self, perspective: str | None = None) -> str:
        """Render two grids side by side: MY FLEET and ATTACK BOARD.

        If perspective is None, defaults to current player.
        Only shows information visible to the given player:
        - Own fleet grid (ships visible)
        - Attack grid (hits/misses the player has fired, no enemy ship positions)
        """
        if perspective is None:
            perspective = self._turn

        fleet = self._grids[perspective]
        attack = self._attack_grids[perspective]

        header = "   " + " ".join(str(c) for c in range(GRID_SIZE))
        sep = "     "

        lines = []
        lines.append(f"  {'MY FLEET':^21s}{sep}{'ATTACK BOARD':^21s}")
        lines.append(f"  {header}{sep}  {header}")

        for r in range(GRID_SIZE):
            fleet_row = f"  {r} " + " ".join(fleet[r][c] for c in range(GRID_SIZE))
            attack_row = f"  {r} " + " ".join(attack[r][c] for c in range(GRID_SIZE))
            lines.append(f"{fleet_row}{sep}{attack_row}")

        lines.append("")
        lines.append(f"  Turn: {self._turn}  |  Move: {self._move_count}")
        lines.append(
            f"  {perspective} ships remaining: {self._ships_remaining(perspective)}  |  "
            f"Opponent ships remaining: {self._ships_remaining(self._opponent(perspective))}"
        )

        return "\n".join(lines)

    # -- Move parsing / formatting -----------------------------------------

    def move_to_str(self, move: Any) -> str:
        r, c = move
        return f"({r},{c})"

    def parse_move(self, text: str, player: str) -> Optional[tuple[int, int]]:
        """Parse LLM output into a (row, col) shot.

        Accepts formats like "(3,5)", "3,5", "(3, 5)", "row 3 col 5", etc.
        """
        text = text.strip()

        # Try (row, col) or row,col pattern
        m = re.search(r'\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?', text)
        if m:
            row, col = int(m.group(1)), int(m.group(2))
            if (row, col) in self.get_legal_moves(player):
                return (row, col)

        # Try "row R col C" pattern
        m = re.search(r'row\s*(\d+)\s*(?:,?\s*col(?:umn)?)\s*(\d+)', text, re.IGNORECASE)
        if m:
            row, col = int(m.group(1)), int(m.group(2))
            if (row, col) in self.get_legal_moves(player):
                return (row, col)

        # Last resort: find all digit sequences and take the first two
        digits = re.findall(r'\d+', text)
        if len(digits) >= 2:
            row, col = int(digits[0]), int(digits[1])
            if (row, col) in self.get_legal_moves(player):
                return (row, col)

        return None

    # -- State for strategies ----------------------------------------------

    def get_state(self, player: str) -> dict:
        """Return game state dict for strategy code.

        Includes both visible information and computed features useful
        for LLM-based strategy reasoning.
        """
        opponent = self._opponent(player)

        # Build my_grid: what the player sees of their own fleet
        my_grid = [row[:] for row in self._grids[player]]

        # Build attack_grid: what the player sees of the opponent
        attack_grid = [row[:] for row in self._attack_grids[player]]

        # Hit map: True where player has scored hits on the opponent
        hit_map = [
            [attack_grid[r][c] == HIT for c in range(GRID_SIZE)]
            for r in range(GRID_SIZE)
        ]

        # Count hits and misses
        my_hits_scored = sum(
            1 for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            if attack_grid[r][c] == HIT
        )
        my_misses = sum(
            1 for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            if attack_grid[r][c] == MISS
        )
        opponent_hits_on_me = len(self._hits_received[player])

        # Compute probability hotspots: cells adjacent to unsunk hits
        # (hits that are part of ships not yet fully sunk)
        unsunk_hit_cells = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if attack_grid[r][c] == HIT:
                    # Check if this hit is on a ship that is already sunk
                    # We know which ships we've sunk on the opponent
                    sunk_cells = set()
                    for sunk_name in self._ships_sunk_by[player]:
                        for cell in self._ship_cells[opponent][sunk_name]:
                            sunk_cells.add(cell)
                    if (r, c) not in sunk_cells:
                        unsunk_hit_cells.add((r, c))

        adjacent_to_unsunk = set()
        for r, c in unsunk_hit_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                        and attack_grid[nr][nc] == WATER):
                    adjacent_to_unsunk.add((nr, nc))

        shots_remaining = sum(
            1 for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            if attack_grid[r][c] == WATER
        )

        return {
            "board": {
                "my_grid": my_grid,
                "attack_grid": attack_grid,
            },
            "my_color": player,
            "move_number": self._move_count,
            "my_ships_remaining": self._ships_remaining(player),
            "opponent_ships_remaining": self._ships_remaining(opponent),
            "my_hits_scored": my_hits_scored,
            "my_misses": my_misses,
            "opponent_hits_on_me": opponent_hits_on_me,
            "ships_sunk_by_me": list(self._ships_sunk_by[player]),
            "ships_sunk_on_me": list(self._ships_sunk_on[player]),
            "hit_map": hit_map,
            "probability_hotspots": len(adjacent_to_unsunk),
            "shots_remaining_to_check": shots_remaining,
        }
