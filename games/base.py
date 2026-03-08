"""Base game interface for all games in the evolution framework."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseGame(ABC):
    """Abstract base class for two-player games.

    All games must implement this interface to work with the
    evolutionary strategy framework.
    """

    @property
    @abstractmethod
    def players(self) -> tuple[str, str]:
        """Return the two player identifiers, e.g. ('red', 'black')."""
        ...

    @property
    @abstractmethod
    def current_player(self) -> str:
        """Return which player's turn it is."""
        ...

    @property
    @abstractmethod
    def move_count(self) -> int:
        """Return total number of moves played so far."""
        ...

    @abstractmethod
    def copy(self) -> "BaseGame":
        """Return a deep copy of the game state."""
        ...

    @abstractmethod
    def get_legal_moves(self, player: str | None = None) -> list:
        """Return list of legal moves for the given player (or current player if None)."""
        ...

    @abstractmethod
    def apply_move(self, move: Any) -> None:
        """Apply a move, updating the game state and switching turns."""
        ...

    @abstractmethod
    def winner(self) -> Optional[str]:
        """Return winner player string, 'draw', or None if game is ongoing."""
        ...

    @abstractmethod
    def render(self, perspective: str | None = None) -> str:
        """Return ASCII representation of the board.

        For hidden-information games, perspective controls what's visible.
        """
        ...

    @abstractmethod
    def get_state(self, player: str) -> dict:
        """Return game state as a dict for strategy code to reference.

        Must include at minimum:
        - 'board': the board representation
        - 'move_number': current move count
        - 'my_color': the requesting player's identifier
        - Game-specific computed features
        """
        ...

    @abstractmethod
    def move_to_str(self, move: Any) -> str:
        """Convert a move object to a human-readable string."""
        ...

    @abstractmethod
    def parse_move(self, text: str, player: str) -> Any:
        """Parse a move string from LLM output and match to a legal move.

        Returns the matched move object, or None if parsing fails.
        """
        ...
