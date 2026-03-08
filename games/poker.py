"""Heads-up Texas Hold'em Poker engine for the evolution framework.

A single hand constitutes one "game". The evolution framework plays many hands.
Two players: player1 (small blind) and player2 (big blind).
"""

import random
import re
from collections import Counter
from itertools import combinations
from typing import Any, Optional

from games.base import BaseGame

# ── Constants ─────────────────────────────────────────────────────────

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}  # 2=0 .. A=12

SMALL_BLIND = 10
BIG_BLIND = 20
STARTING_CHIPS = 1000

HAND_RANKS = {
    "High Card": 0,
    "One Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Four of a Kind": 7,
    "Straight Flush": 8,
    "Royal Flush": 9,
}

BETTING_ROUNDS = ["preflop", "flop", "turn", "river"]


# ── Card utilities ────────────────────────────────────────────────────

def make_deck() -> list[tuple[str, str]]:
    """Return a full 52-card deck as list of (rank, suit) tuples."""
    return [(r, s) for r in RANKS for s in SUITS]


def card_str(card: tuple[str, str]) -> str:
    """Return a short string like 'As' for (A, s)."""
    return f"{card[0]}{card[1]}"


def cards_str(cards: list[tuple[str, str]]) -> str:
    """Return space-separated card strings."""
    return " ".join(card_str(c) for c in cards)


# ── Hand evaluation ───────────────────────────────────────────────────

def _is_flush(cards: list[tuple[str, str]]) -> bool:
    return len(set(c[1] for c in cards)) == 1


def _is_straight(ranks_sorted: list[int]) -> bool:
    """Check if 5 sorted rank values form a straight. Handles A-low (wheel)."""
    if ranks_sorted == list(range(ranks_sorted[0], ranks_sorted[0] + 5)):
        return True
    # Ace-low straight: A 2 3 4 5 -> values [0,1,2,3,12]
    if ranks_sorted == [0, 1, 2, 3, 12]:
        return True
    return False


def _straight_high(ranks_sorted: list[int]) -> int:
    """Return the high card value of a straight. For wheel, high is 3 (the 5)."""
    if ranks_sorted == [0, 1, 2, 3, 12]:
        return 3  # 5-high straight
    return ranks_sorted[-1]


def evaluate_five(cards: list[tuple[str, str]]) -> tuple[int, list[int]]:
    """Evaluate a 5-card hand. Returns (hand_rank, tiebreakers).

    hand_rank: 0 (High Card) to 9 (Royal Flush).
    tiebreakers: list of integers for comparing hands of the same rank,
                 in descending order of importance.
    """
    rank_vals = sorted(RANK_VALUE[c[0]] for c in cards)
    counts = Counter(RANK_VALUE[c[0]] for c in cards)
    flush = _is_flush(cards)
    straight = _is_straight(rank_vals)

    # Group by frequency then by rank (descending) for tiebreaking
    # e.g. for full house KKK77: groups = [(3, 11), (2, 5)]
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freq_pattern = tuple(g[1] for g in groups)

    if flush and straight:
        high = _straight_high(rank_vals)
        if high == 12:  # Ace-high straight flush = Royal Flush
            return (HAND_RANKS["Royal Flush"], [high])
        return (HAND_RANKS["Straight Flush"], [high])

    if freq_pattern == (4, 1):
        quad_rank = groups[0][0]
        kicker = groups[1][0]
        return (HAND_RANKS["Four of a Kind"], [quad_rank, kicker])

    if freq_pattern == (3, 2):
        trip_rank = groups[0][0]
        pair_rank = groups[1][0]
        return (HAND_RANKS["Full House"], [trip_rank, pair_rank])

    if flush:
        return (HAND_RANKS["Flush"], sorted(rank_vals, reverse=True))

    if straight:
        high = _straight_high(rank_vals)
        return (HAND_RANKS["Straight"], [high])

    if freq_pattern == (3, 1, 1):
        trip_rank = groups[0][0]
        kickers = sorted([groups[1][0], groups[2][0]], reverse=True)
        return (HAND_RANKS["Three of a Kind"], [trip_rank] + kickers)

    if freq_pattern == (2, 2, 1):
        pair1 = groups[0][0]
        pair2 = groups[1][0]
        high_pair = max(pair1, pair2)
        low_pair = min(pair1, pair2)
        kicker = groups[2][0]
        return (HAND_RANKS["Two Pair"], [high_pair, low_pair, kicker])

    if freq_pattern == (2, 1, 1, 1):
        pair_rank = groups[0][0]
        kickers = sorted([groups[1][0], groups[2][0], groups[3][0]], reverse=True)
        return (HAND_RANKS["One Pair"], [pair_rank] + kickers)

    # High card
    return (HAND_RANKS["High Card"], sorted(rank_vals, reverse=True))


def evaluate_hand(hole: list[tuple[str, str]],
                  community: list[tuple[str, str]]) -> tuple[int, list[int], str]:
    """Evaluate the best 5-card hand from hole + community cards.

    Returns (hand_rank, tiebreakers, hand_name).
    """
    all_cards = hole + community
    if len(all_cards) < 5:
        # Not enough cards yet; evaluate what we have (partial hand)
        if len(all_cards) == 0:
            return (0, [0], "High Card")
        # Pad evaluation: just return high card of available cards
        rank_vals = sorted((RANK_VALUE[c[0]] for c in all_cards), reverse=True)
        return (0, rank_vals, "High Card")

    best_rank = (-1, [], "High Card")
    rank_names = {v: k for k, v in HAND_RANKS.items()}

    for combo in combinations(all_cards, 5):
        hr, tb = evaluate_five(list(combo))
        if (hr, tb) > (best_rank[0], best_rank[1]):
            best_rank = (hr, tb, rank_names[hr])

    return best_rank


# ── Poker Game ────────────────────────────────────────────────────────

class PokerGame(BaseGame):
    """Heads-up Texas Hold'em. One hand = one game.

    player1 is Small Blind (SB), player2 is Big Blind (BB).
    SB acts first in all betting rounds (heads-up simplification).
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

        # Deck and dealing
        self._deck = make_deck()
        self._rng.shuffle(self._deck)
        self._deck_idx = 0

        # Hole cards
        self._hands: dict[str, list[tuple[str, str]]] = {
            "player1": [],
            "player2": [],
        }
        self._community: list[tuple[str, str]] = []

        # Chip stacks
        self._chips: dict[str, int] = {
            "player1": STARTING_CHIPS,
            "player2": STARTING_CHIPS,
        }

        # Pot and betting
        self._pot: int = 0
        self._round_idx: int = 0  # index into BETTING_ROUNDS
        self._current_bet: int = 0  # current bet level this round
        self._player_bet_round: dict[str, int] = {"player1": 0, "player2": 0}
        self._player_bet_total: dict[str, int] = {"player1": 0, "player2": 0}
        self._turn: str = "player1"  # who acts next
        self._move_count_val: int = 0

        # Action history
        self._actions: dict[str, list[str]] = {"player1": [], "player2": []}
        self._all_actions: list[tuple[str, str]] = []  # (player, action_str)

        # Round action tracking (to know when a round is complete)
        self._acted_this_round: dict[str, bool] = {"player1": False, "player2": False}
        self._last_raiser: Optional[str] = None

        # Game outcome
        self._winner: Optional[str] = None
        self._folded: Optional[str] = None

        # All-in tracking
        self._all_in: dict[str, bool] = {"player1": False, "player2": False}

        # Hands played counter (for multi-hand tracking by external code)
        self._hands_played: int = 0

        # Set up the hand
        self._post_blinds()
        self._deal_hole_cards()

    # ── Internal setup ────────────────────────────────────────────────

    def _draw_cards(self, n: int) -> list[tuple[str, str]]:
        cards = self._deck[self._deck_idx:self._deck_idx + n]
        self._deck_idx += n
        return cards

    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        sb = min(SMALL_BLIND, self._chips["player1"])
        bb = min(BIG_BLIND, self._chips["player2"])

        self._chips["player1"] -= sb
        self._chips["player2"] -= bb
        self._pot = sb + bb

        self._player_bet_round["player1"] = sb
        self._player_bet_round["player2"] = bb
        self._player_bet_total["player1"] = sb
        self._player_bet_total["player2"] = bb
        self._current_bet = bb

        if self._chips["player1"] == 0:
            self._all_in["player1"] = True
        if self._chips["player2"] == 0:
            self._all_in["player2"] = True

    def _deal_hole_cards(self) -> None:
        self._hands["player1"] = self._draw_cards(2)
        self._hands["player2"] = self._draw_cards(2)

    def _deal_community(self) -> None:
        """Deal the next community cards based on current round."""
        round_name = BETTING_ROUNDS[self._round_idx]
        if round_name == "flop":
            self._community.extend(self._draw_cards(3))
        elif round_name in ("turn", "river"):
            self._community.extend(self._draw_cards(1))

    # ── BaseGame properties ───────────────────────────────────────────

    @property
    def players(self) -> tuple[str, str]:
        return ("player1", "player2")

    @property
    def current_player(self) -> str:
        return self._turn

    @property
    def move_count(self) -> int:
        return self._move_count_val

    # ── Copy ──────────────────────────────────────────────────────────

    def copy(self) -> "PokerGame":
        g = PokerGame.__new__(PokerGame)
        g._rng = random.Random()
        g._rng.setstate(self._rng.getstate())
        g._deck = list(self._deck)
        g._deck_idx = self._deck_idx
        g._hands = {k: list(v) for k, v in self._hands.items()}
        g._community = list(self._community)
        g._chips = dict(self._chips)
        g._pot = self._pot
        g._round_idx = self._round_idx
        g._current_bet = self._current_bet
        g._player_bet_round = dict(self._player_bet_round)
        g._player_bet_total = dict(self._player_bet_total)
        g._turn = self._turn
        g._move_count_val = self._move_count_val
        g._actions = {k: list(v) for k, v in self._actions.items()}
        g._all_actions = list(self._all_actions)
        g._acted_this_round = dict(self._acted_this_round)
        g._last_raiser = self._last_raiser
        g._winner = self._winner
        g._folded = self._folded
        g._all_in = dict(self._all_in)
        g._hands_played = self._hands_played
        return g

    # ── Helpers ───────────────────────────────────────────────────────

    def _other(self, player: str) -> str:
        return "player2" if player == "player1" else "player1"

    def _cost_to_call(self, player: str) -> int:
        """How much the player needs to add to match the current bet."""
        diff = self._current_bet - self._player_bet_round[player]
        return min(diff, self._chips[player])

    def _min_raise_total(self) -> int:
        """Minimum total bet for a raise (current bet + big blind increment)."""
        return self._current_bet + BIG_BLIND

    def _advance_round(self) -> None:
        """Move to the next betting round or showdown."""
        self._round_idx += 1

        if self._round_idx >= len(BETTING_ROUNDS):
            self._showdown()
            return

        # Deal community cards for the new round
        self._deal_community()

        # If both all-in, run out remaining community cards
        if self._all_in["player1"] and self._all_in["player2"]:
            self._run_out_board()
            return

        # Reset round betting state
        self._player_bet_round["player1"] = 0
        self._player_bet_round["player2"] = 0
        self._current_bet = 0
        self._acted_this_round = {"player1": False, "player2": False}
        self._last_raiser = None

        # SB (player1) acts first in all rounds
        self._turn = "player1"

    def _run_out_board(self) -> None:
        """Deal remaining community cards and go to showdown (both all-in).

        Assumes current _round_idx has already been dealt.
        """
        self._round_idx += 1
        while self._round_idx < len(BETTING_ROUNDS):
            self._deal_community()
            self._round_idx += 1

        # Keep at "river" for state queries
        self._round_idx = len(BETTING_ROUNDS) - 1
        self._showdown()

    def _showdown(self) -> None:
        """Compare hands and determine the winner."""
        h1 = evaluate_hand(self._hands["player1"], self._community)
        h2 = evaluate_hand(self._hands["player2"], self._community)

        score1 = (h1[0], h1[1])
        score2 = (h2[0], h2[1])

        if score1 > score2:
            self._winner = "player1"
        elif score2 > score1:
            self._winner = "player2"
        else:
            self._winner = "draw"

        self._distribute_pot()

    def _distribute_pot(self) -> None:
        """Give pot to winner, or split on draw."""
        if self._winner == "draw":
            half = self._pot // 2
            self._chips["player1"] += half
            self._chips["player2"] += self._pot - half
        elif self._winner in ("player1", "player2"):
            self._chips[self._winner] += self._pot
        self._pot = 0

    def _is_round_complete(self) -> bool:
        """Check if the current betting round is complete."""
        # If someone folded, hand is over
        if self._folded is not None:
            return True

        p1_acted = self._acted_this_round["player1"]
        p2_acted = self._acted_this_round["player2"]

        # Both must have acted at least once
        if not p1_acted or not p2_acted:
            return False

        # Bets must be matched (or someone is all-in)
        p1_matched = (self._player_bet_round["player1"] == self._current_bet
                      or self._all_in["player1"])
        p2_matched = (self._player_bet_round["player2"] == self._current_bet
                      or self._all_in["player2"])

        return p1_matched and p2_matched

    # ── Legal moves ───────────────────────────────────────────────────

    def get_legal_moves(self, player: str | None = None) -> list[str]:
        """Return list of legal move strings for the given player."""
        if player is None:
            player = self._turn

        if self._winner is not None or self._folded is not None:
            return []

        # Not this player's turn
        if player != self._turn:
            return []

        # Player is all-in, no actions available
        if self._all_in[player]:
            return []

        moves = []
        cost = self._cost_to_call(player)
        chips = self._chips[player]

        if cost == 0:
            # No bet to match
            moves.append("check")
        else:
            moves.append("fold")
            if chips >= cost:
                moves.append("call")

        # Raise: must have chips beyond calling
        if chips > cost:
            min_raise = self._min_raise_total()
            max_raise = self._player_bet_round[player] + chips  # all-in amount

            if min_raise <= max_raise:
                # Add min-raise
                moves.append(f"raise {min_raise}")

                # Add some standard raise sizes if affordable
                for multiplier in [2, 3]:
                    amount = self._current_bet * multiplier
                    if min_raise < amount < max_raise:
                        moves.append(f"raise {amount}")

                # Pot-sized raise
                pot_raise = self._current_bet + self._pot + cost
                if min_raise < pot_raise < max_raise:
                    # Avoid duplicates
                    pot_str = f"raise {pot_raise}"
                    if pot_str not in moves:
                        moves.append(pot_str)

            # All-in is always an option if player has chips
            moves.append("all_in")

        elif chips == cost and cost > 0:
            # Calling would put us all-in; offer all_in as well
            moves.append("all_in")

        # Fold is always available if there is a bet (already added above),
        # but also allow fold even when checking is available (unusual but legal)
        if "fold" not in moves and cost == 0:
            moves.append("fold")

        return moves

    # ── Apply move ────────────────────────────────────────────────────

    def apply_move(self, move: Any) -> None:
        """Apply a move string: fold, check, call, raise N, all_in."""
        move = str(move).strip().lower()
        player = self._turn
        opponent = self._other(player)

        if self._winner is not None or self._folded is not None:
            raise ValueError("Hand is already over")

        if self._all_in[player]:
            raise ValueError(f"{player} is all-in and cannot act")

        cost = self._cost_to_call(player)
        chips = self._chips[player]

        action_str = move  # for history

        if move == "fold":
            self._folded = player
            self._winner = opponent
            self._distribute_pot()
            action_str = "fold"

        elif move == "check":
            if cost != 0:
                raise ValueError(f"Cannot check: must call {cost} or fold")
            action_str = "check"

        elif move == "call":
            if cost == 0:
                raise ValueError("Nothing to call; use 'check'")
            actual = min(cost, chips)
            self._chips[player] -= actual
            self._player_bet_round[player] += actual
            self._player_bet_total[player] += actual
            self._pot += actual
            if self._chips[player] == 0:
                self._all_in[player] = True
            action_str = f"call {actual}"

        elif move.startswith("raise"):
            parts = move.split()
            if len(parts) == 2:
                raise_to = int(parts[1])
            else:
                raise_to = self._min_raise_total()

            # Validate
            additional = raise_to - self._player_bet_round[player]
            if additional <= 0:
                raise ValueError(f"Raise to {raise_to} is not above current bet")
            if additional > chips:
                # Treat as all-in if they can't cover
                additional = chips
                raise_to = self._player_bet_round[player] + additional

            self._chips[player] -= additional
            self._player_bet_round[player] = raise_to
            self._player_bet_total[player] += additional
            self._pot += additional
            self._current_bet = raise_to
            self._last_raiser = player

            if self._chips[player] == 0:
                self._all_in[player] = True

            # After a raise, opponent needs to act again
            self._acted_this_round[opponent] = False

            action_str = f"raise {raise_to}"

        elif move in ("all_in", "allin"):
            amount = chips
            self._chips[player] -= amount
            new_bet = self._player_bet_round[player] + amount
            self._player_bet_round[player] = new_bet
            self._player_bet_total[player] += amount
            self._pot += amount
            self._all_in[player] = True

            if new_bet > self._current_bet:
                self._current_bet = new_bet
                self._last_raiser = player
                self._acted_this_round[opponent] = False

            action_str = f"all_in {amount}"

        else:
            raise ValueError(f"Unknown move: {move}")

        # Record action
        self._actions[player].append(action_str)
        self._all_actions.append((player, action_str))
        self._acted_this_round[player] = True
        self._move_count_val += 1

        # If hand is over (fold), we're done
        if self._folded is not None:
            return

        # Check if the round is complete
        if self._is_round_complete():
            # Both players all-in: run out the board
            if self._all_in["player1"] and self._all_in["player2"]:
                self._advance_round()
            # One player all-in: advance rounds without further betting
            elif self._all_in["player1"] or self._all_in["player2"]:
                # Run out remaining rounds
                while self._round_idx < len(BETTING_ROUNDS) - 1 and self._winner is None:
                    self._advance_round()
                if self._winner is None:
                    self._showdown()
            else:
                self._advance_round()
        else:
            # Switch to the other player
            self._turn = opponent

    # ── Winner ────────────────────────────────────────────────────────

    def winner(self) -> Optional[str]:
        return self._winner

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self, perspective: str | None = None) -> str:
        """Render the hand state. Hidden cards for opponent."""
        lines = []
        round_name = BETTING_ROUNDS[min(self._round_idx, len(BETTING_ROUNDS) - 1)]
        lines.append(f"=== Heads-Up Hold'em === Round: {round_name} ===")
        lines.append("")

        # Community cards
        if self._community:
            lines.append(f"Community: {cards_str(self._community)}")
        else:
            lines.append("Community: (none)")
        lines.append(f"Pot: {self._pot}")
        lines.append("")

        # Player info
        for p in ("player1", "player2"):
            pos = "SB" if p == "player1" else "BB"
            if perspective == p or self._winner is not None:
                hand_str = cards_str(self._hands[p])
            else:
                hand_str = "** **"
            status = ""
            if self._all_in[p]:
                status = " [ALL-IN]"
            if self._folded == p:
                status = " [FOLDED]"
            lines.append(
                f"{p} ({pos}): {hand_str}  "
                f"Chips: {self._chips[p]}{status}"
            )

        # Action history
        lines.append("")
        lines.append("Actions:")
        for player, action in self._all_actions:
            pos = "SB" if player == "player1" else "BB"
            lines.append(f"  {player} ({pos}): {action}")

        if self._winner is not None:
            lines.append("")
            if self._winner == "draw":
                lines.append("Result: Split pot")
            else:
                lines.append(f"Result: {self._winner} wins!")

        return "\n".join(lines)

    # ── Move formatting / parsing ─────────────────────────────────────

    def move_to_str(self, move: Any) -> str:
        return str(move)

    def parse_move(self, text: str, player: str) -> Optional[str]:
        """Parse LLM output into a legal move string.

        Handles various formats: 'fold', 'call', 'check', 'raise 60',
        'raise to 60', 'all in', 'all-in', 'allin', 'go all in', etc.
        """
        text = text.strip().lower()
        legal = self.get_legal_moves(player)

        if not legal:
            return None

        # Direct match
        if text in legal:
            return text

        # Fold
        if "fold" in text:
            if "fold" in legal:
                return "fold"

        # Check
        if "check" in text:
            if "check" in legal:
                return "check"

        # All-in variations
        if any(kw in text for kw in ["all_in", "all-in", "allin", "all in", "shove"]):
            if "all_in" in legal:
                return "all_in"

        # Raise with amount
        raise_match = re.search(r'raise\s*(?:to\s*)?(\d+)', text)
        if raise_match:
            amount = int(raise_match.group(1))
            target = f"raise {amount}"
            if target in legal:
                return target
            # Find closest legal raise
            legal_raises = [m for m in legal if m.startswith("raise")]
            if legal_raises:
                # Allow any valid raise amount within bounds
                min_raise = self._min_raise_total()
                max_raise = self._player_bet_round[player] + self._chips[player]
                if min_raise <= amount <= max_raise:
                    return f"raise {amount}"
                # Clamp to nearest legal
                if amount < min_raise:
                    return legal_raises[0]  # min raise
                if amount > max_raise:
                    return "all_in" if "all_in" in legal else legal_raises[-1]

        # Just "raise" with no amount -> min raise
        if re.search(r'\braise\b', text) and not raise_match:
            legal_raises = [m for m in legal if m.startswith("raise")]
            if legal_raises:
                return legal_raises[0]

        # Call
        if "call" in text:
            if "call" in legal:
                return "call"

        # Last resort: look for a number that could be a raise
        nums = re.findall(r'\b(\d+)\b', text)
        if nums:
            for n in nums:
                amount = int(n)
                if amount >= self._min_raise_total():
                    legal_raises = [m for m in legal if m.startswith("raise")]
                    if legal_raises:
                        max_raise = self._player_bet_round[player] + self._chips[player]
                        if self._min_raise_total() <= amount <= max_raise:
                            return f"raise {amount}"

        return None

    # ── State for strategies ──────────────────────────────────────────

    def get_state(self, player: str) -> dict:
        """Return game state visible to the given player.

        CRITICAL: Never exposes opponent's hole cards.
        """
        opponent = self._other(player)
        cost = self._cost_to_call(player)

        # Evaluate current best hand
        hr, tb, hand_name = evaluate_hand(self._hands[player], self._community)

        # Pot odds
        if cost > 0:
            pot_odds = round(self._pot / cost, 2)
        else:
            pot_odds = float("inf")

        position = "SB" if player == "player1" else "BB"

        return {
            "board": {
                "my_hand": list(self._hands[player]),
                "community": list(self._community),
                "pot": self._pot,
                "my_chips": self._chips[player],
                "opponent_chips": self._chips[opponent],
                "current_bet": self._current_bet,
                "my_bet_this_round": self._player_bet_round[player],
            },
            "my_color": player,
            "move_number": self._move_count_val,
            "round": BETTING_ROUNDS[min(self._round_idx, len(BETTING_ROUNDS) - 1)],
            "hand_strength": hand_name,
            "pot_odds": pot_odds,
            "position": position,
            "opponent_actions_this_hand": [
                a for p, a in self._all_actions if p == opponent
            ],
            "my_actions_this_hand": [
                a for p, a in self._all_actions if p == player
            ],
            "hands_played": self._hands_played,
        }
