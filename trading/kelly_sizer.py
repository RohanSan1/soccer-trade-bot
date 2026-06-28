"""Quarter-Kelly position sizing.

Implements Kelly criterion with fraction Kelly (default 0.25)
and maximum bet cap (default 2% of bankroll).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BetSize:
    """Calculated bet size for a trade."""

    outcome: str
    edge: float
    market_prob: float
    kelly_fraction: float  # Full Kelly fraction
    quarter_kelly: float  # Fractional Kelly
    bet_usd: float  # Final bet size
    bankroll_pct: float  # As percentage of bankroll


class KellySizer:
    """Quarter-Kelly position sizing with bankroll management.

    Formula:
        f = edge / (1 - market_implied_prob)
        bet = kelly_fraction * f * bankroll
        cap at max_bet_pct of bankroll

    Args:
        kelly_fraction: Fraction of full Kelly to use (default 0.25).
        max_bet_pct: Maximum bet as percentage of bankroll (default 0.02).
        min_bet_usd: Minimum bet size in USD (default 5.0).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.02,
        min_bet_usd: float = 5.0,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_bet_usd = min_bet_usd

    def calculate(
        self,
        outcome: str,
        edge: float,
        market_prob: float,
        bankroll: float,
    ) -> BetSize:
        """Calculate optimal bet size.

        Args:
            outcome: 'home', 'draw', or 'away'.
            edge: Model probability minus market probability.
            market_prob: Market implied probability.
            bankroll: Current bankroll in USD.

        Returns:
            BetSize with calculated position.
        """
        if edge <= 0 or market_prob <= 0 or market_prob >= 1:
            return BetSize(
                outcome=outcome,
                edge=edge,
                market_prob=market_prob,
                kelly_fraction=0.0,
                quarter_kelly=0.0,
                bet_usd=0.0,
                bankroll_pct=0.0,
            )

        # Full Kelly: f = edge / (1 - market_prob)
        # This is the optimal fraction for a binary bet
        full_kelly = edge / (1.0 - market_prob)

        # Fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction

        # Calculate bet size
        bet_usd = fractional_kelly * bankroll

        # Apply cap
        max_bet = bankroll * self.max_bet_pct
        if bet_usd > max_bet:
            bet_usd = max_bet
            logger.debug(
                "Bet capped at $%.2f (%.1f%% of bankroll)",
                max_bet, self.max_bet_pct * 100,
            )

        # Apply minimum
        if bet_usd < self.min_bet_usd:
            bet_usd = 0.0

        bankroll_pct = (bet_usd / bankroll * 100) if bankroll > 0 else 0.0

        return BetSize(
            outcome=outcome,
            edge=edge,
            market_prob=market_prob,
            kelly_fraction=full_kelly,
            quarter_kelly=fractional_kelly,
            bet_usd=bet_usd,
            bankroll_pct=bankroll_pct,
        )

    def calculate_all(
        self,
        edges: dict[str, float],
        market_probs: dict[str, float],
        bankroll: float,
    ) -> list[BetSize]:
        """Calculate bet sizes for all outcomes with positive edge.

        Args:
            edges: {'home': 0.08, 'draw': -0.02, 'away': -0.05}
            market_probs: {'home': 0.60, 'draw': 0.22, 'away': 0.18}
            bankroll: Current bankroll.

        Returns:
            List of BetSize for outcomes with edge > 0.
        """
        bets = []

        for outcome in ["home", "draw", "away"]:
            edge = edges.get(outcome, 0.0)
            market_prob = market_probs.get(outcome, 0.0)

            if edge > 0:
                bet = self.calculate(outcome, edge, market_prob, bankroll)
                if bet.bet_usd > 0:
                    bets.append(bet)

        # Sort by edge (best first)
        bets.sort(key=lambda b: b.edge, reverse=True)

        return bets

    def update_bankroll(
        self,
        current_bankroll: float,
        pnl: float,
    ) -> float:
        """Update bankroll after a trade outcome.

        Args:
            current_bankroll: Previous bankroll.
            pnl: Profit/loss from trade.

        Returns:
            New bankroll amount.
        """
        new_bankroll = current_bankroll + pnl
        if new_bankroll < 0:
            logger.warning("Bankroll went negative: $%.2f", new_bankroll)
            return 0.0
        return new_bankroll
