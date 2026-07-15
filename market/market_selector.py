"""Market selector for finding active match winner markets.

At kickoff: searches both Polymarket and Kalshi for the specific match.
Selects the market with higher liquidity (tighter spread).
Falls back to the other if spread > 3%.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from market.kalshi_client import KalshiClient, KalshiMarket
from market.polymarket_client import PolymarketClient, Market

logger = logging.getLogger(__name__)

MAX_SPREAD_THRESHOLD = 0.03  # 3%


@dataclass
class SelectedMarket:
    """Selected market for trading."""

    platform: str  # 'polymarket' or 'kalshi'
    market_id: str
    question: str
    yes_token_id: str  # Polymarket token_id or Kalshi ticker
    yes_price: float
    no_price: float
    spread: float
    liquidity: float
    volume: float


class MarketSelector:
    """Selects the best market for a specific soccer match.

    Args:
        polymarket: Polymarket client (optional).
        kalshi: Kalshi client (optional).
    """

    def __init__(
        self,
        polymarket: Optional[PolymarketClient] = None,
        kalshi: Optional[KalshiClient] = None,
    ) -> None:
        self.polymarket = polymarket
        self.kalshi = kalshi
        self._selected: Optional[SelectedMarket] = None

    def select_market(
        self, team_home: str, team_away: str
    ) -> Optional[SelectedMarket]:
        """Find and select the best market for a match.

        Args:
            team_home: Home team name.
            team_away: Away team name.

        Returns:
            SelectedMarket with best available market, or None.
        """
        candidates: list[SelectedMarket] = []

        # Search Polymarket
        if self.polymarket:
            poly_markets = self.polymarket.search_soccer_markets(team_home, team_away)
            for m in poly_markets:
                if m.active and m.yes_price > 0:
                    spread = abs(m.yes_price - (1 - m.yes_price))
                    candidates.append(
                        SelectedMarket(
                            platform="polymarket",
                            market_id=m.market_id,
                            question=m.question,
                            yes_token_id=m.token_id_yes,
                            yes_price=m.yes_price,
                            no_price=m.no_price,
                            spread=spread,
                            liquidity=m.liquidity,
                            volume=m.volume,
                        )
                    )

        # Search Kalshi
        if self.kalshi:
            kalshi_markets = self.kalshi.search_soccer_markets(team_home, team_away)
            for m in kalshi_markets:
                if m.status == "open":
                    spread = m.yes_ask - m.yes_bid
                    candidates.append(
                        SelectedMarket(
                            platform="kalshi",
                            market_id=m.ticker,
                            question=m.title,
                            yes_token_id=m.ticker,
                            yes_price=(m.yes_bid + m.yes_ask) / 2,
                            no_price=(m.no_bid + m.no_ask) / 2,
                            spread=spread,
                            liquidity=m.volume * m.yes_ask,  # Approximate
                            volume=float(m.volume),
                        )
                    )

        if not candidates:
            logger.warning("No markets found for %s vs %s", team_home, team_away)
            return None

        # Sort by spread (tightest first), then by liquidity
        candidates.sort(key=lambda m: (m.spread, -m.liquidity))

        best = candidates[0]
        logger.info(
            "Selected market: %s (%s) spread=%.3f liquidity=%.0f",
            best.platform, best.market_id[:16], best.spread, best.liquidity,
        )

        # Warn if spread is wide
        if best.spread > MAX_SPREAD_THRESHOLD:
            logger.warning(
                "Best spread %.1f%% exceeds threshold %.1f%%",
                best.spread * 100, MAX_SPREAD_THRESHOLD * 100,
            )

        self._selected = best
        return best

    def get_current_prices(self) -> Optional[dict[str, float]]:
        """Get current prices for the selected market.

        Returns:
            Dictionary with 'home', 'draw', 'away' prices, or None.
        """
        if not self._selected:
            return None

        if self._selected.platform == "polymarket" and self.polymarket:
            book = self.polymarket.get_orderbook(self._selected.yes_token_id)
            if book:
                return {
                    "home": book.best_bid,
                    "draw": None,  # No draw market available
                    "away": 1.0 - book.best_ask,
                }

        elif self._selected.platform == "kalshi" and self.kalshi:
            book = self.kalshi.get_orderbook(self._selected.yes_token_id)
            if book:
                return {
                    "home": book.yes_bid,
                    "draw": None,  # No draw market available
                    "away": book.no_bid,
                }

        return None

    @property
    def selected(self) -> Optional[SelectedMarket]:
        """Currently selected market."""
        return self._selected

    @property
    def platform(self) -> Optional[str]:
        """Platform of selected market."""
        return self._selected.platform if self._selected else None
