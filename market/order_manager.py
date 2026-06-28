"""Order manager for placing, tracking, and canceling orders.

Centralizes order execution across Polymarket and Kalshi.
Handles dry run mode, order tracking, and error recovery.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from market.kalshi_client import KalshiClient
from market.polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a placed or pending order."""

    order_id: str
    platform: str
    market_id: str
    outcome: str  # 'home', 'draw', 'away'
    side: str  # 'buy' or 'sell'
    price: float
    size_usd: float
    status: str  # 'pending', 'filled', 'cancelled', 'error'
    timestamp: float = field(default_factory=time.time)
    error_msg: Optional[str] = None


class OrderManager:
    """Manages order execution across multiple platforms.

    Args:
        polymarket: Polymarket client (optional).
        kalshi: Kalshi client (optional).
        min_bet: Minimum bet size in USD.
        max_bet_pct: Maximum bet as percentage of bankroll.
        dry_run: If True, log orders without placing them.
    """

    def __init__(
        self,
        polymarket: Optional[PolymarketClient] = None,
        kalshi: Optional[KalshiClient] = None,
        min_bet: float = 5.0,
        max_bet_pct: float = 0.02,
        dry_run: bool = True,
    ) -> None:
        self.polymarket = polymarket
        self.kalshi = kalshi
        self.min_bet = min_bet
        self.max_bet_pct = max_bet_pct
        self.dry_run = dry_run
        self._open_orders: Dict[str, Order] = {}
        self._filled_orders: List[Order] = []
        self._bankroll: float = 1000.0  # Default, should be updated

    def set_bankroll(self, amount: float) -> None:
        """Update current bankroll for position sizing."""
        self._bankroll = amount

    def place_order(
        self,
        platform: str,
        market_id: str,
        outcome: str,
        side: str,
        price: float,
        size_usd: float,
    ) -> Optional[Order]:
        """Place an order on the specified platform.

        Args:
            platform: 'polymarket' or 'kalshi'.
            market_id: Market identifier.
            outcome: 'home', 'draw', or 'away'.
            side: 'buy' or 'sell'.
            price: Limit price.
            size_usd: Bet size in USD.

        Returns:
            Order object if placed, None if rejected.
        """
        # Validate bet size
        if size_usd < self.min_bet:
            logger.debug("Bet $%.2f below minimum $%.2f, skipping", size_usd, self.min_bet)
            return None

        max_bet = self._bankroll * self.max_bet_pct
        if size_usd > max_bet:
            logger.warning(
                "Bet $%.2f exceeds max $%.2f (%.1f%% of bankroll), capping",
                size_usd, max_bet, self.max_bet_pct * 100,
            )
            size_usd = max_bet

        # Place order
        order_id = None
        error_msg = None

        try:
            if platform == "polymarket" and self.polymarket:
                # Convert USD to contracts (each contract = $1 at target price)
                contracts = size_usd / price if price > 0 else 0
                order_id = self.polymarket.place_limit_order(
                    token_id=market_id,
                    side=side.upper(),
                    price=price,
                    size=contracts,
                )
            elif platform == "kalshi" and self.kalshi:
                # Kalshi: price in cents, count = number of contracts
                cents = int(price * 100)
                count = int(size_usd)  # 1 contract = $1 at resolution
                order_id = self.kalshi.place_order(
                    ticker=market_id,
                    side=side.upper(),
                    yes_price=cents,
                    count=count,
                )
            else:
                error_msg = f"Unknown platform or client not initialized: {platform}"
                logger.error(error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error("Order placement failed: %s", error_msg)

        # Create order record
        status = "pending" if order_id else "error"
        order = Order(
            order_id=order_id or "",
            platform=platform,
            market_id=market_id,
            outcome=outcome,
            side=side,
            price=price,
            size_usd=size_usd,
            status=status,
            error_msg=error_msg,
        )

        if order_id:
            self._open_orders[order_id] = order
            logger.info(
                "Order placed: %s %s %s @ %.3f x $%.2f [%s]",
                platform, side, outcome, price, size_usd,
                "DRY RUN" if self.dry_run else order_id[:8],
            )
        else:
            logger.warning("Order rejected: %s", error_msg)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order to cancel.

        Returns:
            True if cancelled successfully.
        """
        order = self._open_orders.get(order_id)
        if not order:
            logger.warning("Order %s not found in open orders", order_id)
            return False

        success = False
        try:
            if order.platform == "polymarket" and self.polymarket:
                success = self.polymarket.cancel_order(order_id)
            elif order.platform == "kalshi" and self.kalshi:
                success = self.kalshi.cancel_order(order_id)
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)

        if success:
            order.status = "cancelled"
            self._open_orders.pop(order_id, None)
            logger.info("Order cancelled: %s", order_id)

        return success

    def cancel_all_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders cancelled.
        """
        cancelled = 0
        for order_id in list(self._open_orders.keys()):
            if self.cancel_order(order_id):
                cancelled += 1
        return cancelled

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return list(self._open_orders.values())

    def get_total_exposure(self) -> float:
        """Get total USD exposed across all open orders."""
        return sum(o.size_usd for o in self._open_orders.values())

    def mark_filled(self, order_id: str) -> None:
        """Mark an order as filled."""
        order = self._open_orders.pop(order_id, None)
        if order:
            order.status = "filled"
            self._filled_orders.append(order)
            logger.info("Order filled: %s %s @ %.3f", order.platform, order.outcome, order.price)
