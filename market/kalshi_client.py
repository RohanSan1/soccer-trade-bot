"""Kalshi REST API client.

Handles:
- Market discovery for soccer match winner markets
- RSA-PSS signed request authentication
- Order placement and management
"""
from __future__ import annotations

import base64
import datetime
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils

logger = logging.getLogger(__name__)

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


@dataclass
class KalshiMarket:
    """Kalshi market representation."""

    ticker: str
    title: str
    subtitle: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    open_interest: int
    status: str
   expiration_time: str


@dataclass
class KalshiOrderbook:
    """Kalshi orderbook state."""

    ticker: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    spread: float
    timestamp: float


class KalshiClient:
    """Kalshi REST API client with RSA-PSS authentication.

    Args:
        api_key: Kalshi API key ID.
        private_key_pem: RSA private key in PEM format.
        dry_run: If True, log orders without placing them.
        use_demo: If True, use demo environment.
    """

    def __init__(
        self,
        api_key: str = "",
        private_key_pem: str = "",
        dry_run: bool = True,
        use_demo: bool = False,
    ) -> None:
        self.api_key = api_key
        self.private_key_pem = private_key_pem
        self.dry_run = dry_run
        self.base_url = KALSHI_DEMO_BASE if use_demo else KALSHI_API_BASE
        self._private_key = None
        self._session = requests.Session()

        if private_key_pem:
            try:
                self._private_key = serialization.load_pem_private_key(
                    private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
                    password=None,
                )
            except Exception as e:
                logger.error("Failed to load private key: %s", e)

    def _sign_request(self, method: str, path: str) -> Dict[str, str]:
        """Generate RSA-PSS signed headers for authenticated requests.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API path (e.g., /trade-api/v2/portfolio/orders).

        Returns:
            Dictionary of authentication headers.
        """
        if not self._private_key or not self.api_key:
            return {}

        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))

        # Sign: timestamp + method + path (without query params)
        sign_path = urlparse(path).path
        message = f"{timestamp}{method}{sign_path}".encode()

        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make authenticated request to Kalshi API.

        Args:
            method: HTTP method.
            path: API path.
            params: Query parameters.
            json_data: JSON body for POST/PUT.

        Returns:
            Response JSON or None on error.
        """
        url = f"{self.base_url}{path}"
        headers = self._sign_request(method, path)

        try:
            resp = self._session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            logger.error("Kalshi API error: %s - %s", e.response.status_code, e.response.text[:200])
            return None
        except Exception as e:
            logger.error("Kalshi request failed: %s", e)
            return None

    def search_soccer_markets(
        self, team_home: str, team_away: str
    ) -> List[KalshiMarket]:
        """Search for soccer match winner markets.

        Args:
            team_home: Home team name.
            team_away: Away team name.

        Returns:
            List of matching markets.
        """
        markets = []

        try:
            # Search for open soccer markets
            resp = self._request(
                "GET",
                "/markets",
                params={"limit": 100, "status": "open", "series_ticker": "KXSOCCER"},
            )

            if not resp or "markets" not in resp:
                return markets

            for item in resp["markets"]:
                title = item.get("title", "").lower()
                subtitle = item.get("subtitle", "").lower()

                # Check if both teams are mentioned
                if (
                    team_home.lower() in title
                    and team_away.lower() in title
                ) or (
                    team_home.lower() in subtitle
                    and team_away.lower() in subtitle
                ):
                    market = KalshiMarket(
                        ticker=item.get("ticker", ""),
                        title=item.get("title", ""),
                        subtitle=item.get("subtitle", ""),
                        yes_bid=float(item.get("yes_bid", 0)) / 100,
                        yes_ask=float(item.get("yes_ask", 100)) / 100,
                        no_bid=float(item.get("no_bid", 0)) / 100,
                        no_ask=float(item.get("no_ask", 100)) / 100,
                        volume=item.get("volume", 0),
                        open_interest=item.get("open_interest", 0),
                        status=item.get("status", ""),
                        expiration_time=item.get("expiration_time", ""),
                    )
                    markets.append(market)

        except Exception as e:
            logger.error("Failed to search Kalshi: %s", e)

        return markets

    def get_orderbook(self, ticker: str) -> Optional[KalshiOrderbook]:
        """Get current orderbook for a market.

        Args:
            ticker: Market ticker.

        Returns:
            KalshiOrderbook or None.
        """
        try:
            resp = self._request("GET", f"/markets/{ticker}/orderbook")
            if not resp:
                return None

            orderbook = resp.get("orderbook", {})
            # Kalshi prices are in cents (1-99)
            yes_bid = float(orderbook.get("yes", [{}])[0].get("price", 0)) / 100 if orderbook.get("yes") else 0.0
            yes_ask = float(orderbook.get("yes", [{}])[-1].get("price", 100)) / 100 if orderbook.get("yes") else 1.0

            return KalshiOrderbook(
                ticker=ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=1.0 - yes_ask,
                no_ask=1.0 - yes_bid,
                spread=yes_ask - yes_bid,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error("Failed to get Kalshi orderbook for %s: %s", ticker, e)
            return None

    def place_order(
        self,
        ticker: str,
        side: str,
        yes_price: int,
        count: int,
    ) -> Optional[str]:
        """Place a limit order on Kalshi.

        Args:
            ticker: Market ticker.
            side: 'BUY' or 'SELL'.
            yes_price: Price in cents (1-99).
            count: Number of contracts.

        Returns:
            Order ID if placed, None otherwise.
        """
        if self.dry_run:
            logger.info(
                "[DRY RUN] Kalshi order: %s %s @ %d cents x %d",
                side, ticker, yes_price, count,
            )
            return f"dry_run_{int(time.time())}"

        order_data = {
            "ticker": ticker,
            "action": side.lower(),
            "type": "limit",
            "yes_price": yes_price,
            "count": count,
            "time_in_force": "gtc",
        }

        resp = self._request("POST", "/portfolio/orders", json_data=order_data)
        if resp and "order" in resp:
            order_id = resp["order"].get("order_id", "")
            logger.info("Kalshi order placed: %s", order_id)
            return order_id

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order to cancel.

        Returns:
            True if cancelled successfully.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Cancel Kalshi order: %s", order_id)
            return True

        resp = self._request("DELETE", f"/portfolio/orders/{order_id}")
        return resp is not None

    def get_balance(self) -> Optional[float]:
        """Get current account balance.

        Returns:
            Balance in dollars, or None on error.
        """
        resp = self._request("GET", "/portfolio/balance")
        if resp:
            return float(resp.get("balance", 0)) / 100
        return None
