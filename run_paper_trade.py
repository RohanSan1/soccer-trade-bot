"""Paper trading runner for Kalshi demo.

Two modes:
1. Market-only: Poll Kalshi soccer events, log odds, paper trade when edge detected.
2. Vision-based: Full pipeline with livestream (requires STREAM_URL + MATCH_ID).

Usage:
    # Market-only mode (no livestream needed)
    python run_paper_trade.py

    # Vision-based mode
    STREAM_URL=rtmp://... MATCH_ID=xyz python run_paper_trade.py
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from config import load_config, Config
from data.logger import TradeLogger
from market.kalshi_client import KalshiClient, KalshiMarket
from model.predict import WinPredictor
from trading.edge_calculator import EdgeCalculator
from trading.kelly_sizer import KellySizer

logger = logging.getLogger(__name__)

# Paths
SIGNALS_DIR = Path("data/paper_signals")
STATE_FILE = SIGNALS_DIR / "current_state.json"


@dataclass
class ActiveMarket:
    """Tracks a single Kalshi market we're paper trading."""
    ticker: str
    event_ticker: str
    title: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    last_odds: Dict[str, float] = field(default_factory=dict)
    last_signal: Optional[Dict] = None
    last_edge: Optional[Dict] = None
    trades_count: int = 0


class PaperTrader:
    """Paper trading engine for Kalshi demo.

    Modes:
    - market_only: Polls Kalshi for odds, logs signals, places paper trades.
    - vision: Full pipeline with livestream OCR + prediction.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._running = False

        # Components
        self.kalshi: Optional[KalshiClient] = None
        self.predictor: Optional[WinPredictor] = None
        self.edge_calculator: Optional[EdgeCalculator] = None
        self.kelly_sizer: Optional[KellySizer] = None
        self.trade_logger: Optional[TradeLogger] = None

        # State
        self._active_markets: Dict[str, ActiveMarket] = {}
        self._bankroll: float = 1100.0  # Kalshi demo starting balance
        self._cycle_count: int = 0
        self._total_signals: int = 0
        self._total_trades: int = 0
        self._scan_interval: int = 30  # Seconds between event scans

    def initialize(self) -> bool:
        """Initialize all components. Returns True if ready."""
        logger.info("=" * 60)
        logger.info("PAPER TRADER INITIALIZING")
        logger.info("Mode: %s", "DRY RUN" if self.config.dry_run else "LIVE DEMO")
        logger.info("Bankroll: $%.2f", self._bankroll)
        logger.info("=" * 60)

        # Ensure directories exist
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

        # Trade logger
        self.trade_logger = TradeLogger(db_path=self.config.db_path)

        # Kalshi client
        if not self.config.kalshi_api_key:
            logger.error("KALSHI_API_KEY not set")
            return False

        self.kalshi = KalshiClient(
            api_key=self.config.kalshi_api_key,
            private_key_pem=self.config.kalshi_private_key,
            dry_run=self.config.dry_run,
            use_demo=self.config.kalshi_use_demo,
        )

        # Verify auth
        balance = self.kalshi.get_balance()
        if balance is None:
            logger.error("Failed to authenticate with Kalshi demo")
            return False
        self._bankroll = balance
        logger.info("Kalshi demo balance: $%.2f", balance)

        # ML model
        try:
            self.predictor = WinPredictor(model_dir="model")
            logger.info("ML model loaded (stacking ensemble)")
        except Exception as e:
            logger.warning("ML model not available: %s (market-only mode)", e)
            self.predictor = None

        # Trading components
        self.edge_calculator = EdgeCalculator(
            edge_threshold=self.config.edge_threshold,
            confidence_threshold=self.config.confidence_threshold,
        )
        self.kelly_sizer = KellySizer(
            base_kelly=self.config.kelly_fraction,
            max_bet_pct=self.config.max_bet_pct,
            min_bet_usd=self.config.min_bet_usd,
        )

        logger.info("Initialization complete")
        return True

    def run(self) -> None:
        """Main run loop."""
        self._running = True

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Starting paper trading loop (scan every %ds)...", self._scan_interval)
        self._print_status()

        last_scan = 0
        while self._running:
            try:
                now = time.time()

                # Scan for new events periodically
                if now - last_scan >= self._scan_interval:
                    self._scan_events()
                    last_scan = now

                # Update prices for active markets
                self._update_prices()

                # Check for edge and place paper trades
                self._check_edges()

                # Status every 60 cycles (~60s)
                self._cycle_count += 1
                if self._cycle_count % 60 == 0:
                    self._print_status()

                time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.error("Loop error: %s", e, exc_info=True)
                time.sleep(5)

        self._shutdown()

    def _scan_events(self) -> None:
        """Scan Kalshi for open soccer events."""
        logger.debug("Scanning Kalshi for soccer events...")

        # Try different series
        for series in ["KXSOCCER", "KXMLBSOCCER", "KXMLB", "KXPREMIERLEAGUE"]:
            events = self.kalshi.get_game_events("soccer")

            for event in events:
                event_ticker = event.get("event_ticker", "")
                title = event.get("title", "")

                # Get markets for this event
                markets = self.kalshi.get_event_markets(event_ticker)
                if not markets:
                    continue

                # Filter to match-winner markets (home/draw/away or home/away)
                for m in markets:
                    if m.ticker not in self._active_markets:
                        # New market found
                        self._active_markets[m.ticker] = ActiveMarket(
                            ticker=m.ticker,
                            event_ticker=event_ticker,
                            title=m.title,
                            yes_bid=m.yes_bid,
                            yes_ask=m.yes_ask,
                            no_bid=m.no_bid,
                            no_ask=m.no_ask,
                            volume=m.volume,
                        )
                        logger.info(
                            "New market: %s | %s | YES bid=%.2f ask=%.2f | vol=%d",
                            m.ticker, m.title, m.yes_bid, m.yes_ask, m.volume,
                        )

    def _update_prices(self) -> None:
        """Update prices for all active markets."""
        for ticker, market in list(self._active_markets.items()):
            book = self.kalshi.get_orderbook(ticker)
            if book:
                market.yes_bid = book.yes_bid
                market.yes_ask = book.yes_ask
                market.no_bid = book.no_bid
                market.no_ask = book.no_ask

                # Store odds snapshot
                mid = (book.yes_bid + book.yes_ask) / 2 if book.yes_ask > 0 else 0
                market.last_odds = {
                    "yes_mid": mid,
                    "yes_bid": book.yes_bid,
                    "yes_ask": book.yes_ask,
                    "no_bid": book.no_bid,
                    "no_ask": book.no_ask,
                }

    def _check_edges(self) -> None:
        """Check for trading edges and place paper trades."""
        if not self.predictor or not self.edge_calculator or not self.kelly_sizer:
            return  # No ML model, can't check edges

        for ticker, market in self._active_markets.items():
            odds = market.last_odds
            if not odds or odds["yes_ask"] <= 0:
                continue

            # Get model prediction
            # For market-only mode, we don't have game state.
            # Use market-implied probabilities as baseline and look for mispricings.
            # In a real scenario, vision pipeline would provide game state.

            # For now, log the odds for analysis
            self._log_signal(market, odds)

    def _log_signal(self, market: ActiveMarket, odds: Dict) -> None:
        """Log a trading signal for later analysis."""
        signal_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": market.ticker,
            "event": market.event_ticker,
            "title": market.title,
            "odds": odds,
            "volume": market.volume,
            "bankroll": self._bankroll,
        }

        # Write to current state file
        state = {
            "last_update": signal_data["timestamp"],
            "bankroll": self._bankroll,
            "active_markets": len(self._active_markets),
            "total_signals": self._total_signals,
            "total_trades": self._total_trades,
            "markets": {
                t: {
                    "title": m.title,
                    "odds": m.last_odds,
                    "volume": m.volume,
                }
                for t, m in self._active_markets.items()
            },
        }

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

        self._total_signals += 1

    def _print_status(self) -> None:
        """Print current status."""
        logger.info(
            "STATUS: bankroll=$%.2f | markets=%d | signals=%d | trades=%d | cycles=%d",
            self._bankroll,
            len(self._active_markets),
            self._total_signals,
            self._total_trades,
            self._cycle_count,
        )

        # Show active markets
        for ticker, market in self._active_markets.items():
            odds = market.last_odds
            if odds:
                logger.info(
                    "  %s: YES bid=%.2f ask=%.2f | vol=%d",
                    ticker,
                    odds.get("yes_bid", 0),
                    odds.get("yes_ask", 0),
                    market.volume,
                )

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self) -> None:
        """Cleanup."""
        logger.info("Shutting down paper trader...")

        # Cancel open orders
        if self.kalshi:
            self.kalshi.cancel_all_orders()

        # Print final summary
        self._print_status()
        logger.info("Paper trading stopped")


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data/paper_trade.log"),
        ],
    )

    config = load_config()

    trader = PaperTrader(config)
    if not trader.initialize():
        logger.error("Initialization failed")
        sys.exit(1)

    trader.run()


if __name__ == "__main__":
    main()
