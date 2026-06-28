"""SQLite trade logging.

Single source of truth for all signals, trades, and outcomes.
Tables:
  - signals: every game state snapshot and model prediction
  - trades: every order placed or attempted
  - outcomes: final match result for backtesting
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Generator, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    match_id TEXT NOT NULL,
    clock_minutes INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    ocr_reliable INTEGER,
    model_probs TEXT,          -- JSON: {"home": 0.65, "draw": 0.20, "away": 0.15}
    market_prices TEXT,        -- JSON: {"home": 0.60, "draw": 0.22, "away": 0.18}
    edges TEXT,                -- JSON: {"home": 0.05, "draw": -0.02, "away": -0.03}
    event_label TEXT,
    event_confidence REAL,
    pressure_score REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    match_id TEXT NOT NULL,
    signal_id INTEGER,
    platform TEXT NOT NULL,    -- 'polymarket' or 'kalshi'
    market_id TEXT NOT NULL,
    outcome TEXT NOT NULL,     -- 'home', 'draw', 'away'
    side TEXT NOT NULL,        -- 'buy' or 'sell'
    price REAL NOT NULL,
    size_usd REAL NOT NULL,
    edge REAL,
    kelly_fraction REAL,
    order_id TEXT,
    status TEXT NOT NULL,      -- 'pending', 'filled', 'cancelled', 'error'
    dry_run INTEGER NOT NULL DEFAULT 1,
    error_msg TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL UNIQUE,
    home_team TEXT,
    away_team TEXT,
    final_home_score INTEGER,
    final_away_score INTEGER,
    result TEXT,               -- 'home', 'draw', 'away'
    completed_at REAL
);
"""


class TradeLogger:
    """Thread-safe SQLite logger for signals, trades, and outcomes."""

    def __init__(self, db_path: str = "data/trades.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def log_signal(
        self,
        match_id: str,
        clock_minutes: int,
        home_score: int,
        away_score: int,
        ocr_reliable: bool,
        model_probs: Dict[str, float],
        market_prices: Dict[str, float],
        edges: Dict[str, float],
        event_label: Optional[str] = None,
        event_confidence: Optional[float] = None,
        pressure_score: Optional[float] = None,
    ) -> int:
        """Log a game state snapshot and model prediction. Returns signal ID."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO signals
                   (timestamp, match_id, clock_minutes, home_score, away_score,
                    ocr_reliable, model_probs, market_prices, edges,
                    event_label, event_confidence, pressure_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    match_id,
                    clock_minutes,
                    home_score,
                    away_score,
                    int(ocr_reliable),
                    json.dumps(model_probs),
                    json.dumps(market_prices),
                    json.dumps(edges),
                    event_label,
                    event_confidence,
                    pressure_score,
                ),
            )
            return cur.lastrowid or 0

    def log_trade(
        self,
        match_id: str,
        signal_id: Optional[int],
        platform: str,
        market_id: str,
        outcome: str,
        side: str,
        price: float,
        size_usd: float,
        edge: float,
        kelly_fraction: float,
        order_id: Optional[str] = None,
        status: str = "pending",
        dry_run: bool = True,
        error_msg: Optional[str] = None,
    ) -> int:
        """Log a trade attempt. Returns trade ID."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO trades
                   (timestamp, match_id, signal_id, platform, market_id,
                    outcome, side, price, size_usd, edge, kelly_fraction,
                    order_id, status, dry_run, error_msg)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    match_id,
                    signal_id,
                    platform,
                    market_id,
                    outcome,
                    side,
                    price,
                    size_usd,
                    edge,
                    kelly_fraction,
                    order_id,
                    status,
                    int(dry_run),
                    error_msg,
                ),
            )
            return cur.lastrowid or 0

    def update_trade_status(
        self,
        trade_id: int,
        status: str,
        order_id: Optional[str] = None,
        error_msg: Optional[str] = None,
    ) -> None:
        """Update trade status after execution attempt."""
        with self._cursor() as cur:
            cur.execute(
                """UPDATE trades SET status=?, order_id=COALESCE(?, order_id),
                   error_msg=COALESCE(?, error_msg) WHERE id=?""",
                (status, order_id, error_msg, trade_id),
            )

    def log_outcome(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        final_home_score: int,
        final_away_score: int,
    ) -> None:
        """Log final match outcome for backtesting."""
        if final_home_score > final_away_score:
            result = "home"
        elif final_home_score < final_away_score:
            result = "away"
        else:
            result = "draw"
        with self._cursor() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO outcomes
                   (match_id, home_team, away_team, final_home_score,
                    final_away_score, result, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_id,
                    home_team,
                    away_team,
                    final_home_score,
                    final_away_score,
                    result,
                    time.time(),
                ),
            )

    def get_signals_for_match(self, match_id: str) -> list[Dict[str, Any]]:
        """Retrieve all signals for a given match."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM signals WHERE match_id=? ORDER BY timestamp",
                (match_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_trades_for_match(self, match_id: str) -> list[Dict[str, Any]]:
        """Retrieve all trades for a given match."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades WHERE match_id=? ORDER BY timestamp",
                (match_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_recent_errors(self, window_seconds: int = 60) -> int:
        """Count API errors in the last N seconds."""
        cutoff = time.time() - window_seconds
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM trades WHERE status='error' AND timestamp > ?",
                (cutoff,),
            )
            return cur.fetchone()[0]

    def get_total_pnl(self, match_id: Optional[str] = None) -> float:
        """Calculate total P&L from filled trades (simplified)."""
        query = "SELECT SUM(size_usd * (price - 0.5) * 2) FROM trades WHERE status='filled'"
        params: tuple = ()
        if match_id:
            query += " AND match_id=?"
            params = (match_id,)
        with self._cursor() as cur:
            cur.execute(query, params)
            result = cur.fetchone()[0]
            return float(result) if result else 0.0

    def export_match_data(self, match_id: str) -> Dict[str, Any]:
        """Export all data for a match as a dictionary."""
        return {
            "signals": self.get_signals_for_match(match_id),
            "trades": self.get_trades_for_match(match_id),
            "outcome": self._get_outcome(match_id),
        }

    def _get_outcome(self, match_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM outcomes WHERE match_id=?", (match_id,))
            row = cur.fetchone()
            return dict(row) if row else None
