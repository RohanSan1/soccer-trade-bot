"""Edge calculator for model vs market probability comparison.

Computes the difference between model-predicted probabilities
and market-implied probabilities for each outcome.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    """Edge calculation for a single outcome."""

    outcome: str  # 'home', 'draw', 'away'
    model_prob: float
    market_prob: float
    edge: float  # model_prob - market_prob
    has_edge: bool  # edge > threshold


@dataclass
class EdgeAnalysis:
    """Complete edge analysis for all outcomes."""

    edges: Dict[str, EdgeResult]
    best_edge: Optional[EdgeResult]
    any_tradable: bool


class EdgeCalculator:
    """Calculates betting edge from model vs market probabilities.

    Args:
        edge_threshold: Minimum edge to consider tradable (default 5%).
    """

    def __init__(self, edge_threshold: float = 0.05) -> None:
        self.edge_threshold = edge_threshold

    def calculate(
        self,
        model_probs: Dict[str, float],
        market_prices: Dict[str, float],
    ) -> EdgeAnalysis:
        """Calculate edge for all outcomes.

        Args:
            model_probs: Model predictions {'home': 0.65, 'draw': 0.20, 'away': 0.15}.
            market_prices: Market implied probabilities {'home': 0.60, 'draw': 0.22, 'away': 0.18}.

        Returns:
            EdgeAnalysis with edge for each outcome.
        """
        edges: Dict[str, EdgeResult] = {}

        for outcome in ["home", "draw", "away"]:
            model_p = model_probs.get(outcome, 0.0)
            market_p = market_prices.get(outcome, 0.0)

            edge = model_p - market_p
            has_edge = edge >= self.edge_threshold

            edges[outcome] = EdgeResult(
                outcome=outcome,
                model_prob=model_p,
                market_prob=market_p,
                edge=edge,
                has_edge=has_edge,
            )

            if has_edge:
                logger.info(
                    "Edge found: %s model=%.3f market=%.3f edge=+%.3f",
                    outcome, model_p, market_p, edge,
                )

        # Find best edge
        tradable = [e for e in edges.values() if e.has_edge]
        best = max(tradable, key=lambda e: e.edge) if tradable else None

        return EdgeAnalysis(
            edges=edges,
            best_edge=best,
            any_tradable=len(tradable) > 0,
        )

    def has_significant_edge(
        self,
        model_probs: Dict[str, float],
        market_prices: Dict[str, float],
        min_outcomes: int = 1,
    ) -> bool:
        """Quick check if any outcome has significant edge.

        Args:
            model_probs: Model predictions.
            market_prices: Market prices.
            min_outcomes: Minimum number of outcomes with edge required.

        Returns:
            True if at least min_outcomes have edge > threshold.
        """
        count = 0
        for outcome in ["home", "draw", "away"]:
            model_p = model_probs.get(outcome, 0.0)
            market_p = market_prices.get(outcome, 0.0)
            if model_p - market_p >= self.edge_threshold:
                count += 1
        return count >= min_outcomes
