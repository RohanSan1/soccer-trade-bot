"""Fetch pre-match statistics for live matches.

Pulls from FBref (soccerdata) and Transfermarkt:
- ELO ratings
- Form (last 5 matches)
- H2H records
- Squad values
- Injury lists
- Referee statistics
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreMatchStats:
    """Pre-match statistics for a team."""

    team_name: str
    elo: float = 1500.0
    form_points: int = 0  # Points from last 5 matches
    h2h_winrate: float = 0.5
    squad_value_eur: float = 0.0
    injuries_count: int = 0
    press_pct: float = 0.0  # PPDA-derived
    xg_last5: float = 0.0
    xga_last5: float = 0.0
    days_since_last_match: int = 7
    referee_cards_per_game: float = 3.5


@dataclass
class MatchPreMatchData:
    """Combined pre-match data for a match."""

    home: PreMatchStats
    away: PreMatchStats
    competition_tier: int = 2  # 1=UCL, 2=Big5, 3=lower
    match_importance: float = 0.5


class PreMatchFetcher:
    """Fetch pre-match statistics from multiple sources.

    Args:
        use_cache: Whether to cache results locally.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self._cache: Dict[str, MatchPreMatchData] = {}

    def fetch(
        self,
        team_home: str,
        team_away: str,
        competition: str = "",
    ) -> MatchPreMatchData:
        """Fetch pre-match data for a match.

        Args:
            team_home: Home team name.
            team_away: Away team name.
            competition: Competition name (for tier classification).

        Returns:
            MatchPreMatchData with stats for both teams.
        """
        cache_key = f"{team_home}_{team_away}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        home_stats = self._fetch_team_stats(team_home)
        away_stats = self._fetch_team_stats(team_away)

        # H2H
        home_stats.h2h_winrate = self._fetch_h2h(team_home, team_away)

        # Competition tier
        tier = self._classify_competition(competition)

        # Match importance
        importance = self._estimate_importance(team_home, team_away, competition)

        data = MatchPreMatchData(
            home=home_stats,
            away=away_stats,
            competition_tier=tier,
            match_importance=importance,
        )

        if self.use_cache:
            self._cache[cache_key] = data

        return data

    def _fetch_team_stats(self, team_name: str) -> PreMatchStats:
        """Fetch statistics for a single team."""
        stats = PreMatchStats(team_name=team_name)

        try:
            import soccerdata as sd

            # FBref
            fbref = sd.FBref("2023-2024", "Big 5 European Leagues Combined")

            # Get team stats
            try:
                team_stats = fbref.read_team_standard_stats()
                if team_name in team_stats.index:
                    row = team_stats.loc[team_name]
                    stats.xg_last5 = float(row.get("xG", 0))
                    stats.xga_last5 = float(row.get("xGA", 0))
            except Exception:
                pass

        except ImportError:
            logger.debug("soccerdata not installed, using defaults")

        # Club ELO
        try:
            import soccerdata as sd
            club_elo = sd.ClubElo()
            # Would need actual ELO lookup here
        except Exception:
            pass

        return stats

    def _fetch_h2h(self, team_home: str, team_away: str) -> float:
        """Fetch head-to-head win rate for home team."""
        # Would implement H2H lookup from historical data
        return 0.45  # Default

    def _classify_competition(self, competition: str) -> int:
        """Classify competition tier."""
        competition_lower = competition.lower()
        if "champions" in competition_lower or "ucl" in competition_lower:
            return 1
        elif any(x in competition_lower for x in ["premier", "la liga", "bundesliga", "serie a", "ligue 1"]):
            return 2
        else:
            return 3

    def _estimate_importance(
        self, team_home: str, team_away: str, competition: str
    ) -> float:
        """Estimate match importance (0.0-1.0)."""
        # Would implement based on league position, title race, etc.
        return 0.5
