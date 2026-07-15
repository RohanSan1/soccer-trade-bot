"""Fetch pre-match statistics for live matches.

Pulls from multiple free sources:
- Club ELO (clubelo.com) for ELO ratings
- FBref via soccerdata for xG and stats
- Cached locally to avoid repeated API calls
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "soccer-trade-bot"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ELO cache (updated weekly)
ELO_CACHE_FILE = CACHE_DIR / "club_elo.json"
ELO_CACHE_TTL = 7 * 24 * 3600  # 7 days


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


# Team name mapping for clubelo.com (common variations)
TEAM_NAME_MAP = {
    "manchester city": "ManCity",
    "manchester united": "ManUnited",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "west ham": "WestHam",
    "west ham united": "WestHam",
    "wolves": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "aston villa": "AstonVilla",
    "nottingham forest": "NottmForest",
    "real madrid": "RealMadrid",
    "fc barcelona": "Barcelona",
    "atletico madrid": "AtleticoMadrid",
    "bayern munich": "BayernMunich",
    "borussia Dortmund": "BVB",
    "paris saint-germain": "PSG",
    "paris saint germain": "PSG",
    "juventus": "Juventus",
    "ac milan": "ACMilan",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "napoli": "Napoli",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "liverpool": "Liverpool",
    "everton": "Everton",
    "fulham": "Fulham",
    "crystal palace": "CrystalPalace",
    "brentford": "Brentford",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "burnley": "Burnley",
    "luton": "Luton",
    "luton town": "Luton",
    "sheffield united": "SheffieldUtd",
    "lazio": "Lazio",
    "roma": "Roma",
    "as roma": "Roma",
    "fiorentina": "Fiorentina",
    "atalanta": "Atalanta",
    "torino": "Torino",
    "bayer leverkusen": "Leverkusen",
    "bayer 04 leverkusen": "Leverkusen",
    "rasenballsport leipzig": "RBLeipzig",
    "rb leipzig": "RBLeipzig",
    "eintracht frankfurt": "EintrachtFrankfurt",
    "vfl wolfsburg": "Wolfsburg",
    "vfB stuttgart": "Stuttgart",
    "sc freiburg": "Freiburg",
    "tsg 1899 Hoffenheim": "Hoffenheim",
    "olympique lyonnais": "Lyon",
    "olympique de marseille": "Marseille",
    "stade rennais": "Rennes",
    "stade brestois": "Brest",
    "lille": "Lille",
    "stade de reims": "Reims",
    "monaco": "Monaco",
    "as monaco": "Monaco",
    "valencia": "Valencia",
    "villarreal": "Villarreal",
    "real sociedad": "RealSociedad",
    "real betis": "RealBetis",
    "athletic club": "AthleticBilbao",
    "athletic bilbao": "AthleticBilbao",
    "girona": "Girona",
    "girona fc": "Girona",
}


def _normalize_team_name(name: str) -> str:
    """Normalize team name for clubelo.com lookup."""
    name_lower = name.lower().strip()
    return TEAM_NAME_MAP.get(name_lower, name)


class PreMatchFetcher:
    """Fetch pre-match statistics from multiple sources.

    Args:
        use_cache: Whether to cache results locally.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self._cache: Dict[str, MatchPreMatchData] = {}
        self._elo_cache: Dict[str, float] = {}
        self._load_elo_cache()

    def _load_elo_cache(self) -> None:
        """Load ELO cache from disk."""
        if ELO_CACHE_FILE.exists():
            try:
                age = time.time() - ELO_CACHE_FILE.stat().st_mtime
                if age < ELO_CACHE_TTL:
                    with open(ELO_CACHE_FILE) as f:
                        self._elo_cache = json.load(f)
                    logger.debug("Loaded %d ELO entries from cache", len(self._elo_cache))
            except Exception as e:
                logger.debug("Failed to load ELO cache: %s", e)

    def _save_elo_cache(self) -> None:
        """Save ELO cache to disk."""
        try:
            with open(ELO_CACHE_FILE, "w") as f:
                json.dump(self._elo_cache, f)
        except Exception as e:
            logger.debug("Failed to save ELO cache: %s", e)

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

        # Fetch ELO from clubelo.com
        elo = self._fetch_elo(team_name)
        stats.elo = elo

        # Try to fetch xG from FBref
        try:
            xg, xga = self._fetch_fbref_xg(team_name)
            stats.xg_last5 = xg
            stats.xga_last5 = xga
        except Exception:
            pass

        return stats

    def _fetch_elo(self, team_name: str) -> float:
        """Fetch ELO rating from clubelo.com."""
        normalized = _normalize_team_name(team_name)

        # Check cache first
        if normalized in self._elo_cache:
            return self._elo_cache[normalized]

        try:
            url = f"http://clubelo.com/{normalized}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Parse ELO from the page (simple extraction)
                text = response.text
                # Look for the ELO number in the page
                import re
                elo_match = re.search(r'"Elo":\s*(\d+)', text)
                if elo_match:
                    elo = float(elo_match.group(1))
                    self._elo_cache[normalized] = elo
                    self._save_elo_cache()
                    logger.debug("Fetched ELO for %s: %.0f", team_name, elo)
                    return elo
        except Exception as e:
            logger.debug("Failed to fetch ELO for %s: %s", team_name, e)

        # Default ELO
        return 1500.0

    def _fetch_fbref_xg(self, team_name: str) -> tuple[float, float]:
        """Fetch xG stats from FBref via soccerdata."""
        try:
            import soccerdata as sd

            # Determine current season
            from datetime import datetime
            now = datetime.now()
            if now.month >= 8:
                season = f"{now.year}-{now.year + 1}"
            else:
                season = f"{now.year - 1}-{now.year}"

            fbref = sd.FBref(season, "Big 5 European Leagues Combined")

            # Get team stats
            team_stats = fbref.read_team_standard_stats()
            if team_name in team_stats.index:
                row = team_stats.loc[team_name]
                xg = float(row.get("xG", 0))
                xga = float(row.get("xGA", 0))
                return xg, xga

        except ImportError:
            logger.debug("soccerdata not installed")
        except Exception as e:
            logger.debug("Failed to fetch FBref xG for %s: %s", team_name, e)

        return 0.0, 0.0

    def _fetch_h2h(self, team_home: str, team_away: str) -> float:
        """Fetch head-to-head win rate for home team.

        Returns:
            Home team win rate (0.0-1.0).
        """
        # TODO: Implement actual H2H lookup from historical data
        # For now, use ELO difference as a proxy
        home_elo = self._fetch_elo(team_home)
        away_elo = self._fetch_elo(team_away)

        # ELO-based expected win probability
        expected = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        return expected

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
        """Estimate match importance (0.0-1.0).

        Higher for:
        - Champions League knockout stages
        - Top-of-table clashes
        - Derby matches
        """
        importance = 0.5

        # UCL knockout = high importance
        if "champions" in competition.lower() or "ucl" in competition.lower():
            importance = 0.8

        # Top teams = higher importance
        home_elo = self._fetch_elo(team_home)
        away_elo = self._fetch_elo(team_away)
        avg_elo = (home_elo + away_elo) / 2

        if avg_elo > 1700:
            importance = min(1.0, importance + 0.2)
        elif avg_elo > 1600:
            importance = min(1.0, importance + 0.1)

        return importance
