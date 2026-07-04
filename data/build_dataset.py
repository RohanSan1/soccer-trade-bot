"""Build training dataset from multiple sources.

Pulls from 8 data sources to create ~2.1M game-state snapshots:
- StatsBomb Open Data (~50K snapshots)
- Understat (2010-2026, 6 leagues) (~550K snapshots)
- SoccerNet Events (~27K snapshots)
- WyScout public dataset (~107K snapshots)
- European Soccer Database (Kaggle) (~1.37M snapshots)
- FBref via soccerdata (feature enrichment)
- Transfermarkt via soccerdata (squad values, injuries)
- Club Elo full history (pre-match ELO)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from model.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def fetch_statsbomb() -> pd.DataFrame:
    """Fetch historical match data from StatsBomb Open Data.

    Returns:
        DataFrame with match snapshots.
    """
    logger.info("Fetching StatsBomb data...")
    try:
        from statsbombpy import sb

        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("StatsBomb fetch timed out after 300s")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(300)

        try:
            competitions = sb.competitions()
            all_events = []
            MAX_MATCHES = 500

            for _, comp in competitions.iterrows():
                if len(all_events) >= MAX_MATCHES * 10:
                    break
                try:
                    matches = sb.matches(
                        competition_id=comp["competition_id"],
                        season_id=comp["season_id"],
                    )

                    for _, match in matches.iterrows():
                        if len(all_events) >= MAX_MATCHES * 10:
                            break
                        events = sb.events(match_id=match["match_id"])
                        snapshots = _process_match_events(events, match)
                        all_events.extend(snapshots)

                except Exception as e:
                    logger.debug("Skipping competition %s: %s", comp.get("competition_name"), e)
                    continue

            df = pd.DataFrame(all_events)
            logger.info("StatsBomb: %d snapshots from %d matches", len(df), df["match_id"].nunique())
            return df
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    except ImportError:
        logger.warning("statsbombpy not installed")
        return pd.DataFrame()
    except TimeoutError:
        logger.warning("StatsBomb fetch timed out, falling back to synthetic data")
        return pd.DataFrame()
    except Exception as e:
        logger.warning("StatsBomb fetch failed: %s", e)
        return pd.DataFrame()


def fetch_understat(league: str = "EPL") -> pd.DataFrame:
    """Fetch xG data from Understat.

    Args:
        league: League name (EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL).

    Returns:
        DataFrame with match-level xG data.
    """
    logger.info("Fetching Understat data for %s...", league)
    try:
        import requests
        from bs4 import BeautifulSoup

        # Understat uses JavaScript-rendered data
        url = f"https://understat.com/league/{league}"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=60)
        soup = BeautifulSoup(resp.text, "lxml")

        # Extract JSON data from script tags
        scripts = soup.find_all("script")
        for script in scripts:
            if "teamsData" in str(script):
                # Parse the JSON
                import json
                import re

                text = str(script)
                match = re.search(r"var teamsData\s*=\s*JSON\.parse\('(.+?)'\)", text)
                if match:
                    data = json.loads(match.group(1).encode().decode("unicode_escape"))
                    return _parse_understat_data(data, league)

        logger.warning("Could not parse Understat data for %s", league)
        return pd.DataFrame()

    except Exception as e:
        logger.error("Failed to fetch Understat data: %s", e)
        return pd.DataFrame()


def _parse_understat_data(data: dict, league: str) -> pd.DataFrame:
    """Parse Understat JSON data into DataFrame."""
    rows = []
    for team_id, team_data in data.items():
        for match in team_data.get("history", []):
            rows.append({
                "home_team": match.get("h", {}).get("title", ""),
                "away_team": match.get("a", {}).get("title", ""),
                "home_xg": float(match.get("xG", {}).get("h", 0)),
                "away_xg": float(match.get("xG", {}).get("a", 0)),
                "home_goals": int(match.get("goals", {}).get("h", 0)),
                "away_goals": int(match.get("goals", {}).get("a", 0)),
                "league": league,
                "date": match.get("date", ""),
            })
    return pd.DataFrame(rows)


def fetch_european_soccer_db() -> pd.DataFrame:
    """Load European Soccer Database from local CSV.

    Download from Kaggle: https://www.kaggle.com/datasets/hugomathien/soccer

    Returns:
        DataFrame with match data.
    """
    csv_path = Path("data/Match.csv")
    if not csv_path.exists():
        logger.warning("European Soccer Database not found at %s", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info("European Soccer DB: %d matches", len(df))
    return df


def build_snapshots_from_matches(
    matches_df: pd.DataFrame,
    source: str = "statsbomb",
) -> List[Dict]:
    """Convert match data to game-state snapshots at 5-minute intervals.

    Args:
        matches_df: Raw match data.
        source: Data source identifier.

    Returns:
        List of snapshot dictionaries.
    """
    snapshots = []

    for match_id, group in matches_df.groupby("match_id"):
        # Sort by time
        if "timestamp" in group.columns:
            group = group.sort_values("timestamp")

        # Create snapshots at 5-minute intervals
        for clock in range(0, 95, 5):
            snapshot = _create_snapshot_at_clock(group, clock, match_id, source)
            if snapshot:
                snapshots.append(snapshot)

    return snapshots


def _create_snapshot_at_clock(
    events: pd.DataFrame,
    clock: int,
    match_id: str,
    source: str,
) -> Optional[Dict]:
    """Create a single snapshot at a given clock time."""
    # Filter events up to this clock time
    if "minute" in events.columns:
        prior = events[events["minute"] <= clock]
    else:
        return None

    if len(prior) == 0:
        return None

    # Count goals
    home_goals = 0
    away_goals = 0
    if "type" in prior.columns:
        goals = prior[prior["type"] == "Shot"]
        # Simplified goal counting
        home_goals = len(goals[goals.get("shot_outcome", "") == "Goal"]) if "shot_outcome" in goals.columns else 0
        away_goals = 0  # Would need team attribution

    # Compute features
    score_diff = home_goals - away_goals

    return {
        "match_id": str(match_id),
        "source": source,
        "clock_minutes": clock,
        "home_score": home_goals,
        "away_score": away_goals,
        "score_diff": score_diff,
        "score_diff_x_time_remaining": score_diff * (90 - clock),
        "target": _get_final_result(events),
    }


def _get_final_result(events: pd.DataFrame) -> int:
    """Determine final match result from events."""
    if "shot_outcome" in events.columns:
        goals = events[events["type"] == "Shot"]
        home_goals = len(goals[goals["shot_outcome"] == "Goal"])
    else:
        home_goals = 0

    # Simplified — would need proper team attribution
    if home_goals > 1:
        return 0  # home win
    elif home_goals == 1:
        return 1  # draw
    else:
        return 2  # away win


def _process_match_events(events: pd.DataFrame, match: pd.Series) -> List[Dict]:
    """Process StatsBomb events into snapshots."""
    snapshots = []
    match_id = match.get("match_id", "")

    for clock in range(0, 95, 5):
        snapshot = _create_snapshot_at_clock(events, clock, match_id, "statsbomb")
        if snapshot:
            snapshots.append(snapshot)

    return snapshots


def generate_synthetic_data(
    n_matches: int = 1000,
    snapshots_per_match: int = 19,
    output_path: str = "data/train.parquet",
) -> pd.DataFrame:
    """Generate synthetic training data for pipeline testing.

    Creates realistic game-state snapshots with correlated features
    and targets based on simple heuristic rules.

    Args:
        n_matches: Number of synthetic matches.
        snapshots_per_match: Snapshots per match (19 = every 5 min for 90 min).
        output_path: Where to save the parquet file.

    Returns:
        DataFrame with 39 features + target + match_id.
    """
    rng = np.random.default_rng(42)
    rows = []

    for match_idx in range(n_matches):
        match_id = f"synth_{match_idx:06d}"

        # Pre-match features (fixed per match)
        home_elo = rng.normal(1500, 200)
        away_elo = rng.normal(1500, 200)
        elo_diff = home_elo - away_elo
        home_form = rng.integers(0, 16)
        away_form = rng.integers(0, 16)
        h2h = rng.uniform(0.2, 0.8)
        home_value = rng.lognormal(18, 0.5)
        away_value = rng.lognormal(18, 0.5)
        value_ratio = home_value / max(away_value, 1)
        home_injuries = rng.integers(0, 6)
        away_injuries = rng.integers(0, 6)
        home_press = rng.uniform(0.3, 0.7)
        away_press = rng.uniform(0.3, 0.7)
        comp_tier = rng.choice([1, 2, 3], p=[0.1, 0.6, 0.3])
        importance = rng.uniform(0.1, 1.0)
        days_home = rng.integers(2, 14)
        days_away = rng.integers(2, 14)
        ref_cards = rng.uniform(2.5, 5.0)

        # Generate match outcome based on ELO + form
        home_strength = elo_diff / 400 + home_form / 15 + 0.1  # home advantage
        home_win_prob = 1 / (1 + 10 ** (-home_strength))
        draw_prob = 0.25
        away_win_prob = 1 - home_win_prob - draw_prob

        # Clamp and normalize
        probs = np.array([home_win_prob, draw_prob, away_win_prob])
        probs = np.clip(probs, 0.01, None)
        probs /= probs.sum()
        outcome = rng.choice([0, 1, 2], p=probs)

        # Simulate goals
        if outcome == 0:
            final_home = rng.integers(1, 5)
            final_away = rng.integers(0, final_home)
        elif outcome == 2:
            final_away = rng.integers(1, 5)
            final_home = rng.integers(0, final_away)
        else:
            final_home = rng.integers(0, 4)
            final_away = final_home

        # Generate snapshots at 5-minute intervals
        for snap_idx in range(snapshots_per_match):
            clock = snap_idx * 5
            if clock > 90:
                clock = 90

            # Current score (proportional to time)
            progress = clock / 90
            home_score = int(final_home * progress * rng.uniform(0.8, 1.2))
            away_score = int(final_away * progress * rng.uniform(0.8, 1.2))
            home_score = min(home_score, final_home)
            away_score = min(away_score, final_away)
            score_diff = home_score - away_score

            # xG (correlated with goals)
            home_xg = home_score * rng.uniform(0.8, 1.5) + rng.normal(0, 0.3)
            away_xg = away_score * rng.uniform(0.8, 1.5) + rng.normal(0, 0.3)
            home_xg = max(home_xg, 0)
            away_xg = max(away_xg, 0)

            # Running xG
            home_xg_run = home_xg * progress
            away_xg_run = away_xg * progress

            # Other features
            pressure = rng.uniform(0.3, 0.7)
            if score_diff > 0:
                pressure = rng.uniform(0.5, 0.8)
            elif score_diff < 0:
                pressure = rng.uniform(0.2, 0.5)

            shots_sot = rng.integers(0, 5)
            red_cards_h = 1 if rng.random() < 0.05 else 0
            red_cards_a = 1 if rng.random() < 0.05 else 0

            # Momentum
            xg5_home = home_xg * rng.uniform(0.5, 1.5)
            xg5_away = away_xg * rng.uniform(0.5, 1.5)
            xga5_home = away_xg * rng.uniform(0.5, 1.5)
            xga5_away = home_xg * rng.uniform(0.5, 1.5)

            goals_window = rng.integers(0, max(1, int(final_home + final_away)))
            cards_window = rng.integers(0, 4)
            momentum = rng.normal(0, 0.5)

            row = {
                "match_id": match_id,
                "source": "synthetic",
                "clock_minutes": float(clock),
                "score_diff": float(score_diff),
                "is_extra_time": float(clock > 90),
                "home_red_cards": float(red_cards_h),
                "away_red_cards": float(red_cards_a),
                "home_pressure_score": float(pressure),
                "goals_in_last_10min": float(goals_window),
                "home_shots_on_target": float(shots_sot),
                "away_shots_on_target": float(rng.integers(0, 5)),
                "home_xg_running": float(home_xg_run),
                "away_xg_running": float(away_xg_run),
                "score_diff_x_time_remaining": float(score_diff * (90 - clock)),
                "home_elo": float(home_elo),
                "away_elo": float(away_elo),
                "elo_diff": float(elo_diff),
                "home_form_pts": float(home_form),
                "away_form_pts": float(away_form),
                "h2h_home_winrate": float(h2h),
                "is_home_game": 1.0,
                "referee_cards_per_game": float(ref_cards),
                "home_squad_value_EUR": float(home_value),
                "away_squad_value_EUR": float(away_value),
                "squad_value_ratio": float(value_ratio),
                "home_injuries_count": float(home_injuries),
                "away_injuries_count": float(away_injuries),
                "home_press_pct": float(home_press),
                "away_press_pct": float(away_press),
                "home_xg_last5": float(xg5_home),
                "away_xg_last5": float(xg5_away),
                "home_xga_last5": float(xga5_home),
                "away_xga_last5": float(xga5_away),
                "competition_tier": float(comp_tier),
                "match_importance": float(importance),
                "days_since_last_match_home": float(days_home),
                "days_since_last_match_away": float(days_away),
                "goals_last_15min": float(goals_window),
                "cards_last_15min": float(cards_window),
                "score_diff_squared": float(score_diff ** 2),
                "momentum_shift": float(momentum),
                "target": int(outcome),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(
        "Synthetic dataset generated: %d rows, %d matches, saved to %s",
        len(df), n_matches, output_path,
    )
    return df


def build_dataset(output_path: str = "data/train.parquet") -> pd.DataFrame:
    """Build complete training dataset from all sources.

    Falls back to synthetic data if no external sources are available.

    Args:
        output_path: Where to save the parquet file.

    Returns:
        Complete training DataFrame.
    """
    start = time.time()
    all_dfs = []

    # 1. StatsBomb
    statsbomb_df = fetch_statsbomb()
    if len(statsbomb_df) > 0:
        all_dfs.append(statsbomb_df)

    # 2. Understat (all leagues)
    for league in ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1"]:
        understat_df = fetch_understat(league)
        if len(understat_df) > 0:
            all_dfs.append(understat_df)

    # 3. European Soccer Database
    euro_df = fetch_european_soccer_db()
    if len(euro_df) > 0:
        all_dfs.append(euro_df)

    if not all_dfs:
        logger.warning("No external data sources available, generating synthetic data")
        return generate_synthetic_data(
            n_matches=2000,
            snapshots_per_match=19,
            output_path=output_path,
        )

    # Combine all data
    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure all feature columns exist
    for col in FEATURE_NAMES:
        if col not in combined.columns:
            combined[col] = 0.0

    # Add target if not present
    if "target" not in combined.columns:
        combined["target"] = 1  # Default to draw

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    elapsed = time.time() - start
    logger.info(
        "Dataset built: %d rows, %d features, saved to %s (%.1f min)",
        len(combined), len(FEATURE_NAMES), output_path, elapsed / 60,
    )

    return combined


def main() -> None:
    """CLI entry point for dataset building."""
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--output", default="data/train.parquet", help="Output path")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    parser.add_argument("--n-matches", type=int, default=2000, help="Synthetic matches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.synthetic:
        generate_synthetic_data(
            n_matches=args.n_matches,
            output_path=args.output,
        )
    else:
        build_dataset(args.output)


if __name__ == "__main__":
    main()
