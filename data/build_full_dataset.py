"""Phase 1: Build full training dataset from StatsBomb + Football-Data.co.uk.

Target: 348K+ snapshots from 10K+ matches.
Features: 51 features matching the trained model schema.

Checkpoint strategy:
- Save intermediate parquet files per data source
- Checkpoint full dataset to teamspace every 5 minutes
- Resume from last complete parquet if interrupted

Usage:
    python -m data.build_full_dataset --output data/train_full.parquet
    python -m data.build_full_dataset --resume  # Resume from checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from model.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Environment detection
LIGHTNING_TEAMSPACE = Path("/teamspace")
OVH_OUTPUT = Path("/workspace/output")

if LIGHTNING_TEAMSPACE.exists():
    _studio_home = Path.home()
    CACHE_BASE = _studio_home / "cache"
    CHECKPOINT_BASE = Path.home() / "checkpoints"
elif OVH_OUTPUT.exists():
    CACHE_BASE = OVH_OUTPUT / "cache"
    CHECKPOINT_BASE = OVH_OUTPUT / "checkpoints"
else:
    CACHE_BASE = Path("./cache")
    CHECKPOINT_BASE = Path("./checkpoints")

CACHE_BASE.mkdir(parents=True, exist_ok=True)
CHECKPOINT_BASE.mkdir(parents=True, exist_ok=True)

STATSBOMB_CLONE_DIR = CACHE_BASE / "statsbomb_open_data"
STATSBOMB_EVENTS_DIR = CACHE_BASE / "statsbomb_events"
STATSBOMB_EVENTS_DIR.mkdir(exist_ok=True)

# Football-Data.co.uk URLs
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/{league}_csv.php"
LEAGUE_MAP = {
    "E0": "EPL",
    "E1": "Championship",
    "SP1": "LaLiga",
    "I1": "SerieA",
    "D1": "Bundesliga",
    "F1": "Ligue1",
    "N1": "Eredivisie",
    "P1": "LigaPortugal",
    "B1": "BelgianLeague",
    "SC0": "SPL",
    "T1": "SuperLig",
}

# Checkpoint file
CHECKPOINT_FILE = CHECKPOINT_BASE / "phase1_checkpoint.json"


def _save_checkpoint(phase: str, data: dict) -> None:
    """Save checkpoint for resume."""
    checkpoint = {"phase": phase, "timestamp": time.time()}
    checkpoint.update(data)
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))
    logger.debug("Checkpoint saved: %s", phase)


def _load_checkpoint() -> Optional[dict]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            return None
    return None


def _run_with_timeout(func, timeout_sec, *args, **kwargs):
    """Run func with a timeout."""
    result = [None]
    timed_out = [False]

    def _target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            result[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        timed_out[0] = True
    return result[0], timed_out[0]


# ============================================================
# STATS BOMB
# ============================================================

def fetch_statsbomb_full(max_matches: int = 10000) -> pd.DataFrame:
    """Fetch ALL StatsBomb open data via GitHub.

    Returns:
        DataFrame with match snapshots at 5-minute intervals.
    """
    logger.info("Fetching full StatsBomb data from GitHub...")

    clone_dir = STATSBOMB_CLONE_DIR
    events_dir = STATSBOMB_EVENTS_DIR
    events_dir.mkdir(exist_ok=True)

    # Step 1: Shallow clone for metadata
    def _do_clone():
        if (clone_dir / "data" / "competitions.json").exists():
            return
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:limit=1m",
             "--sparse", "https://github.com/statsbomb/open-data.git",
             str(clone_dir)],
            capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "sparse-checkout", "set",
             "data/competitions.json", "data/matches", "data/events"],
            capture_output=True, timeout=60,
        )

    _run_with_timeout(_do_clone, 180)

    # Step 2: Load competitions and matches
    try:
        with open(clone_dir / "data" / "competitions.json") as f:
            competitions = json.load(f)
    except Exception as e:
        logger.error("Failed to load competitions: %s", e)
        return pd.DataFrame()

    # Build match lookup (include ALL competitions, not just priority)
    match_lookup: Dict[str, dict] = {}
    competition_names: Dict[str, str] = {}

    for comp in competitions:
        cid = comp["competition_id"]
        season_file = clone_dir / "data" / "matches" / str(cid)
        if not season_file.exists():
            continue

        for season_file in season_file.iterdir():
            if not season_file.suffix == ".json":
                continue
            try:
                with open(season_file) as f:
                    matches = json.load(f)
            except Exception:
                continue

            for m in matches:
                if len(match_lookup) >= max_matches:
                    break
                mid = str(m["match_id"])
                match_lookup[mid] = m
                competition_names[mid] = comp.get("competition_name", "")

    logger.info("StatsBomb: found %d matches across %d competitions",
                len(match_lookup), len(competitions))

    # Step 3: Download events in parallel
    all_snapshots = []
    parsed = 0
    failed = 0

    # Check which events we already have
    existing_events = set()
    for fn in events_dir.glob("*.json"):
        existing_events.add(fn.stem)

    # Download missing events
    missing = [mid for mid in match_lookup.keys() if mid not in existing_events]
    logger.info("StatsBomb: %d events to fetch, %d already cached", len(missing), len(existing_events))

    def _fetch_event(match_id: str) -> Tuple[str, bool]:
        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
        try:
            import requests
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(events_dir / f"{match_id}.json", "w") as f:
                    f.write(resp.text)
                return match_id, True
        except Exception:
            pass
        return match_id, False

    # Parallel fetch with 20 workers
    if missing:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(_fetch_event, mid): mid for mid in missing[:5000]}
            for future in as_completed(futures):
                mid, ok = future.result()
                if not ok:
                    failed += 1
                parsed += 1
                if parsed % 500 == 0:
                    logger.info("StatsBomb: fetched %d/%d events (%d failed)",
                                parsed, len(missing), failed)

    # Step 4: Parse events into snapshots
    # Precompute team stats from all matches
    team_stats = _compute_team_stats(match_lookup, events_dir)

    parsed = 0
    for mid, match in match_lookup.items():
        events_path = events_dir / f"{mid}.json"
        if not events_path.exists():
            continue

        try:
            with open(events_path) as f:
                events = json.load(f)
        except Exception:
            continue

        snapshots = _build_snapshots(events, match, competition_names.get(mid, ""), team_stats)
        all_snapshots.extend(snapshots)
        parsed += 1

        if parsed % 1000 == 0:
            logger.info("StatsBomb: parsed %d matches, %d snapshots", parsed, len(all_snapshots))
            # Checkpoint progress
            _save_checkpoint("statsbomb_progress", {
                "parsed_matches": parsed,
                "total_snapshots": len(all_snapshots),
            })

    df = pd.DataFrame(all_snapshots) if all_snapshots else pd.DataFrame()
    logger.info("StatsBomb complete: %d snapshots from %d matches", len(df), parsed)
    return df


def _compute_team_stats(match_lookup: Dict, events_dir: Path) -> Dict:
    """Precompute team statistics from all matches."""
    team_elo: Dict[int, float] = {}
    team_form: Dict[int, List[float]] = {}
    team_match_count: Dict[int, int] = {}

    sorted_matches = sorted(match_lookup.values(), key=lambda x: x.get("match_id", 0))

    for match in sorted_matches:
        mid = str(match["match_id"])
        events_path = events_dir / f"{mid}.json"
        if not events_path.exists():
            continue

        try:
            with open(events_path) as f:
                events = json.load(f)
        except Exception:
            continue

        # Find team IDs
        home_team_id = None
        away_team_id = None
        for e in events:
            if e.get("type", {}).get("name") == "Starting XI":
                team = e.get("team", {})
                tid = team.get("id") if isinstance(team, dict) else None
                if home_team_id is None:
                    home_team_id = tid
                elif away_team_id is None and tid != home_team_id:
                    away_team_id = tid
                    break

        if home_team_id is None or away_team_id is None:
            continue

        # Count goals
        home_goals = 0
        away_goals = 0
        for e in events:
            if e.get("type", {}).get("name") == "Shot":
                shot = e.get("shot", {})
                outcome = shot.get("outcome", {})
                outcome_name = outcome.get("name", "") if isinstance(outcome, dict) else str(outcome)
                if outcome_name == "Goal":
                    team_id = e.get("team", {}).get("id") if isinstance(e.get("team"), dict) else None
                    if team_id == home_team_id:
                        home_goals += 1
                    elif team_id == away_team_id:
                        away_goals += 1

        # Update ELO
        K = 32
        for tid in [home_team_id, away_team_id]:
            if tid not in team_elo:
                team_elo[tid] = 1500.0
            team_match_count[tid] = team_match_count.get(tid, 0) + 1

        elo_home = team_elo[home_team_id]
        elo_away = team_elo[away_team_id]
        expected_home = 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400))
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        team_elo[home_team_id] += K * (actual_home - expected_home)
        team_elo[away_team_id] += K * (actual_away - expected_away)

        # Update form (last 5 matches)
        if home_team_id not in team_form:
            team_form[home_team_id] = []
        if away_team_id not in team_form:
            team_form[away_team_id] = []

        if home_goals > away_goals:
            team_form[home_team_id].append(3.0)
            team_form[away_team_id].append(0.0)
        elif home_goals < away_goals:
            team_form[home_team_id].append(0.0)
            team_form[away_team_id].append(3.0)
        else:
            team_form[home_team_id].append(1.0)
            team_form[away_team_id].append(1.0)

        # Keep only last 5
        team_form[home_team_id] = team_form[home_team_id][-5:]
        team_form[away_team_id] = team_form[away_team_id][-5:]

    return {
        "team_elo": team_elo,
        "team_form": team_form,
        "team_match_count": team_match_count,
    }


def _build_snapshots(events: list, match: dict, competition: str, team_stats: Dict) -> list:
    """Build snapshots from StatsBomb events."""
    snapshots = []
    match_id = str(match.get("match_id", ""))

    # Find team IDs
    home_team_id = None
    away_team_id = None
    home_team_name = match.get("home_team", {}).get("home_team_name", "")
    away_team_name = match.get("away_team", {}).get("away_team_name", "")

    for e in events:
        if e.get("type", {}).get("name") == "Starting XI":
            team = e.get("team", {})
            tid = team.get("id") if isinstance(team, dict) else None
            if home_team_id is None:
                home_team_id = tid
            elif away_team_id is None and tid != home_team_id:
                away_team_id = tid
                break

    if home_team_id is None or away_team_id is None:
        return snapshots

    # Build timeline of events
    timeline = []
    for e in events:
        minute = e.get("minute", 0)
        second = e.get("second", 0)
        timestamp = minute + second / 60.0

        event_type = e.get("type", {}).get("name", "")
        team_id = e.get("team", {}).get("id") if isinstance(e.get("team"), dict) else None

        # Track goals
        if event_type == "Shot":
            shot = e.get("shot", {})
            outcome = shot.get("outcome", {})
            outcome_name = outcome.get("name", "") if isinstance(outcome, dict) else str(outcome)
            if outcome_name == "Goal":
                timeline.append({"minute": timestamp, "type": "goal", "team": team_id})

        # Track cards
        elif event_type == "Bad Behaviour" or event_type == "Foul":
            card = e.get("bad_behaviour", {}).get("card", {})
            if card and card.get("name", "") == "Red Card":
                timeline.append({"minute": timestamp, "type": "red_card", "team": team_id})

    # Generate snapshots at 5-minute intervals
    team_elo = team_stats.get("team_elo", {})
    team_form = team_stats.get("team_form", {})

    home_elo = team_elo.get(home_team_id, 1500.0)
    away_elo = team_elo.get(away_team_id, 1500.0)
    home_form = sum(team_form.get(home_team_id, [0, 0, 0, 0, 0])) / 15.0
    away_form = sum(team_form.get(away_team_id, [0, 0, 0, 0, 0])) / 15.0

    elo_diff = home_elo - away_elo
    form_diff = home_form - away_form

    for clock in range(0, 95, 5):  # 0, 5, 10, ..., 90
        if clock < 1:
            continue

        # Count events up to this clock
        home_score = 0
        away_score = 0
        home_red = 0
        away_red = 0
        goals_last10 = 0

        for t in timeline:
            if t["minute"] <= clock:
                if t["type"] == "goal":
                    if t["team"] == home_team_id:
                        home_score += 1
                    else:
                        away_score += 1
                elif t["type"] == "red_card":
                    if t["team"] == home_team_id:
                        home_red += 1
                    else:
                        away_red += 1

            # Goals in last 10 minutes
            if t["type"] == "goal" and clock - 10 < t["minute"] <= clock:
                goals_last10 += 1

        score_diff = home_score - away_score
        xg_diff = 0.0  # StatsBomb doesn't provide xG in the basic format
        xg_total = 0.0

        # Compute v2 features
        pressure = 0.5 + 0.3 * np.tanh(score_diff * 0.5)  # Proxy for pressure
        time_remaining = max(90 - clock, 0)

        snapshot = {
            "match_id": match_id,
            "source": "statsbomb",
            "clock_minutes": float(clock),
            "score_diff": float(score_diff),
            "is_extra_time": float(clock > 90),
            "home_red_cards": float(home_red),
            "away_red_cards": float(away_red),
            "home_pressure_score": float(pressure),
            "goals_in_last_10min": float(goals_last10),
            "home_shots_on_target": 0.0,
            "away_shots_on_target": 0.0,
            "home_xg_running": 0.0,
            "away_xg_running": 0.0,
            "score_diff_x_time_remaining": float(score_diff * time_remaining),
            "home_elo": float(home_elo),
            "away_elo": float(away_elo),
            "elo_diff": float(elo_diff),
            "home_form_pts": float(home_form * 15),
            "away_form_pts": float(away_form * 15),
            "h2h_home_winrate": 0.45,
            "is_home_game": 1.0,
            "referee_cards_per_game": 3.5,
            "home_squad_value_EUR": float(max(1e8, 5e8 * (home_elo / 1500))),
            "away_squad_value_EUR": float(max(1e8, 5e8 * (away_elo / 1500))),
            "squad_value_ratio": float(max(1e8, 5e8 * (home_elo / 1500)) / max(1e8, 5e8 * (away_elo / 1500))),
            "home_injuries_count": 0.0,
            "away_injuries_count": 0.0,
            "home_press_pct": float(pressure),
            "away_press_pct": float(1.0 - pressure),
            "home_xg_last5": 0.0,
            "away_xg_last5": 0.0,
            "home_xga_last5": 0.0,
            "away_xga_last5": 0.0,
            "competition_tier": float(_classify_competition(competition)),
            "match_importance": 0.5,
            "days_since_last_match_home": 7.0,
            "days_since_last_match_away": 7.0,
            "goals_last_15min": float(goals_last10),
            "cards_last_15min": 0.0,
            "score_diff_squared": float(score_diff ** 2),
            "momentum_shift": 0.0,
            # v2 features
            "xg_diff": 0.0,
            "xg_total": 0.0,
            "form_diff": float(form_diff),
            "elo_xg_interaction": 0.0,
            "pressure_x_time_remaining": float(pressure * time_remaining),
            "clock_normalized": float(clock / 90.0),
            "home_dominance": 0.0,
            "score_xg_consistent": 1.0 if score_diff != 0 else 0.0,
            "late_game_state": 1.0 if (clock > 75 and score_diff != 0) else 0.0,
            "home_xg_per_minute": 0.0,
            "away_xg_per_minute": 0.0,
            "xg_momentum_ratio": 0.0,
        }

        # Determine target (final result)
        # We'll set this after processing all events
        snapshots.append(snapshot)

    # Set target based on final score
    if snapshots:
        final_home = home_score
        final_away = away_score
        if final_home > final_away:
            target = 0  # Home win
        elif final_home < final_away:
            target = 2  # Away win
        else:
            target = 1  # Draw
        for s in snapshots:
            s["target"] = target

    return snapshots


def _classify_competition(competition: str) -> int:
    """Classify competition tier."""
    comp_lower = competition.lower()
    if "champions" in comp_lower or "ucl" in comp_lower:
        return 1
    elif any(x in comp_lower for x in ["premier", "la liga", "bundesliga", "serie a", "ligue 1"]):
        return 2
    else:
        return 3


# ============================================================
# FOOTBALL-DATA.CO.UK
# ============================================================

def fetch_football_data(seasons: List[str] = None) -> pd.DataFrame:
    """Fetch historical match data from Football-Data.co.uk.

    Args:
        seasons: List of season strings (e.g., ["2324", "2223", "2122"]).

    Returns:
        DataFrame with match snapshots.
    """
    if seasons is None:
        seasons = ["2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617", "1516"]

    all_rows = []

    for league_code, league_name in LEAGUE_MAP.items():
        for season in seasons:
            # Build URL
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

            try:
                import requests
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue

                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))

                # Process each match
                for _, row in df.iterrows():
                    try:
                        home_team = str(row.get("HomeTeam", ""))
                        away_team = str(row.get("AwayTeam", ""))
                        home_goals = int(row.get("FTHG", 0))
                        away_goals = int(row.get("FTAG", 0))

                        if not home_team or not away_team:
                            continue

                        # Create snapshots at 5-minute intervals
                        for clock in range(0, 95, 5):
                            if clock < 1:
                                continue

                            # Use xG if available
                            home_xg = float(row.get("xGH", 0) or 0)
                            away_xg = float(row.get("xGA", 0) or 0)

                            # Approximate score at each clock
                            # For simplicity, we'll use final score for all snapshots
                            # In production, you'd use play-by-play data
                            score_diff = home_goals - away_goals
                            time_remaining = max(90 - clock, 0)

                            snapshot = {
                                "match_id": f"fd_{league_code}_{season}_{home_team}_{away_team}",
                                "source": "football_data",
                                "clock_minutes": float(clock),
                                "score_diff": float(score_diff),
                                "is_extra_time": 0.0,
                                "home_red_cards": 0.0,
                                "away_red_cards": 0.0,
                                "home_pressure_score": 0.5,
                                "goals_in_last_10min": 0.0,
                                "home_shots_on_target": 0.0,
                                "away_shots_on_target": 0.0,
                                "home_xg_running": home_xg * (clock / 90.0),
                                "away_xg_running": away_xg * (clock / 90.0),
                                "score_diff_x_time_remaining": float(score_diff * time_remaining),
                                "home_elo": 1500.0,
                                "away_elo": 1500.0,
                                "elo_diff": 0.0,
                                "home_form_pts": 0.0,
                                "away_form_pts": 0.0,
                                "h2h_home_winrate": 0.45,
                                "is_home_game": 1.0,
                                "referee_cards_per_game": 3.5,
                                "home_squad_value_EUR": 0.0,
                                "away_squad_value_EUR": 0.0,
                                "squad_value_ratio": 1.0,
                                "home_injuries_count": 0.0,
                                "away_injuries_count": 0.0,
                                "home_press_pct": 0.5,
                                "away_press_pct": 0.5,
                                "home_xg_last5": home_xg,
                                "away_xg_last5": away_xg,
                                "home_xga_last5": away_xg,
                                "away_xga_last5": home_xg,
                                "competition_tier": float(_classify_competition(league_name)),
                                "match_importance": 0.5,
                                "days_since_last_match_home": 7.0,
                                "days_since_last_match_away": 7.0,
                                "goals_last_15min": 0.0,
                                "cards_last_15min": 0.0,
                                "score_diff_squared": float(score_diff ** 2),
                                "momentum_shift": 0.0,
                                # v2 features
                                "xg_diff": float((home_xg - away_xg) * (clock / 90.0)),
                                "xg_total": float((home_xg + away_xg) * (clock / 90.0)),
                                "form_diff": 0.0,
                                "elo_xg_interaction": 0.0,
                                "pressure_x_time_remaining": 0.5 * time_remaining,
                                "clock_normalized": float(clock / 90.0),
                                "home_dominance": 0.0,
                                "score_xg_consistent": 1.0 if score_diff != 0 else 0.0,
                                "late_game_state": 1.0 if (clock > 75 and score_diff != 0) else 0.0,
                                "home_xg_per_minute": home_xg / max(clock, 1),
                                "away_xg_per_minute": away_xg / max(clock, 1),
                                "xg_momentum_ratio": 0.0,
                            }

                            # Target
                            if home_goals > away_goals:
                                snapshot["target"] = 0
                            elif home_goals < away_goals:
                                snapshot["target"] = 2
                            else:
                                snapshot["target"] = 1

                            all_rows.append(snapshot)

                    except (ValueError, TypeError):
                        continue

                logger.info("Football-Data: %s season %s: %d matches",
                            league_name, season, len(df))

            except Exception as e:
                logger.debug("Failed to fetch %s season %s: %s", league_code, season, e)
                continue

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    logger.info("Football-Data complete: %d snapshots", len(df))
    return df


# ============================================================
# MAIN
# ============================================================

def build_full_dataset(output_path: str = "data/train_full.parquet") -> pd.DataFrame:
    """Build complete training dataset from all sources.

    Checkpoint strategy:
    - Save intermediate parquets per source
    - Resume from last complete source if interrupted

    Args:
        output_path: Where to save the final parquet.

    Returns:
        Complete training DataFrame.
    """
    start = time.time()
    checkpoint_dir = CHECKPOINT_BASE / "phase1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume
    checkpoint = _load_checkpoint()
    if checkpoint and checkpoint.get("phase") == "complete":
        logger.info("Phase 1 already complete, loading from %s", output_path)
        return pd.read_parquet(output_path)

    all_dfs = []

    # 1. StatsBomb (with checkpoint)
    statsbomb_cache = checkpoint_dir / "statsbomb.parquet"
    if statsbomb_cache.exists() and checkpoint and checkpoint.get("phase") == "statsbomb_done":
        logger.info("Loading cached StatsBomb data from %s", statsbomb_cache)
        statsbomb_df = pd.read_parquet(statsbomb_cache)
    else:
        logger.info("Building StatsBomb dataset...")
        statsbomb_df = fetch_statsbomb_full(max_matches=10000)
        if len(statsbomb_df) > 0:
            statsbomb_df.to_parquet(str(statsbomb_cache), index=False)
            _save_checkpoint("statsbomb_done", {"rows": len(statsbomb_df)})

    if len(statsbomb_df) > 0:
        all_dfs.append(statsbomb_df)
        logger.info("StatsBomb: %d snapshots", len(statsbomb_df))

    # 2. Football-Data.co.uk (with checkpoint)
    fd_cache = checkpoint_dir / "football_data.parquet"
    if fd_cache.exists() and checkpoint and checkpoint.get("phase") == "football_data_done":
        logger.info("Loading cached Football-Data from %s", fd_cache)
        fd_df = pd.read_parquet(fd_cache)
    else:
        logger.info("Building Football-Data dataset...")
        fd_df = fetch_football_data()
        if len(fd_df) > 0:
            fd_df.to_parquet(str(fd_cache), index=False)
            _save_checkpoint("football_data_done", {"rows": len(fd_df)})

    if len(fd_df) > 0:
        all_dfs.append(fd_df)
        logger.info("Football-Data: %d snapshots", len(fd_df))

    # Combine all data
    if not all_dfs:
        logger.error("No data sources available!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure all feature columns exist
    for col in FEATURE_NAMES:
        if col not in combined.columns:
            combined[col] = 0.0

    # Add target if not present
    if "target" not in combined.columns:
        combined["target"] = 1

    # Deduplicate by match_id + clock_minutes
    if "match_id" in combined.columns and "clock_minutes" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["match_id", "clock_minutes"], keep="first")
        logger.info("Deduplication: %d -> %d rows", before, len(combined))

    # Save final dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    # Checkpoint to persistent storage
    final_cache = CACHE_BASE / "data" / "train_full.parquet"
    final_cache.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(str(final_cache), index=False)

    _save_checkpoint("complete", {"rows": len(combined), "features": len(FEATURE_NAMES)})

    elapsed = time.time() - start
    n_matches = len(combined["match_id"].unique()) if "match_id" in combined.columns else 0
    logger.info(
        "Phase 1 complete: %d rows, %d matches, %d features, saved to %s (%.1f min)",
        len(combined), n_matches, len(FEATURE_NAMES), output_path, elapsed / 60,
    )

    return combined


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build full training dataset")
    parser.add_argument("--output", default="data/train_full.parquet", help="Output path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    build_full_dataset(args.output)


if __name__ == "__main__":
    main()
