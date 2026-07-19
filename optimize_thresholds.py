"""Threshold optimization — vectorized, fast."""
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "score_diff", "clock_minutes", "is_extra_time", "home_red_cards", "away_red_cards",
    "home_pressure_score", "goals_in_last_10min", "home_shots_on_target", "away_shots_on_target",
    "home_xg_running", "away_xg_running", "score_diff_x_time_remaining", "home_elo", "away_elo",
    "elo_diff", "home_form_pts", "away_form_pts", "h2h_home_winrate", "is_home_game",
    "referee_cards_per_game", "home_squad_value_EUR", "away_squad_value_EUR", "squad_value_ratio",
    "home_injuries_count", "away_injuries_count", "home_press_pct", "away_press_pct",
    "home_xg_last5", "away_xg_last5", "home_xga_last5", "away_xga_last5", "competition_tier",
    "match_importance", "days_since_last_match_home", "days_since_last_match_away",
    "goals_last_15min", "cards_last_15min", "score_diff_squared", "momentum_shift",
    "xg_diff", "xg_total", "form_diff", "elo_xg_interaction", "pressure_x_time_remaining",
    "clock_normalized", "home_dominance", "score_xg_consistent", "late_game_state",
    "home_xg_per_minute", "away_xg_per_minute", "xg_momentum_ratio",
]


def engineer_features(df):
    df = df.copy()
    clock = df["clock_minutes"].replace(0, 1)
    xg_diff = df["home_xg_running"] - df["away_xg_running"]
    xg_total = df["home_xg_running"] + df["away_xg_running"]
    df["xg_diff"] = xg_diff
    df["xg_total"] = xg_total
    df["form_diff"] = df["home_form_pts"] - df["away_form_pts"]
    df["elo_xg_interaction"] = df["elo_diff"] * xg_diff
    df["pressure_x_time_remaining"] = df["home_pressure_score"] * (90.0 - clock)
    df["clock_normalized"] = clock / 90.0
    df["home_dominance"] = xg_diff / (xg_total + 0.1)
    df["score_xg_consistent"] = (
        ((df["score_diff"] > 0) & (xg_diff > 0)) |
        ((df["score_diff"] < 0) & (xg_diff < 0)) |
        (df["score_diff"] == 0)
    ).astype(float)
    df["late_game_state"] = ((clock > 75) & (df["score_diff"] != 0)).astype(float)
    df["home_xg_per_minute"] = df["home_xg_running"] / clock
    df["away_xg_per_minute"] = df["away_xg_running"] / clock
    df["xg_momentum_ratio"] = xg_diff / (xg_total + 0.1)
    return df


def simulate_market(X_val, y_val, X_train, y_train, noise_std=0.05, overround=0.04):
    rng = np.random.RandomState(42)
    clock_bins = np.arange(0, 95, 5)
    score_bins = np.arange(-3, 4)
    n_clock, n_score = len(clock_bins), len(score_bins)

    ci = np.clip(np.digitize(X_train[:, 1], clock_bins) - 1, 0, n_clock - 1)
    si = np.clip(np.digitize(X_train[:, 0], score_bins) - 1, 0, n_score - 1)

    global_freq = np.bincount(y_train, minlength=3).astype(float) / len(y_train)
    lookup = np.tile(global_freq, (n_clock, n_score, 1)).astype(np.float32)

    for c in range(n_clock):
        for s in range(n_score):
            mask = (ci == c) & (si == s)
            n = mask.sum()
            if n >= 10:
                lookup[c, s] = np.bincount(y_train[mask], minlength=3).astype(float) / n

    vci = np.clip(np.digitize(X_val[:, 1], clock_bins) - 1, 0, n_clock - 1)
    vsi = np.clip(np.digitize(X_val[:, 0], score_bins) - 1, 0, n_score - 1)
    market = lookup[vci, vsi].copy()
    market += rng.normal(0, noise_std, market.shape)
    market = np.clip(market, 0.02, 0.95)
    market *= (1.0 + overround) / market.sum(axis=1, keepdims=True)
    return market.astype(np.float32)


def simulate_trading(model_probs, y_val, market, et, ct, kf, bankroll=10000.0, max_bet_pct=0.02, min_bet=5.0):
    edges = model_probs - market
    tradable = (model_probs >= ct) & (edges >= et)
    masked = np.where(tradable, edges, -1.0)
    best_idx = np.argmax(masked, axis=1)
    best_edges = masked[np.arange(len(y_val)), best_idx]
    has_trade = best_edges >= 0

    if not has_trade.any():
        return {"pnl": 0, "trades": 0, "win_rate": 0, "roi": 0, "max_dd": 0}

    ti = np.where(has_trade)[0]
    te, tb, tm = best_edges[ti], best_idx[ti], market[ti, best_idx[ti]]
    ta = y_val[ti]

    full_kelly = te / (1.0 - tm)
    conf = model_probs[ti, tb] + te
    cm = np.where(conf > 0.8, 1.3, np.where(conf > 0.7, 1.1,
          np.where(conf > 0.6, 1.0, np.where(conf > 0.5, 0.8, 0.5))))

    bets = full_kelly * kf * cm * bankroll
    bets = np.minimum(bets, bankroll * max_bet_pct)
    bets = np.where(bets < min_bet, 0.0, bets)
    nz = bets > 0

    if not nz.any():
        return {"pnl": 0, "trades": 0, "win_rate": 0, "roi": 0, "max_dd": 0}

    bets, tb, tm, ta = bets[nz], tb[nz], tm[nz], ta[nz]
    odds = 1.0 / tm
    wins = ta == tb
    profits = np.where(wins, bets * (odds - 1.0), -bets)

    pnl = profits.sum()
    total_bet = bets.sum()
    n_trades = len(bets)
    win_rate = wins.sum() / n_trades * 100
    roi = pnl / total_bet * 100 if total_bet > 0 else 0

    cum = np.cumsum(profits)
    peak = np.maximum.accumulate(bankroll + cum)
    dd = ((peak - (bankroll + cum)) / peak).max() * 100

    return {"pnl": round(float(pnl), 2), "trades": int(n_trades),
            "win_rate": round(float(win_rate), 1), "roi": round(float(roi), 1),
            "max_dd": round(float(dd), 1)}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Threshold Optimization", flush=True)
    print("=" * 60, flush=True)

    print("\n[1] Loading data...", flush=True)
    df = pd.read_parquet("data/train_full.parquet")
    df = engineer_features(df)
    print(f"  Engineered: {time.time()-t0:.1f}s", flush=True)

    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[feat_cols].values.astype(np.float32)
    y = df["target"].values.astype(int)
    print(f"  Arrays: {X.shape}, {time.time()-t0:.1f}s", flush=True)

    # Split using pandas isin (fast on strings)
    um = df["match_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(um)
    split = int(len(um) * 0.8)
    train_set = set(um[:split].tolist())
    train_mask = df["match_id"].isin(train_set).values
    val_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"  Split: train={len(X_train)}, val={len(X_val)}, {time.time()-t0:.1f}s", flush=True)

    print("\n[2] Loading model...", flush=True)
    from model.train import SoccerEnsemble
    ensemble = SoccerEnsemble.load("model")
    print(f"  Loaded: {time.time()-t0:.1f}s", flush=True)

    print("\n[3] Predictions...", flush=True)
    model_probs = ensemble.predict(X_val)
    print(f"  {model_probs.shape}: {time.time()-t0:.1f}s", flush=True)

    from sklearn.metrics import log_loss
    ll = log_loss(y_val, model_probs)
    acc = (np.argmax(model_probs, axis=1) == y_val).mean()
    print(f"  Log loss: {ll:.4f}, Accuracy: {acc:.1%}", flush=True)

    print("\n[4] Market simulation...", flush=True)
    market = simulate_market(X_val, y_val, X_train, y_train)
    print(f"  Done: {time.time()-t0:.1f}s", flush=True)

    print("\n[5] Grid search (630 combos)...", flush=True)
    edge_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    conf_range = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    kelly_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    results = []
    best_pnl = -float("inf")
    best = None
    for et in edge_range:
        for ct in conf_range:
            for kf in kelly_range:
                r = simulate_trading(model_probs, y_val, market, et, ct, kf)
                r["edge"] = et; r["conf"] = ct; r["kelly"] = kf
                results.append(r)
                if r["pnl"] > best_pnl:
                    best_pnl = r["pnl"]; best = r

    print(f"  Done: {time.time()-t0:.1f}s", flush=True)
    results.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n{'='*70}", flush=True)
    print("TOP 10", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Edge':>6} {'Conf':>6} {'Kelly':>6} {'PnL':>10} {'Trades':>7} {'Win%':>6} {'ROI%':>7} {'MaxDD%':>7}", flush=True)
    for r in results[:10]:
        print(f"{r['edge']:>6.2f} {r['conf']:>6.2f} {r['kelly']:>6.2f} "
              f"${r['pnl']:>9.2f} {r['trades']:>7d} {r['win_rate']:>5.1f}% "
              f"{r['roi']:>6.1f}% {r['max_dd']:>6.1f}%", flush=True)

    print(f"\nWORST 5:", flush=True)
    for r in results[-5:]:
        print(f"{r['edge']:>6.2f} {r['conf']:>6.2f} {r['kelly']:>6.2f} "
              f"${r['pnl']:>9.2f} {r['trades']:>7d} {r['win_rate']:>5.1f}% "
              f"{r['roi']:>6.1f}% {r['max_dd']:>6.1f}%", flush=True)

    pd.DataFrame(results).to_csv("model/threshold_search_results.csv", index=False)
    bp = {"edge_threshold": float(best["edge"]), "confidence_threshold": float(best["conf"]),
          "kelly_fraction": float(best["kelly"]), "simulated_pnl": float(best["pnl"]),
          "simulated_roi": float(best["roi"]), "simulated_win_rate": float(best["win_rate"]),
          "simulated_trades": int(best["trades"]), "simulated_max_drawdown": float(best["max_dd"])}
    with open("model/optimized_thresholds.json", "w") as f:
        json.dump(bp, f, indent=2)

    print(f"\nBEST:", flush=True)
    for k, v in bp.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
