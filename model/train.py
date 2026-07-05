"""XGBoost + LightGBM + CatBoost ensemble training.

Trains an ensemble of XGBoost, LightGBM, and CatBoost classifiers with:
- GroupKFold by match_id (no data leakage)
- Optuna hyperparameter optimization
- Isotonic regression calibration
- Checkpointing every 15 minutes to Object Storage
- Saves calibrated model artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss

from model.calibrate import ProbabilityCalibrator
from model.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Checkpoint directory (Object Storage mount)
CHECKPOINT_DIR = Path("/workspace/output/checkpoints")
MODEL_DIR = Path("/workspace/output/model")


def load_training_data(
    parquet_path: str = "data/train.parquet",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data from parquet file.

    Args:
        parquet_path: Path to training data parquet.

    Returns:
        Tuple of (features_df, X, y, groups).
    """
    parquet_path_str = str(parquet_path)
    if not Path(parquet_path_str).exists():
        logger.info("Parquet not found at %s — building dataset from real sources", parquet_path_str)
        from data.build_dataset import build_dataset
        df = build_dataset(output_path=parquet_path_str)
        logger.info("Dataset saved to %s (%d rows)", parquet_path_str, len(df))
    else:
        df = pd.read_parquet(parquet_path_str)

    # Ensure correct feature columns
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(int)
    groups = df["match_id"].values

    logger.info(
        "Loaded %d samples, %d features, %d matches",
        len(X), len(FEATURE_NAMES), len(np.unique(groups)),
    )

    return df, X, y, groups


def _save_checkpoint(trial_results: List[Dict], best_params: Dict, best_score: float,
                     phase: str = "optuna") -> None:
    """Save checkpoint to Object Storage every 15 minutes."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Save trial history
    trials_path = CHECKPOINT_DIR / "optuna_trials.csv"
    pd.DataFrame(trial_results).to_csv(str(trials_path), index=False)

    # Save best params
    best_path = CHECKPOINT_DIR / "optuna_best_params.json"
    best_data = {
        "best_score": best_score,
        "best_params": best_params,
        "phase": phase,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": len(trial_results),
    }
    best_path.write_text(json.dumps(best_data, indent=2))

    logger.info("Checkpoint saved: %s (score=%.4f, %d trials)", phase, best_score, len(trial_results))


def _save_interim_model(xgb_model, lgbm_model, cb_model, calibrator,
                        xgb_weight, lgbm_weight, cb_weight, score: float) -> None:
    """Save interim best model to checkpoint directory."""
    interim_dir = CHECKPOINT_DIR / "interim_model"
    interim_dir.mkdir(parents=True, exist_ok=True)

    if xgb_model is not None:
        joblib.dump(xgb_model, interim_dir / "xgb_soccer.pkl")
    if lgbm_model is not None:
        joblib.dump(lgbm_model, interim_dir / "lgbm_soccer.pkl")
    if cb_model is not None:
        joblib.dump(cb_model, interim_dir / "catboost_soccer.pkl")
    if calibrator is not None:
        calibrator.save(str(interim_dir / "calibrator.pkl"))

    meta = {
        "xgb_weight": xgb_weight,
        "lgbm_weight": lgbm_weight,
        "cb_weight": cb_weight,
        "best_score": score,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "target_classes": ["home", "draw", "away"],
    }
    (interim_dir / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Interim model saved (score=%.4f)", score)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        params: Optional hyperparameters (uses defaults if None).

    Returns:
        Trained XGBClassifier.
    """
    logger.info("Training XGBoost...")

    base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
    }

    if params:
        final_params = {**base_params, **params}
    else:
        final_params = {
            **base_params,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
        }

    final_params["early_stopping_rounds"] = 50
    model = xgb.XGBClassifier(**final_params)
    fit_idx = np.random.permutation(len(X_train))
    model.fit(
        X_train[fit_idx], y_train[fit_idx],
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_val)
    ll = log_loss(y_val, y_pred_proba)
    logger.info("XGBoost validation log loss: %.4f", ll)

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
) -> lgb.LGBMClassifier:
    """Train LightGBM classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        params: Optional hyperparameters (uses defaults if None).

    Returns:
        Trained LGBMClassifier.
    """
    logger.info("Training LightGBM...")

    if params:
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        lgb_params.update(params)
    else:
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    y_pred_proba = model.predict_proba(X_val)
    ll = log_loss(y_val, y_pred_proba)
    logger.info("LightGBM validation log loss: %.4f", ll)

    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
):
    """Train CatBoost classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        params: Optional hyperparameters (uses defaults if None).

    Returns:
        Trained CatBoostClassifier.
    """
    from catboost import CatBoostClassifier

    logger.info("Training CatBoost...")

    if params:
        cb_params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "classes_count": 3,
            "random_seed": 42,
            "verbose": 50,
            "early_stopping_rounds": 50,
        }
        cb_params.update(params)
    else:
        cb_params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "classes_count": 3,
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 1500,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": 50,
            "early_stopping_rounds": 50,
        }

    model = CatBoostClassifier(**cb_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_val)
    ll = log_loss(y_val, y_pred_proba)
    logger.info("CatBoost validation log loss: %.4f", ll)

    return model


class SoccerEnsemble:
    """Ensemble of XGBoost, LightGBM, and CatBoost for win probability prediction.

    Weighted average of model outputs before calibration.
    """

    def __init__(
        self,
        xgb_model=None,
        lgbm_model=None,
        cb_model=None,
        calibrator: Optional[ProbabilityCalibrator] = None,
        xgb_weight: float = 0.33,
        lgbm_weight: float = 0.33,
        cb_weight: float = 0.34,
    ) -> None:
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.cb_model = cb_model
        self.calibrator = calibrator
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.cb_weight = cb_weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities.

        Args:
            X: Feature array (N, 39).

        Returns:
            Calibrated probabilities (N, 3) for [home, draw, away].
        """
        probs_list = []

        if self.xgb_model is not None:
            xgb_probs = self.xgb_model.predict_proba(X)
            probs_list.append((xgb_probs, self.xgb_weight))

        if self.lgbm_model is not None:
            lgbm_probs = self.lgbm_model.predict_proba(X)
            probs_list.append((lgbm_probs, self.lgbm_weight))

        if self.cb_model is not None:
            cb_probs = self.cb_model.predict_proba(X)
            probs_list.append((cb_probs, self.cb_weight))

        if not probs_list:
            raise ValueError("No models in ensemble")

        # Weighted average
        total_weight = sum(w for _, w in probs_list)
        ensemble_probs = sum(p * w for p, w in probs_list) / total_weight

        # Calibrate
        if self.calibrator is not None:
            ensemble_probs = self.calibrator.predict(ensemble_probs)

        return ensemble_probs

    def predict_single(self, X: np.ndarray) -> np.ndarray:
        """Predict for a single sample.

        Args:
            X: Feature array (1, 39).

        Returns:
            Calibrated probabilities (3,).
        """
        return self.predict(X)[0]

    def save(self, output_dir: str = "model") -> None:
        """Save all model artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, output_path / "xgb_soccer.pkl")
            logger.info("XGBoost model saved")

        if self.lgbm_model is not None:
            joblib.dump(self.lgbm_model, output_path / "lgbm_soccer.pkl")
            logger.info("LightGBM model saved")

        if self.cb_model is not None:
            joblib.dump(self.cb_model, output_path / "catboost_soccer.pkl")
            logger.info("CatBoost model saved")

        if self.calibrator is not None:
            self.calibrator.save(str(output_path / "calibrator.pkl"))

        # Save metadata
        meta = {
            "xgb_weight": self.xgb_weight,
            "lgbm_weight": self.lgbm_weight,
            "cb_weight": self.cb_weight,
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "target_classes": ["home", "draw", "away"],
        }
        (output_path / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: str = "model") -> "SoccerEnsemble":
        """Load ensemble from disk."""
        model_path = Path(model_dir)

        xgb_model = None
        lgbm_model = None
        cb_model = None
        calibrator = None

        xgb_path = model_path / "xgb_soccer.pkl"
        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)

        lgbm_path = model_path / "lgbm_soccer.pkl"
        if lgbm_path.exists():
            lgbm_model = joblib.load(lgbm_path)

        cb_path = model_path / "catboost_soccer.pkl"
        if cb_path.exists():
            cb_model = joblib.load(cb_path)

        cal_path = model_path / "calibrator.pkl"
        if cal_path.exists():
            calibrator = ProbabilityCalibrator.load(str(cal_path))

        # Load weights from metadata
        meta_path = model_path / "ensemble_meta.json"
        xgb_weight = 0.33
        lgbm_weight = 0.33
        cb_weight = 0.34
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            xgb_weight = meta.get("xgb_weight", 0.33)
            lgbm_weight = meta.get("lgbm_weight", 0.33)
            cb_weight = meta.get("cb_weight", 0.34)

        ensemble = cls(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            cb_model=cb_model,
            calibrator=calibrator,
            xgb_weight=xgb_weight,
            lgbm_weight=lgbm_weight,
            cb_weight=cb_weight,
        )
        logger.info("Ensemble loaded from %s", model_dir)
        return ensemble


def _optuna_objective(
    trial, X_train, y_train, X_val, y_val, groups_train,
    checkpoint_trials: List[Dict], checkpoint_timer: List[float],
    use_catboost: bool = True,
) -> float:
    """Optuna objective function for hyperparameter search."""
    # XGBoost params
    xgb_params = {
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("xgb_n_est", 500, 3000, step=100),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
    }

    # LightGBM params
    lgb_params = {
        "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 255),
        "max_depth": trial.suggest_int("lgb_max_depth", 3, 15),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("lgb_n_est", 500, 3000, step=100),
        "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("lgb_min_child", 5, 50),
        "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-8, 10.0, log=True),
    }

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, params=xgb_params)

    # Train LightGBM
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val, params=lgb_params)

    # Train CatBoost (optional)
    cb_model = None
    if use_catboost:
        cb_params = {
            "depth": trial.suggest_int("cb_depth", 3, 10),
            "learning_rate": trial.suggest_float("cb_lr", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("cb_iter", 500, 3000, step=100),
            "l2_leaf_reg": trial.suggest_float("cb_l2", 1e-8, 10.0, log=True),
        }
        cb_model = train_catboost(X_train, y_train, X_val, y_val, params=cb_params)

    # Ensemble prediction (before calibration)
    xgb_probs = xgb_model.predict_proba(X_val)
    lgbm_probs = lgbm_model.predict_proba(X_val)

    if cb_model is not None:
        cb_probs = cb_model.predict_proba(X_val)
        # Equal weight average for trial evaluation
        ensemble_probs = (xgb_probs + lgbm_probs + cb_probs) / 3.0
    else:
        ensemble_probs = (xgb_probs + lgbm_probs) / 2.0

    # Evaluate
    score = log_loss(y_val, ensemble_probs)

    # Record trial result
    trial_result = {
        "trial": trial.number,
        "log_loss": score,
        "xgb_depth": xgb_params["max_depth"],
        "xgb_lr": xgb_params["learning_rate"],
        "lgb_num_leaves": lgb_params["num_leaves"],
        "lgb_lr": lgb_params["learning_rate"],
    }
    checkpoint_trials.append(trial_result)

    # Checkpoint every 15 minutes
    now = time.time()
    if now - checkpoint_timer[0] >= 900:  # 15 minutes
        # Find best score so far
        best_idx = np.argmin([t["log_loss"] for t in checkpoint_trials])
        best_trial = checkpoint_trials[best_idx]
        _save_checkpoint(checkpoint_trials, trial.params, best_trial["log_loss"], "optuna")

        # Save interim model if this is the best
        if score <= best_trial["log_loss"]:
            _save_interim_model(xgb_model, lgbm_model, cb_model, None, 0.33, 0.33, 0.34, score)

        checkpoint_timer[0] = now

    return score


def run_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_train: np.ndarray,
    n_trials: int = 200,
    use_catboost: bool = True,
) -> Tuple[Dict, float]:
    """Run Optuna hyperparameter optimization.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        groups_train: Group labels for training data.
        n_trials: Number of Optuna trials.
        use_catboost: Whether to include CatBoost.

    Returns:
        Tuple of (best_params, best_score).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("Starting Optuna optimization: %d trials", n_trials)

    checkpoint_trials = []
    checkpoint_timer = [time.time()]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_train, y_train, X_val, y_val, groups_train,
            checkpoint_trials, checkpoint_timer, use_catboost,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_score = study.best_value

    logger.info("Optuna complete: best log_loss=%.4f", best_score)
    logger.info("Best params: %s", json.dumps(best_params, indent=2))

    # Final checkpoint
    _save_checkpoint(checkpoint_trials, best_params, best_score, "optuna_final")

    return best_params, best_score


def train(
    data_path: str = "data/train.parquet",
    output_dir: str = "model",
    use_optuna: bool = False,
    optuna_trials: int = 200,
    use_catboost: bool = True,
) -> SoccerEnsemble:
    """Full training pipeline.

    Args:
        data_path: Path to training data.
        output_dir: Where to save model artifacts.
        use_optuna: Whether to run Optuna optimization.
        optuna_trials: Number of Optuna trials.
        use_catboost: Whether to include CatBoost.

    Returns:
        Trained SoccerEnsemble.
    """
    start = time.time()

    # Load data
    df, X, y, groups = load_training_data(data_path)

    # Split by match (GroupKFold ensures no match leakage)
    unique_matches = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_matches)
    split_idx = int(len(unique_matches) * 0.8)
    train_matches = set(unique_matches[:split_idx])
    val_matches = set(unique_matches[split_idx:])

    train_mask = np.isin(groups, list(train_matches))
    val_mask = np.isin(groups, list(val_matches))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    groups_train = groups[train_mask]

    # Shuffle training data to prevent class-ordered splits
    train_shuffle = np.random.permutation(len(X_train))
    X_train = X_train[train_shuffle]
    y_train = y_train[train_shuffle]
    groups_train = groups_train[train_shuffle]

    logger.info(
        "Split: %d train, %d val (%d train matches, %d val matches)",
        len(X_train), len(X_val), len(train_matches), len(val_matches),
    )

    # Hyperparameter optimization
    xgb_params = None
    lgb_params = None
    cb_params = None

    if use_optuna:
        best_params, best_score = run_optuna(
            X_train, y_train, X_val, y_val, groups_train,
            n_trials=optuna_trials, use_catboost=use_catboost,
        )

        # Extract per-model params from Optuna results
        xgb_params = {
            "max_depth": best_params.get("xgb_max_depth", 6),
            "learning_rate": best_params.get("xgb_lr", 0.05),
            "n_estimators": best_params.get("xgb_n_est", 1500),
            "subsample": best_params.get("xgb_subsample", 0.8),
            "colsample_bytree": best_params.get("xgb_colsample", 0.8),
            "min_child_weight": best_params.get("xgb_min_child", 5),
            "reg_alpha": best_params.get("xgb_reg_alpha", 0.1),
            "reg_lambda": best_params.get("xgb_reg_lambda", 0.1),
        }
        lgb_params = {
            "num_leaves": best_params.get("lgb_num_leaves", 63),
            "max_depth": best_params.get("lgb_max_depth", 8),
            "learning_rate": best_params.get("lgb_lr", 0.05),
            "n_estimators": best_params.get("lgb_n_est", 1500),
            "subsample": best_params.get("lgb_subsample", 0.8),
            "colsample_bytree": best_params.get("lgb_colsample", 0.8),
            "min_child_samples": best_params.get("lgb_min_child", 20),
            "reg_alpha": best_params.get("lgb_reg_alpha", 0.1),
            "reg_lambda": best_params.get("lgb_reg_lambda", 0.1),
        }
        if use_catboost:
            cb_params = {
                "depth": best_params.get("cb_depth", 6),
                "learning_rate": best_params.get("cb_lr", 0.05),
                "iterations": best_params.get("cb_iter", 1500),
                "l2_leaf_reg": best_params.get("cb_l2", 3.0),
            }

    # Train models with best params (or defaults)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, params=xgb_params)
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val, params=lgb_params)

    cb_model = None
    if use_catboost:
        cb_model = train_catboost(X_train, y_train, X_val, y_val, params=cb_params)

    # Find optimal ensemble weights via grid search
    logger.info("Optimizing ensemble weights...")
    xgb_probs = xgb_model.predict_proba(X_val)
    lgbm_probs = lgbm_model.predict_proba(X_val)

    if cb_model is not None:
        cb_probs = cb_model.predict_proba(X_val)
        best_ll = float("inf")
        best_weights = (0.33, 0.33, 0.34)
        for w1 in np.arange(0.1, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.8 - w1, 0.1):
                w3 = max(0.1, 1.0 - w1 - w2)
                if w1 + w2 + w3 < 0.9:
                    continue
                ens = (xgb_probs * w1 + lgbm_probs * w2 + cb_probs * w3) / (w1 + w2 + w3)
                ll = log_loss(y_val, ens)
                if ll < best_ll:
                    best_ll = ll
                    best_weights = (w1, w2, w3)
        xgb_weight, lgbm_weight, cb_weight = best_weights
        logger.info("Best weights: XGB=%.2f, LGBM=%.2f, CB=%.2f (log_loss=%.4f)",
                    xgb_weight, lgbm_weight, cb_weight, best_ll)
    else:
        best_ll = float("inf")
        best_weights = (0.5, 0.5)
        for w1 in np.arange(0.1, 0.9, 0.1):
            w2 = 1.0 - w1
            ens = (xgb_probs * w1 + lgbm_probs * w2)
            ll = log_loss(y_val, ens)
            if ll < best_ll:
                best_ll = ll
                best_weights = (w1, w2)
        xgb_weight, lgbm_weight = best_weights
        cb_weight = 0.0
        logger.info("Best weights: XGB=%.2f, LGBM=%.2f (log_loss=%.4f)",
                    xgb_weight, lgbm_weight, best_ll)

    # Create ensemble with optimal weights
    ensemble = SoccerEnsemble(
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        cb_model=cb_model,
        xgb_weight=xgb_weight,
        lgbm_weight=lgbm_weight,
        cb_weight=cb_weight,
    )

    # Calibrate on validation set
    raw_probs = ensemble.predict(X_val)
    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_val, raw_probs)
    ensemble.calibrator = calibrator

    # Final evaluation
    calibrated_probs = calibrator.predict(raw_probs)
    final_ll = log_loss(y_val, calibrated_probs)

    # Per-class Brier scores
    for i, cls_name in enumerate(["home", "draw", "away"]):
        binary_true = (y_val == i).astype(float)
        bs = brier_score_loss(binary_true, calibrated_probs[:, i])
        logger.info("Brier score (%s): %.4f", cls_name, bs)

    logger.info("Final calibrated log loss: %.4f", final_ll)

    # Save to local output
    ensemble.save(output_dir)

    # Checkpoint to Object Storage
    if MODEL_DIR.parent.exists():
        ensemble.save(str(MODEL_DIR))
        logger.info("Model checkpointed to %s", MODEL_DIR)

    elapsed = time.time() - start
    logger.info("Training complete in %.1f seconds", elapsed)

    return ensemble


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM + CatBoost ensemble"
    )
    parser.add_argument("--data", default="data/train.parquet", help="Training data")
    parser.add_argument("--output", default="model", help="Output directory")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--optuna-trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--no-catboost", action="store_true", help="Disable CatBoost")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train(
        data_path=args.data,
        output_dir=args.output,
        use_optuna=args.optuna,
        optuna_trials=args.optuna_trials,
        use_catboost=not args.no_catboost,
    )


if __name__ == "__main__":
    main()
