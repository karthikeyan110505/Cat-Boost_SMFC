#!/usr/bin/env python3
"""Train a CatBoost regressor for 78-hour-ahead SMFC voltage forecasting.

This module is designed to run as a script and to be imported from notebooks.
It performs:
1) data loading + quality inspection,
2) time-series feature engineering,
3) chronological train/validation/test splitting,
4) CatBoost hyperparameter tuning,
5) final evaluation + visualization,
6) artifact saving for production inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("smfc_catboost")
HORIZON_STEPS = 13  # 13 x 6h = 78h


@dataclass
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def parse_smfc_timestamp(time_text: str, origin: str = "2024-01-01") -> pd.Timestamp:
    """Parse values like 'day-3; 7pm' into real timestamps.

    Args:
        time_text: Original time text from dataset.
        origin: Starting reference date for day-1.

    Returns:
        pandas Timestamp.
    """
    pattern = r"day-(\d+)\s*;\s*(\d+)(am|pm)"
    match = re.fullmatch(pattern, str(time_text).strip().lower())
    if not match:
        raise ValueError(f"Unrecognized TIME value: {time_text}")

    day_number = int(match.group(1))
    hour_12 = int(match.group(2))
    period = match.group(3)

    if hour_12 < 1 or hour_12 > 12:
        raise ValueError(f"Invalid hour in TIME value: {time_text}")

    if period == "am":
        hour_24 = 0 if hour_12 == 12 else hour_12
    else:
        hour_24 = 12 if hour_12 == 12 else hour_12 + 12

    base = pd.Timestamp(origin)
    return base + pd.Timedelta(days=day_number - 1, hours=hour_24)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate input SMFC dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_columns = {"TIME", "VOLTAGE"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(expected_columns)}")

    df = df.copy()
    df["timestamp"] = df["TIME"].map(parse_smfc_timestamp)
    df["voltage"] = pd.to_numeric(df["VOLTAGE"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return df[["timestamp", "voltage"]]


def inspect_data(df: pd.DataFrame) -> Dict[str, float]:
    """Return data quality summary metrics."""
    diffs_h = df["timestamp"].diff().dropna().dt.total_seconds() / 3600
    summary = {
        "rows": int(len(df)),
        "missing_voltage": int(df["voltage"].isna().sum()),
        "min_voltage": float(df["voltage"].min()),
        "max_voltage": float(df["voltage"].max()),
        "mean_voltage": float(df["voltage"].mean()),
        "median_sampling_hours": float(diffs_h.median()) if not diffs_h.empty else np.nan,
        "irregular_intervals": int((diffs_h != 6).sum()) if not diffs_h.empty else 0,
    }
    return summary


def engineer_features(df: pd.DataFrame, horizon_steps: int = HORIZON_STEPS) -> pd.DataFrame:
    """Create forecasting features and future target."""
    out = df.copy()

    lag_steps = [1, 2, 3, 6, 12, 13, 24]
    for lag in lag_steps:
        out[f"voltage_lag_{lag}"] = out["voltage"].shift(lag)

    for window in [3, 6, 12, 24]:
        roll = out["voltage"].rolling(window=window, min_periods=window)
        out[f"roll_mean_{window}"] = roll.mean()
        out[f"roll_std_{window}"] = roll.std()
        out[f"roll_min_{window}"] = roll.min()
        out[f"roll_max_{window}"] = roll.max()

    out["diff_1"] = out["voltage"].diff(1)
    out["diff_2"] = out["voltage"].diff(2)
    out["roc_1"] = out["voltage"].pct_change(periods=1)
    out["acceleration"] = out["diff_1"].diff(1)

    out["roll_range_6"] = out["roll_max_6"] - out["roll_min_6"]
    out["roll_cv_6"] = out["roll_std_6"] / (out["roll_mean_6"].abs() + 1e-6)
    out["voltage_stability_12"] = 1 / (out["roll_std_12"] + 1e-6)
    out["ema_6"] = out["voltage"].ewm(span=6, adjust=False).mean()
    out["ema_12"] = out["voltage"].ewm(span=12, adjust=False).mean()
    out["ema_gap"] = out["ema_6"] - out["ema_12"]

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    out["target_78h"] = out["voltage"].shift(-horizon_steps)
    out = out.drop(columns=["hour", "day_of_week"])

    # Keep rows where target and minimum lag windows are available.
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def temporal_split(features_df: pd.DataFrame) -> SplitData:
    """Chronological 70/15/15 split."""
    if len(features_df) < 30:
        raise ValueError("Not enough samples after feature engineering for robust split.")

    feature_cols = [c for c in features_df.columns if c not in {"timestamp", "voltage", "target_78h"}]
    x = features_df[feature_cols]
    y = features_df["target_78h"]

    n = len(features_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return SplitData(
        x_train=x.iloc[:train_end],
        y_train=y.iloc[:train_end],
        x_val=x.iloc[train_end:val_end],
        y_val=y.iloc[train_end:val_end],
        x_test=x.iloc[val_end:],
        y_test=y.iloc[val_end:],
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape(y_true, y_pred),
    }


def tune_catboost(split: SplitData, random_seed: int = 42) -> Tuple[CatBoostRegressor, Dict[str, float], Dict[str, float]]:
    """Small, deterministic hyperparameter search using validation RMSE."""
    param_grid = [
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3, "iterations": 400},
        {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5, "iterations": 500},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "iterations": 500},
        {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7, "iterations": 700},
        {"depth": 7, "learning_rate": 0.02, "l2_leaf_reg": 9, "iterations": 900},
    ]

    best_model = None
    best_params = None
    best_metrics = None

    for params in param_grid:
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=random_seed,
            verbose=False,
            **params,
        )
        model.fit(
            split.x_train,
            split.y_train,
            eval_set=(split.x_val, split.y_val),
            use_best_model=True,
            early_stopping_rounds=80,
        )
        val_pred = model.predict(split.x_val)
        metrics = evaluate(split.y_val.values, val_pred)

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_model = model
            best_params = params
            best_metrics = metrics

    assert best_model is not None and best_params is not None and best_metrics is not None
    return best_model, best_params, best_metrics


def save_plots(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, title_suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual", marker="o")
    plt.plot(y_pred, label="Predicted", marker="x")
    plt.title(f"Actual vs Predicted ({title_suffix})")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"actual_vs_pred_{title_suffix.lower()}.png", dpi=150)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuals, alpha=0.8)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residual Plot ({title_suffix})")
    plt.xlabel("Predicted Voltage")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(output_dir / f"residuals_{title_suffix.lower()}.png", dpi=150)
    plt.close()


def run_pipeline(data_path: Path, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    quality = inspect_data(df)
    feats = engineer_features(df, horizon_steps=HORIZON_STEPS)
    split = temporal_split(feats)

    model, best_params, val_metrics = tune_catboost(split)

    test_pred = model.predict(split.x_test)
    test_metrics = evaluate(split.y_test.values, test_pred)

    model.save_model(str(output_dir / "catboost_smfc_78h.cbm"))
    (output_dir / "feature_columns.json").write_text(
        json.dumps(split.x_train.columns.tolist(), indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        {
            "actual": split.y_test.values,
            "predicted": test_pred,
            "residual": split.y_test.values - test_pred,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    importances = model.get_feature_importance(prettified=True)
    importances.to_csv(output_dir / "feature_importance.csv", index=False)

    save_plots(output_dir, split.y_val.values, model.predict(split.x_val), "Validation")
    save_plots(output_dir, split.y_test.values, test_pred, "Test")

    summary = {
        "data_quality": quality,
        "samples_after_feature_engineering": len(feats),
        "split_sizes": {
            "train": int(len(split.x_train)),
            "validation": int(len(split.x_val)),
            "test": int(len(split.x_test)),
        },
        "best_params": best_params,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_latest_feature_row(recent_df: pd.DataFrame) -> pd.DataFrame:
    """Build engineered features and return only latest row for inference."""
    recent_df = recent_df.copy()
    recent_df["timestamp"] = recent_df["TIME"].map(parse_smfc_timestamp)
    recent_df["voltage"] = pd.to_numeric(recent_df["VOLTAGE"], errors="coerce")
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)

    feats = engineer_features(recent_df.assign(dummy_target=np.nan).drop(columns=["dummy_target"], errors="ignore"), horizon_steps=0)
    if feats.empty:
        raise ValueError("Insufficient history in recent data to generate features.")

    feature_cols = [c for c in feats.columns if c not in {"timestamp", "voltage", "target_78h"}]
    return feats.iloc[[-1]][feature_cols]


def predict_from_csv(model_path: Path, feature_cols_path: Path, input_csv: Path) -> float:
    """Predict one 78h-ahead value from the latest available history in input_csv."""
    if not model_path.exists() or not feature_cols_path.exists() or not input_csv.exists():
        raise FileNotFoundError("Model, feature column file, and input CSV must all exist.")

    model = CatBoostRegressor()
    model.load_model(str(model_path))

    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    recent_df = pd.read_csv(input_csv)
    latest_features = prepare_latest_feature_row(recent_df)

    missing_cols = [c for c in feature_cols if c not in latest_features.columns]
    if missing_cols:
        raise ValueError(f"Missing expected features for inference: {missing_cols}")

    latest_features = latest_features[feature_cols]
    pred = float(model.predict(latest_features)[0])
    return pred


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CatBoost model for SMFC 78h voltage forecasting")
    parser.add_argument("--data", type=Path, default=Path("data_smfc_data.csv"), help="Path to input CSV")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Output directory")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()

    try:
        summary = run_pipeline(args.data, args.output)
        LOGGER.info("Training complete. Summary:\n%s", json.dumps(summary, indent=2))
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
