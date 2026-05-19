# train_model.py — Eco-Ring LightGBM demand prediction trainer
#
# Outputs:
#   model/demand_model.lgb      — saved model (load with lgb.Booster)
#   data/model_metrics.json     — MAE / RMSE / R² (overall + per ring)
#   data/feature_importance.png — gain-based feature importance chart

import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH   = "data/demand_log.csv"
MODEL_PATH = "model/demand_model.lgb"
METRICS_PATH = "data/model_metrics.json"
FI_PLOT_PATH = "data/feature_importance.png"

FEATURES = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "slot_of_day", "week_number",
    "is_weekend", "is_exam_week", "is_holiday",
    "stop_id", "stop_type", "ring_id",
]
CATEGORICAL_FEATURES = ["stop_id", "stop_type", "ring_id"]
TARGET      = "actual_demand"
TRAIN_RATIO = 0.80


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp", "date"])

    # Booleans are stored as strings "True"/"False" in CSV
    for col in ["is_weekend", "is_exam_week", "is_holiday"]:
        df[col] = df[col].astype(str).map({"True": True, "False": False}).astype(int)

    # Cyclic encoding — makes hour 23 adjacent to hour 0 (and day 6 adjacent to day 0)
    # Important for models that would otherwise treat these as maximally far apart.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # pandas category dtype — required for LightGBM native categorical handling
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# Time-based train/test split
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, ratio: float) -> tuple:
    """Split by date, not by row. First `ratio` of dates → train, rest → test.

    Time-based splitting prevents future data from leaking into training,
    which random splitting would do and which inflates metrics unrealistically.
    """
    unique_dates = sorted(df["date"].unique())
    cutoff_idx   = int(len(unique_dates) * ratio)
    train_dates  = set(unique_dates[:cutoff_idx])
    test_dates   = set(unique_dates[cutoff_idx:])

    train_df = df[df["date"].isin(train_dates)].copy()
    test_df  = df[df["date"].isin(test_dates)].copy()

    t0 = pd.Timestamp(min(train_dates)).date()
    t1 = pd.Timestamp(max(train_dates)).date()
    v0 = pd.Timestamp(min(test_dates)).date()
    v1 = pd.Timestamp(max(test_dates)).date()
    print(f"    Train : {len(train_df):>7,} rows  ({t0} → {t1})")
    print(f"    Test  : {len(test_df):>7,} rows  ({v0} → {v1})")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(train_df: pd.DataFrame) -> lgb.LGBMRegressor:
    X = train_df[FEATURES]
    y = train_df[TARGET]

    # Last 10% of training rows → early-stopping validation
    # (kept in chronological order so it reflects near-future performance)
    val_size   = int(len(X) * 0.10)
    X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

    model = lgb.LGBMRegressor(
        objective         = "regression_l1",  # optimises MAE directly
        num_leaves        = 63,
        learning_rate     = 0.05,
        n_estimators      = 1000,
        min_child_samples = 20,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 5,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set            = [(X_val, y_val)],
        categorical_feature = CATEGORICAL_FEATURES,
        callbacks           = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    n_trees = model.booster_.num_trees()
    print(f"    Trees used (early stopping): {n_trees} / 1000")
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def _fmt_row(label, m):
    return f"    {label:<20}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}"


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------

def save_feature_importance(model: lgb.LGBMRegressor, path: str):
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot.barh(ax=ax, color="steelblue")
    ax.set_title("LightGBM Feature Importance (gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n=== Eco-Ring — LightGBM Training ===\n")

    # 1. Load
    print("[1/5] Loading and preparing data...")
    df = load_and_prepare(CSV_PATH)
    print(f"    {len(df):,} rows  |  {len(FEATURES)} features  |  target: {TARGET}")

    # 2. Split
    print("\n[2/5] Time-based train/test split (80 / 20)...")
    train_df, test_df = time_split(df, TRAIN_RATIO)

    # 3. Train
    print("\n[3/5] Training LightGBM...")
    model = train(train_df)

    # 4. Evaluate
    print("\n[4/5] Evaluating on test set...")
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].values
    y_pred = np.clip(model.predict(X_test), 0, None)

    test_df = test_df.copy()
    test_df["predicted"] = y_pred

    overall = compute_metrics(y_test, y_pred)

    print(f"\n    {'Ring':<20}  {'MAE':>8}  {'RMSE':>8}  {'R²':>7}  {'Rows':>7}")
    print(f"    {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")
    print(f"    {'ALL RINGS':<20}  {overall['mae']:>8.4f}  {overall['rmse']:>8.4f}  {overall['r2']:>7.4f}  {len(test_df):>7,}")

    per_ring = {}
    for ring_id, group in test_df.groupby("ring_id", observed=True):
        m = compute_metrics(group[TARGET], group["predicted"])
        m["rows"] = len(group)
        per_ring[str(ring_id)] = m
        print(f"    {str(ring_id):<20}  {m['mae']:>8.4f}  {m['rmse']:>8.4f}  {m['r2']:>7.4f}  {m['rows']:>7,}")

    # Project target check
    mae_ok = "PASS" if overall["mae"] <= 2.0 else "FAIL"
    r2_ok  = "PASS" if overall["r2"]  >= 0.85 else "FAIL"
    print(f"\n    MAE ≤ 2.0  →  {overall['mae']:.4f}  [{mae_ok}]")
    print(f"    R²  ≥ 0.85 →  {overall['r2']:.4f}  [{r2_ok}]")

    # 5. Sample predictions
    print("\n[5/5] Sample predictions (20 rows, sorted by timestamp)...")
    cols = ["timestamp", "ring_id", "stop_id", "stop_name", TARGET, "predicted"]
    sample = test_df[cols].copy()
    sample["ring_id"] = sample["ring_id"].astype(str)
    sample["stop_id"] = sample["stop_id"].astype(str)
    sample = sample.sample(20, random_state=42).sort_values("timestamp")
    sample["predicted"] = sample["predicted"].round(2)
    print(sample.to_string(index=False))

    # Save model
    print()
    os.makedirs("model", exist_ok=True)
    model.booster_.save_model(MODEL_PATH)
    print(f"    Model saved   : {MODEL_PATH}")

    # Save feature importance plot
    save_feature_importance(model, FI_PLOT_PATH)
    print(f"    FI plot saved : {FI_PLOT_PATH}")

    # Save metrics JSON
    n_trees = model.booster_.num_trees()
    metrics_doc = {
        "overall": {
            **overall,
            "n_trees_used": n_trees,
            "train_rows":   len(train_df),
            "test_rows":    len(test_df),
        },
        "per_ring": per_ring,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_doc, f, indent=2)
    print(f"    Metrics saved : {METRICS_PATH}")

    print("\n=====================================\n")


if __name__ == "__main__":
    main()
