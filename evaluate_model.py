# evaluate_model.py — Eco-Ring visual model audit
#
# Run AFTER train_model.py:
#   python evaluate_model.py
#
# Loads model/demand_model.lgb, recreates the same train/test split,
# and saves 5 diagnostic plots to data/model_evaluation.png

import os
import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_model import (
    CSV_PATH,
    FEATURES,
    TARGET,
    TRAIN_RATIO,
    compute_metrics,
    load_and_prepare,
    time_split,
)

MODEL_PATH     = "model/demand_model.lgb"
EVAL_PLOT_PATH = "data/model_evaluation.png"


def load_model(path: str) -> lgb.Booster:
    if not os.path.exists(path):
        print(f"Error: '{path}' not found. Run train_model.py first.")
        sys.exit(1)
    return lgb.Booster(model_file=path)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(ax, y_true, y_pred):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.15, s=5, color="steelblue")
    upper = max(y_true.max(), y_pred.max()) + 1
    ax.plot([0, upper], [0, upper], color="red", linewidth=1,
            linestyle="--", label="Perfect fit")
    ax.set_title("Predicted vs Actual Demand")
    ax.set_xlabel("Actual Demand (passengers)")
    ax.set_ylabel("Predicted Demand (passengers)")
    ax.legend(fontsize=8)


def plot_residuals(ax, y_true, y_pred):
    residuals = y_pred - y_true
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linewidth=1, linestyle="--", label="Zero error")
    ax.set_title("Residual Distribution  (predicted − actual)")
    ax.set_xlabel("Residual (passengers)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.text(
        0.97, 0.95,
        f"mean = {residuals.mean():.2f}\nstd  = {residuals.std():.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def plot_error_by_hour(ax, test_df):
    mae_by_hour = (
        (test_df["predicted"] - test_df[TARGET]).abs()
        .groupby(test_df["hour"])
        .mean()
    )
    ax.bar(mae_by_hour.index, mae_by_hour.values, color="steelblue", width=0.8)
    ax.axhline(2.0, color="red", linewidth=1, linestyle="--", label="MAE = 2 target")
    ax.set_title("Mean Absolute Error by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MAE (passengers)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(fontsize=8)


def plot_error_by_ring(ax, test_df):
    ring_labels = test_df["ring_id"].astype(str)
    mae_by_ring = (
        (test_df["predicted"] - test_df[TARGET]).abs()
        .groupby(ring_labels)
        .mean()
        .sort_values(ascending=False)
    )
    bars = ax.bar(mae_by_ring.index, mae_by_ring.values, color="teal", width=0.6)
    ax.axhline(2.0, color="red", linewidth=1, linestyle="--", label="MAE = 2 target")
    peak = mae_by_ring.max()
    for bar, val in zip(bars, mae_by_ring.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + peak * 0.015,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_title("Mean Absolute Error by Ring")
    ax.set_xlabel("Ring")
    ax.set_ylabel("MAE (passengers)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    ax.set_ylim(0, peak * 1.25)


def plot_feature_importance(ax, model: lgb.Booster):
    names  = model.feature_name()
    values = model.feature_importance(importance_type="gain")
    ser    = pd.Series(values, index=names).sort_values(ascending=True)
    ser.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Feature Importance (gain)")
    ax.set_xlabel("Importance")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n=== Eco-Ring — Model Evaluation ===\n")

    print("Loading and preparing data...")
    df = load_and_prepare(CSV_PATH)
    _, test_df = time_split(df, TRAIN_RATIO)

    print("\nLoading model from", MODEL_PATH)
    model = load_model(MODEL_PATH)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].values
    y_pred = np.clip(model.predict(X_test), 0, None)

    test_df = test_df.copy()
    test_df["predicted"] = y_pred

    m = compute_metrics(y_test, y_pred)
    print(f"\nTest metrics — MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    print("\nRendering plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Eco-Ring — Model Evaluation Dashboard", fontsize=14, fontweight="bold")

    plot_predicted_vs_actual(axes[0, 0], y_test, y_pred)
    plot_residuals(          axes[0, 1], y_test, y_pred)
    plot_error_by_hour(      axes[0, 2], test_df)
    plot_error_by_ring(      axes[1, 0], test_df)
    plot_feature_importance( axes[1, 1], model)
    axes[1, 2].axis("off")   # reserved for future diagnostics

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(EVAL_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {EVAL_PLOT_PATH}")
    plt.show()

    print("\n=====================================\n")


if __name__ == "__main__":
    main()
