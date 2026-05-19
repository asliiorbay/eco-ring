# simulate_decisions.py — Eco-Ring dispatch simulation over the held-out test set
#
# Outputs:
#   data/decision_log.csv — per-slot, per-ring decisions with reason and wait columns
#
# Run: python simulate_decisions.py

import numpy as np
import lightgbm as lgb

from decision_engine import DISPATCH_CONFIG, DispatchEngine
from train_model import FEATURES, TRAIN_RATIO, load_and_prepare, time_split

CSV_PATH    = "data/demand_log.csv"
MODEL_PATH  = "model/demand_model.lgb"
OUTPUT_PATH = "data/decision_log.csv"


# ============================================================================
# Helpers
# ============================================================================

def build_per_stop_preds(slot) -> dict:
    """Reshape one timestamp's rows into {ring_id: {stop_id: predicted_demand}}."""
    result = {}
    for ring_id, group in slot.groupby("ring_id", observed=True):
        result[str(ring_id)] = dict(
            zip(group["stop_id"].astype(str), group["_pred"])
        )
    return result


def print_kpi_summary(log_df) -> None:
    cfg        = DISPATCH_CONFIG
    dispatched = log_df[log_df["decision"] == "DISPATCH"]
    n_dispatch = len(dispatched)
    n_hold     = (log_df["decision"] == "HOLD").sum()
    n_merge    = (log_df["decision"] == "MERGE").sum()

    avg_wait     = dispatched["wait_minutes_before_dispatch"].mean() if n_dispatch else 0.0
    occ_rate     = (dispatched["predicted_total"] / cfg["bus_capacity"]).mean() if n_dispatch else 0.0
    near_empty_n = (dispatched["predicted_total"] < cfg["near_empty_threshold"]).sum()
    near_empty_p = near_empty_n / n_dispatch * 100 if n_dispatch else 0.0

    wait_ok = "PASS" if avg_wait     <= cfg["max_wait_minutes"]  else "FAIL"
    occ_ok  = "PASS" if occ_rate     >= cfg["min_occupancy_rate"] else "FAIL"
    nem_ok  = "PASS" if near_empty_p <  10                        else "FAIL"

    print("\n" + "=" * 56)
    print("  KPI SUMMARY")
    print("=" * 56)
    print(f"  Avg wait before DISPATCH  : {avg_wait:5.1f} min"
          f"   (target ≤ {cfg['max_wait_minutes']} min)     [{wait_ok}]")
    print(f"  Estimated occupancy rate  : {occ_rate:5.1%}"
          f"      (target ≥ {cfg['min_occupancy_rate']:.0%})            [{occ_ok}]")
    print(f"  Near-empty trip rate      : {near_empty_p:5.1f}%"
          f"      (target < 10%)             [{nem_ok}]")
    print(f"  Decision counts           : "
          f"DISPATCH={n_dispatch}  HOLD={n_hold}  MERGE={n_merge}")
    print("=" * 56 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n=== Eco-Ring — Decision Simulation ===\n")

    # 1. Load and prepare data (same preprocessing as train_model.py)
    print("[1/5] Loading and preparing data...")
    df = load_and_prepare(CSV_PATH)
    print(f"    {len(df):,} rows loaded")

    # 2. Replicate the exact 80/20 time-based split used during training
    print("\n[2/5] Replicating time-based train/test split...")
    _, test_df = time_split(df, TRAIN_RATIO)

    # 3. Load the saved LightGBM model
    print("\n[3/5] Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"    Loaded: {MODEL_PATH}")

    # 4. Predict all test rows in one batch (fast: no Python loop over rows)
    print("\n[4/5] Predicting per-(ring, stop) demand for test set...")
    preds = np.clip(model.predict(test_df[FEATURES]), 0, None)
    test_df = test_df.copy()
    test_df["_pred"] = preds
    print(f"    {len(test_df):,} predictions generated")

    # 5. Simulate dispatch decisions slot by slot in chronological order
    print("\n[5/5] Running dispatch simulation (5-min slots)...")
    engine  = DispatchEngine()
    n_slots = 0

    for ts, slot in test_df.groupby("timestamp", sort=True):
        per_stop_preds = build_per_stop_preds(slot)
        engine.predict_and_decide(ts, per_stop_preds)
        n_slots += 1

    print(f"    Processed {n_slots:,} slots")

    # Save decision log
    log_df = engine.get_log()
    log_df.to_csv(OUTPUT_PATH, index=False)
    print(f"    Decision log saved: {OUTPUT_PATH}  ({len(log_df):,} rows)")

    # Print KPI summary against scope targets
    print_kpi_summary(log_df)


if __name__ == "__main__":
    main()
