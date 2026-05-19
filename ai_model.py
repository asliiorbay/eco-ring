# ai_model.py — Eco-Ring AI inference module  (Aşama 2)
#
# Exposes three functions to app.py:
#   predict_for_stop(stop_id, when)        → float  demand at one stop
#   predict_for_all_active_rings(when)     → dict   {ring_id: {stop_id: demand}}
#   wait_estimate_for_stop(stop_id, when)  → int    estimated minutes to next bus
#
# Caching strategy:
#   Model loaded once per server process  (@st.cache_resource)
#   Predictions cached per 5-minute slot  (@st.cache_data, ttl=300)
#   The cache key is the rounded slot timestamp — predictions are reused
#   across Streamlit reruns within the same 5-minute slot and refreshed
#   automatically at each new slot boundary.

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from metu_rings_data import (
    RINGS, STOPS,
    get_rings_serving_stop, get_stops_for_ring, is_ring_active_on,
)
from decision_engine import DISPATCH_CONFIG

# ============================================================================
# Constants
# ============================================================================

MODEL_PATH     = "model/demand_model.lgb"
TRAINING_START = datetime.date(2025, 9, 15)   # Monday — first day of training data

# Category lists in the exact sorted order produced by pandas astype("category")
# on the full training CSV.  Verified against demand_log.csv — do not change
# without retraining the model.
ALL_STOP_IDS = [
    "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20",
    "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28",
]
ALL_RING_IDS = [
    "gray_day", "gray_night", "light_brown", "navy",
    "orange", "purple", "turquoise", "yellow_red",
]
ALL_STOP_TYPES = ["academic", "dorm", "leisure", "service", "transit_hub"]

FEATURES = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "slot_of_day", "week_number",
    "is_weekend", "is_exam_week", "is_holiday",
    "stop_id", "stop_type", "ring_id",
]


# ============================================================================
# Time helpers
# ============================================================================

def _round_to_slot(dt: datetime.datetime) -> datetime.datetime:
    """Round down to the nearest 5-minute boundary (ensures stable cache keys)."""
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def map_to_training_date(now: datetime.datetime) -> datetime.datetime:
    """Map any wall-clock datetime into the training window.

    The model's dominant signals are hour, minute, and day-of-week — not the
    literal calendar date.  To generate live predictions we preserve the real
    hour, minute, and weekday, then anchor to the first matching weekday in
    the training window (2025-09-15 Mon → 2025-09-21 Sun).

    Example: a real Tuesday at 14:30  →  2025-09-16 14:30

    Approximation: is_exam_week and is_holiday are always 0 for the mapped
    date (the first week of the semester has neither).  This is acceptable
    because the model's occupancy signal is dominated by time-of-day and
    day-of-week, not by calendar events.
    """
    offset      = (now.weekday() - TRAINING_START.weekday()) % 7
    mapped_date = TRAINING_START + datetime.timedelta(days=offset)
    return datetime.datetime(
        mapped_date.year, mapped_date.month, mapped_date.day,
        now.hour, now.minute, 0,
    )


def _is_ring_active(ring_id: str, when: datetime.datetime) -> bool:
    """Return True if ring_id is scheduled to operate at `when`.

    Mirrors decision_engine._is_active() — kept here as a local copy to
    avoid importing a private helper from another module.
    """
    if not is_ring_active_on(ring_id, when.weekday()):
        return False
    ring = RINGS[ring_id]
    sh, sm = map(int, ring["start_time"].split(":"))
    eh, em = map(int, ring["end_time"].split(":"))
    t, start, end = when.time(), datetime.time(sh, sm), datetime.time(eh, em)
    if end < start:                    # midnight-crossing window (e.g. purple)
        return t >= start or t <= end
    return start <= t <= end


# ============================================================================
# Model loading — one load per Streamlit server process
# ============================================================================

@st.cache_resource
def _load_model():
    """Load the LightGBM Booster once and keep it in memory."""
    import lightgbm as lgb
    return lgb.Booster(model_file=MODEL_PATH)


# ============================================================================
# Feature construction
# ============================================================================

def _build_rows(pairs: list[tuple], mapped_dt: datetime.datetime) -> pd.DataFrame:
    """Build a batch feature DataFrame for a list of (ring_id, stop_id) pairs.

    All time-derived scalars are computed once from mapped_dt and broadcast
    across every row — one model.predict() call covers the full batch.
    """
    hour        = mapped_dt.hour
    dow         = mapped_dt.weekday()
    slot_of_day = (hour * 60 + mapped_dt.minute) // 5
    week_number = int(mapped_dt.isocalendar().week)
    is_weekend  = int(dow >= 5)

    rows = [
        {
            "hour":         hour,
            "hour_sin":     np.sin(2 * np.pi * hour / 24),
            "hour_cos":     np.cos(2 * np.pi * hour / 24),
            "day_of_week":  dow,
            "dow_sin":      np.sin(2 * np.pi * dow / 7),
            "dow_cos":      np.cos(2 * np.pi * dow / 7),
            "slot_of_day":  slot_of_day,
            "week_number":  week_number,
            "is_weekend":   is_weekend,
            "is_exam_week": 0,
            "is_holiday":   0,
            "stop_id":      stop_id,
            "stop_type":    STOPS[stop_id]["type"],
            "ring_id":      ring_id,
        }
        for ring_id, stop_id in pairs
    ]

    df = pd.DataFrame(rows)
    df["stop_id"]   = pd.Categorical(df["stop_id"],   categories=ALL_STOP_IDS)
    df["stop_type"] = pd.Categorical(df["stop_type"], categories=ALL_STOP_TYPES)
    df["ring_id"]   = pd.Categorical(df["ring_id"],   categories=ALL_RING_IDS)
    return df[FEATURES]


# ============================================================================
# Cached prediction worker
# ============================================================================

@st.cache_data(ttl=300)
def _predict_cached(when_rounded: datetime.datetime) -> dict:
    """Predict demand for all active (ring, stop) pairs at the given slot.

    Keyed by the 5-minute-rounded timestamp; Streamlit refreshes the cache
    automatically after 300 s (one slot boundary).
    Returns {ring_id: {stop_id: predicted_demand}} for active rings only.
    Inactive rings are omitted entirely — an empty dict means no rings are
    running at this time.
    """
    mapped_dt = map_to_training_date(when_rounded)

    pairs = [
        (ring_id, stop_id)
        for ring_id in RINGS
        if _is_ring_active(ring_id, when_rounded)
        for stop_id in get_stops_for_ring(ring_id)
    ]
    if not pairs:
        return {}

    df    = _build_rows(pairs, mapped_dt)
    preds = np.clip(_load_model().predict(df), 0, None)

    result: dict = {}
    for (ring_id, stop_id), pred in zip(pairs, preds):
        result.setdefault(ring_id, {})[stop_id] = float(pred)
    return result


# ============================================================================
# Public API
# ============================================================================

def predict_for_all_active_rings(when: datetime.datetime | None = None) -> dict:
    """Return {ring_id: {stop_id: predicted_demand}} for all currently active rings."""
    return _predict_cached(_round_to_slot(when or datetime.datetime.now()))


def predict_for_stop(stop_id: str, when: datetime.datetime | None = None) -> float:
    """Return the predicted demand for a single stop across all its active rings."""
    all_preds = _predict_cached(_round_to_slot(when or datetime.datetime.now()))
    return sum(
        all_preds.get(ring_id, {}).get(stop_id, 0.0)
        for ring_id in get_rings_serving_stop(stop_id)
    )


def wait_estimate_for_stop(stop_id: str, when: datetime.datetime | None = None) -> int:
    """Estimate minutes to the next bus for a student at stop_id.

    Returns the minimum estimated wait across all rings serving this stop.
    Minimum is correct: a student boards whichever bus arrives first — the
    earliest dispatch wins, not the average.  This mirrors how ride-hailing
    apps show the nearest vehicle rather than the mean of all vehicles nearby.

    Demand-to-wait mapping (derived from DISPATCH_CONFIG thresholds):
      ring total >= dispatch_threshold (39)  →  2 min  (dispatch imminent)
      ring total >= merge_threshold    (15)  →  4 min  (merge or short hold)
      ring total <  merge_threshold          →  7 min  (low demand, HOLD likely)

    Capped at max_wait_minutes (8).
    """
    cfg       = DISPATCH_CONFIG
    all_preds = _predict_cached(_round_to_slot(when or datetime.datetime.now()))
    min_wait  = cfg["max_wait_minutes"]   # pessimistic starting point

    for ring_id in get_rings_serving_stop(stop_id):
        stop_preds = all_preds.get(ring_id, {})
        if not stop_preds:
            continue
        ring_total = sum(stop_preds.values())

        if ring_total >= cfg["dispatch_threshold"]:
            bracket = 2
        elif ring_total >= cfg["merge_threshold"]:
            bracket = 4
        else:
            bracket = 7

        min_wait = min(min_wait, bracket)

    return min_wait
