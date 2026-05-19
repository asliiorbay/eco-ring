# simulator.py — Eco-Ring synthetic passenger demand generator
#
# To change the random seed: open metu_rings_data.py and edit
# SIMULATION_CONFIG["random_seed"]. Re-running this script with
# the same seed always produces the exact same demand_log.csv.
#
# To enable all 8 rings: change the ACTIVE_RINGS list at the
# bottom of this file (inside main()) from 2 entries to all 8.

import datetime
import os

import numpy as np
import pandas as pd

from metu_rings_data import (
    DEMAND_MULTIPLIERS,
    RINGS,
    SEMESTER_CALENDAR,
    SIMULATION_CONFIG,
    STOPS,
    get_stops_for_ring,
    is_ring_active_on,
)

# ---------------------------------------------------------------------------
# Hourly demand profiles
# Key: stop type | Value: dict mapping hour (0–23) → expected passengers
# per 5-minute slot at baseline (before any day-level multiplier is applied).
# Linear interpolation between adjacent hours gives smooth intra-hour curves.
# ---------------------------------------------------------------------------
HOURLY_PROFILES = {
    "academic": {
        0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5,
        6: 1.0, 7: 2.0, 8: 8.0, 9: 6.0, 10: 7.0, 11: 4.0,
        12: 5.0, 13: 5.0, 14: 3.0, 15: 4.0, 16: 7.0, 17: 5.0,
        18: 2.0, 19: 1.0, 20: 1.0, 21: 0.5, 22: 0.5, 23: 0.5,
    },
    "dorm": {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
        6: 3.0, 7: 7.0, 8: 9.0, 9: 4.0, 10: 2.0, 11: 2.0,
        12: 3.0, 13: 3.0, 14: 2.0, 15: 2.0, 16: 3.0, 17: 6.0,
        18: 8.0, 19: 6.0, 20: 3.0, 21: 2.0, 22: 1.0, 23: 1.0,
    },
    "transit_hub": {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
        6: 2.0, 7: 3.0, 8: 7.0, 9: 6.0, 10: 5.0, 11: 4.0,
        12: 5.0, 13: 5.0, 14: 4.0, 15: 4.0, 16: 5.0, 17: 6.0,
        18: 4.0, 19: 3.0, 20: 2.0, 21: 1.0, 22: 1.0, 23: 1.0,
    },
    "leisure": {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
        6: 1.0, 7: 1.0, 8: 2.0, 9: 2.0, 10: 2.0, 11: 3.0,
        12: 7.0, 13: 6.0, 14: 3.0, 15: 3.0, 16: 4.0, 17: 5.0,
        18: 5.0, 19: 4.0, 20: 2.0, 21: 2.0, 22: 1.0, 23: 1.0,
    },
    "service": {
        0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5,
        6: 1.0, 7: 1.0, 8: 3.0, 9: 5.0, 10: 4.0, 11: 4.0,
        12: 4.0, 13: 3.0, 14: 3.0, 15: 3.0, 16: 3.0, 17: 2.0,
        18: 1.0, 19: 1.0, 20: 1.0, 21: 0.5, 22: 0.5, 23: 0.5,
    },
}

_SIM_START = datetime.date.fromisoformat(SIMULATION_CONFIG["simulation_start_date"])


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_hourly_profile(stop_type: str) -> dict:
    return HOURLY_PROFILES[stop_type]


def get_base_demand(stop_type: str, hour: int, minute: int) -> float:
    """Linearly interpolate between adjacent hour values for smooth demand curves."""
    profile = build_hourly_profile(stop_type)
    v0 = profile[hour]
    v1 = profile[(hour + 1) % 24]
    return v0 + (v1 - v0) * (minute / 60.0)


def _is_in_date_ranges(date: datetime.date, ranges: list) -> bool:
    for start_str, end_str in ranges:
        if datetime.date.fromisoformat(start_str) <= date <= datetime.date.fromisoformat(end_str):
            return True
    return False


def get_day_multiplier(date: datetime.date) -> float:
    """Return the demand multiplier for a calendar date.

    Priority order (highest wins): holiday > exam_week > break > weekend > 1.0
    All values come from DEMAND_MULTIPLIERS in metu_rings_data.py.
    """
    if date.isoformat() in SEMESTER_CALENDAR["holidays"]:
        return DEMAND_MULTIPLIERS["holiday"]
    if _is_in_date_ranges(date, SEMESTER_CALENDAR["exam_weeks"]):
        return DEMAND_MULTIPLIERS["exam_week"]
    if _is_in_date_ranges(date, SEMESTER_CALENDAR.get("break_weeks", [])):
        return DEMAND_MULTIPLIERS["break"]
    if date.weekday() >= 5:
        return DEMAND_MULTIPLIERS["weekend"]
    return 1.0


def generate_demand(base: float, multiplier: float, rng: np.random.Generator) -> tuple:
    """Apply multiplier and sample Poisson noise.

    Returns (expected_demand, actual_demand).
    Keeping expected_demand lets you audit what the profile contributed
    vs. what randomness added — useful when evaluating the AI model later.
    """
    expected = base * multiplier
    actual = int(rng.poisson(max(expected, 0.0)))
    return round(expected, 4), actual


def _get_week_number(date: datetime.date) -> int:
    return (date - _SIM_START).days // 7 + 1


def simulate_ring_day(ring_id: str, date: datetime.date, rng: np.random.Generator) -> list:
    """Generate all demand rows for one ring on one calendar date.

    Returns an empty list if the ring does not operate on that day.
    Handles rings that cross midnight (e.g., purple 20:30–00:30) by
    computing the correct calendar date for each slot.
    """
    if not is_ring_active_on(ring_id, date.weekday()):
        return []

    ring = RINGS[ring_id]
    stop_ids = get_stops_for_ring(ring_id)
    multiplier = get_day_multiplier(date)

    start_h, start_m = map(int, ring["start_time"].split(":"))
    end_h,   end_m   = map(int, ring["end_time"].split(":"))
    start_minutes = start_h * 60 + start_m
    end_minutes   = end_h   * 60 + end_m

    # Rings ending after midnight (e.g., purple 20:30–00:30)
    if end_minutes < start_minutes:
        end_minutes += 24 * 60

    is_exam    = _is_in_date_ranges(date, SEMESTER_CALENDAR["exam_weeks"])
    is_holiday = date.isoformat() in SEMESTER_CALENDAR["holidays"]
    is_weekend = date.weekday() >= 5
    week_num   = _get_week_number(date)

    rows = []
    for slot_offset in range(0, end_minutes - start_minutes + 1, 5):
        total_minutes  = start_minutes + slot_offset
        actual_minutes = total_minutes % (24 * 60)
        h              = actual_minutes // 60
        m              = actual_minutes % 60
        slot_of_day    = actual_minutes // 5

        slot_date = date if total_minutes < 24 * 60 else date + datetime.timedelta(days=1)
        ts = datetime.datetime.combine(slot_date, datetime.time(h, m))

        for stop_id in stop_ids:
            stop      = STOPS[stop_id]
            stop_type = stop["type"]
            base      = get_base_demand(stop_type, h, m)
            expected, actual = generate_demand(base, multiplier, rng)

            rows.append({
                "timestamp":       ts,
                "date":            slot_date,
                "day_of_week":     date.weekday(),
                "hour":            h,
                "minute":          m,
                "week_number":     week_num,
                "slot_of_day":     slot_of_day,
                "is_weekend":      is_weekend,
                "is_exam_week":    is_exam,
                "is_holiday":      is_holiday,
                "ring_id":         ring_id,
                "stop_id":         stop_id,
                "stop_name":       stop["name_en"],
                "stop_type":       stop_type,
                "base_demand":     round(base, 4),
                "expected_demand": expected,
                "actual_demand":   actual,
            })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(active_rings: list = None):
    """Run the full simulation and write data/demand_log.csv.

    active_rings — list of ring IDs to simulate.
    Default is the 2-ring prototype. To enable all 8 rings, pass:
        main(active_rings=list(RINGS.keys()))
    """
    if active_rings is None:
        active_rings = [
            "yellow_red", "light_brown", "turquoise", "orange",
            "navy", "purple", "gray_day", "gray_night",
        ]

    seed       = SIMULATION_CONFIG["random_seed"]
    rng        = np.random.default_rng(seed)
    start_date = _SIM_START
    n_days     = SIMULATION_CONFIG["simulation_days"]

    all_rows = []
    for day_offset in range(n_days):
        date = start_date + datetime.timedelta(days=day_offset)
        for ring_id in active_rings:
            all_rows.extend(simulate_ring_day(ring_id, date, rng))

    df = pd.DataFrame(all_rows)

    os.makedirs("data", exist_ok=True)
    output_path = "data/demand_log.csv"
    df.to_csv(output_path, index=False)

    file_kb = os.path.getsize(output_path) / 1024
    s = df["actual_demand"]

    print()
    print("=== Eco-Ring Simulation Summary ===")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Date range       : {df['date'].min()} → {df['date'].max()}")
    print(f"  Active rings     : {', '.join(active_rings)}")
    print(f"  Output file      : {output_path} ({file_kb:.1f} KB)")
    print(f"  actual_demand    :")
    print(f"    mean = {s.mean():.2f}")
    print(f"    min  = {int(s.min())}")
    print(f"    max  = {int(s.max())}")
    print(f"    std  = {s.std():.2f}")
    print("===================================")
    print()

    return df


if __name__ == "__main__":
    main()
