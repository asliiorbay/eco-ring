# decision_engine.py — Eco-Ring demand-responsive dispatch decision engine
#
# Outputs (via simulate_decisions.py):
#   data/decision_log.csv — per-slot, per-ring dispatch decisions

import datetime

import pandas as pd

from metu_rings_data import RINGS, get_stops_for_ring, is_ring_active_on

# ============================================================================
# DISPATCH_CONFIG — all tunable thresholds in one place
# ============================================================================

DISPATCH_CONFIG = {
    "bus_capacity":           60,    # ODTÜ standard ring bus capacity
    "min_occupancy_rate":     0.65,  # KPI: ≥65% occupancy target
    "dispatch_threshold":     39,    # = bus_capacity × min_occupancy_rate
    "merge_threshold":        15,    # roughly half of dispatch — two rings together hit threshold
    "near_empty_threshold":    6,    # = bus_capacity × 0.10 (KPI: <10% near-empty trips)
    "max_wait_minutes":        8,    # KPI: avg wait ≤ 8 min
    "prediction_horizon_min": 10,    # AI predicts next 10 min (scope)
    "time_granularity_min":    5,    # slot width; HOLD increments wait counter by this value
}


# ============================================================================
# Helpers
# ============================================================================

def _is_active(ring_id: str, current_time: datetime.datetime) -> bool:
    """Return True if ring_id is scheduled to operate at current_time."""
    if not is_ring_active_on(ring_id, current_time.weekday()):
        return False

    ring = RINGS[ring_id]
    sh, sm = map(int, ring["start_time"].split(":"))
    eh, em = map(int, ring["end_time"].split(":"))
    t     = current_time.time()
    start = datetime.time(sh, sm)
    end   = datetime.time(eh, em)

    # Midnight-crossing window (e.g. purple: 20:30 → 00:30)
    if end < start:
        return t >= start or t <= end
    return start <= t <= end


def _find_merge_partner(ring_id: str, demand: dict, excluded: set) -> str | None:
    """Return the first eligible merge partner for ring_id, or None.

    A valid partner must:
      - not be in excluded (already dispatched or merged this slot)
      - have demand in [merge_threshold, dispatch_threshold]
      - share at least one common stop with ring_id
    """
    cfg     = DISPATCH_CONFIG
    stops_a = set(get_stops_for_ring(ring_id))

    for other_id, other_demand in demand.items():
        if other_id == ring_id or other_id in excluded:
            continue
        if not (cfg["merge_threshold"] <= other_demand <= cfg["dispatch_threshold"]):
            continue
        if stops_a & set(get_stops_for_ring(other_id)):
            return other_id

    return None


# ============================================================================
# DispatchEngine
# ============================================================================

class DispatchEngine:
    """Stateful per-ring dispatch decision maker.

    Call predict_and_decide() once per 5-minute slot in chronological order.
    Retrieve the full decision log with get_log().
    """

    def __init__(self):
        self.wait_minutes: dict[str, int] = {rid: 0 for rid in RINGS}
        self._log: list[dict] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def predict_and_decide(
        self,
        current_time: datetime.datetime,
        per_stop_preds: dict,
    ) -> list[dict]:
        """Compute one decision per active ring for the given 5-minute slot.

        Parameters
        ----------
        current_time   : timestamp of the slot being evaluated
        per_stop_preds : {ring_id: {stop_id: predicted_demand_float}}

        Returns
        -------
        List of decision dicts for active rings only (inactive rings omitted).
        """
        cfg = DISPATCH_CONFIG

        # -- Aggregate per-stop predictions → per-ring totals (active rings only) --
        active_demand: dict[str, float] = {}
        peak_stop: dict[str, str]       = {}

        for ring_id in RINGS:
            if not _is_active(ring_id, current_time):
                continue
            stops = per_stop_preds.get(ring_id, {})
            if stops:
                active_demand[ring_id] = sum(stops.values())
                peak_stop[ring_id]     = max(stops, key=stops.get)
            else:
                active_demand[ring_id] = 0.0
                peak_stop[ring_id]     = ""

        decisions: dict[str, dict] = {}

        # -- Pass 1: WAIT_OVERRIDE and threshold-based DISPATCH --
        for ring_id, total in active_demand.items():
            wait = self.wait_minutes[ring_id]

            if wait >= cfg["max_wait_minutes"]:
                # Priority 1: force-dispatch after max wait regardless of demand
                decisions[ring_id] = self._record(
                    current_time, ring_id, total, peak_stop[ring_id],
                    "DISPATCH", "max_wait_exceeded", wait,
                )
                self.wait_minutes[ring_id] = 0

            elif total >= cfg["dispatch_threshold"]:
                # Priority 2: demand meets the occupancy threshold
                decisions[ring_id] = self._record(
                    current_time, ring_id, total, peak_stop[ring_id],
                    "DISPATCH", "threshold_met", wait,
                )
                self.wait_minutes[ring_id] = 0

        # -- Pass 2: MERGE for eligible undecided rings, then HOLD for the rest --
        already_merged: set[str] = set()

        for ring_id, total in active_demand.items():
            if ring_id in decisions or ring_id in already_merged:
                continue

            merged = False

            if cfg["merge_threshold"] <= total <= cfg["dispatch_threshold"]:
                # Priority 3: look for a merge partner (excludes dispatched + already merged)
                excluded = set(decisions.keys()) | already_merged
                partner  = _find_merge_partner(ring_id, active_demand, excluded)

                if partner is not None:
                    wait_a = self.wait_minutes[ring_id]
                    wait_b = self.wait_minutes[partner]

                    decisions[ring_id] = self._record(
                        current_time, ring_id, total, peak_stop[ring_id],
                        "MERGE", f"merge_with_{partner}", wait_a,
                    )
                    decisions[partner] = self._record(
                        current_time, partner, active_demand[partner], peak_stop[partner],
                        "MERGE", f"merge_with_{ring_id}", wait_b,
                    )
                    # Reset both counters in the same pass before returning
                    self.wait_minutes[ring_id] = 0
                    self.wait_minutes[partner] = 0
                    already_merged.add(ring_id)
                    already_merged.add(partner)
                    merged = True

            if not merged:
                # Priority 4: HOLD — distinguish why we're waiting
                reason = (
                    "low_demand" if total < cfg["merge_threshold"]
                    else "waiting"      # demand in merge range but no partner found this slot
                )
                decisions[ring_id] = self._record(
                    current_time, ring_id, total, peak_stop[ring_id],
                    "HOLD", reason, self.wait_minutes[ring_id],
                )
                self.wait_minutes[ring_id] += cfg["time_granularity_min"]

        result = list(decisions.values())
        self._log.extend(result)
        return result

    # ------------------------------------------------------------------
    # Log access
    # ------------------------------------------------------------------

    def get_log(self) -> pd.DataFrame:
        """Return the full decision log as a DataFrame."""
        return pd.DataFrame(self._log)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _record(
        self,
        timestamp: datetime.datetime,
        ring_id: str,
        predicted_total: float,
        predicted_max_stop: str,
        decision: str,
        reason: str,
        wait_minutes_before_dispatch: int,
    ) -> dict:
        return {
            "timestamp":                    timestamp,
            "ring_id":                      ring_id,
            # Sum of predicted boardings across all stops on the route cycle,
            # not simultaneous on-board passenger count.
            "predicted_total":              round(predicted_total, 2),
            "predicted_max_stop":           predicted_max_stop,
            "decision":                     decision,
            "reason":                       reason,
            "wait_minutes_before_dispatch": wait_minutes_before_dispatch,
        }


# ============================================================================
# __main__ — run simulate_decisions.py for the full simulation
# ============================================================================

if __name__ == "__main__":
    print("decision_engine.py defines DispatchEngine.")
    print("Run simulate_decisions.py to execute a full simulation over the test set.")
