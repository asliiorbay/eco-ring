# app.py — Eco-Ring Streamlit webapp  (Aşama 2: AI integration)
#
# ─────────────────────────────────────────────────────────────────
# HOW TO TEST FROM A PHONE (capstone demo):
#   1. Get your laptop's local IP:
#        run in Terminal →  ipconfig getifaddr en0
#   2. Start the app:
#        streamlit run app.py
#   3. Streamlit prints a "Network URL", e.g. http://192.168.x.x:8501
#   4. Open that URL in phone Safari or Chrome
#      (laptop and phone must be on the same WiFi network)
#   5. iOS Safari  → Share button → "Add to Home Screen" → native-app feel
#      Android Chrome → ⋮ menu  → "Add to Home Screen"
# ─────────────────────────────────────────────────────────────────
# OPERATOR PASSWORD:
#   Default fallback : ecoring2026   (works out of the box, no setup needed)
#   To override      : export ECORING_ADMIN_PW="yourpassword"
#                      streamlit run app.py
#   ⚠ Prototype-grade access control only — plaintext env var, not hashed.
#      Suitable for capstone demo; not for production deployment.
# ─────────────────────────────────────────────────────────────────

import datetime
import os
import random
import sqlite3
from collections import Counter

import pandas as pd
import streamlit as st

from metu_rings_data import STOPS, get_rings_serving_stop
from ai_model import map_to_training_date, predict_for_all_active_rings, wait_estimate_for_stop
from decision_engine import DispatchEngine

# ============================================================================
# Configuration
# ============================================================================

OPERATOR_PASSWORD = os.environ.get("ECORING_ADMIN_PW", "ecoring2026")
DB_PATH           = "database.db"

# ============================================================================
# Database helpers
# Each function opens and closes its own connection — safe for SQLite under
# Streamlit's multi-threaded serving model.
# ============================================================================

def init_db() -> None:
    """Create the button_presses table if it doesn't exist. Called once at startup."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS button_presses (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                stop_id   TEXT    NOT NULL,
                stop_name TEXT    NOT NULL
            )
        """)
        conn.commit()


def log_press(stop_id: str, stop_name: str) -> None:
    """Insert one button-press row with the current local timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO button_presses (timestamp, stop_id, stop_name) VALUES (?, ?, ?)",
            (ts, stop_id, stop_name),
        )
        conn.commit()


def get_recent_presses() -> pd.DataFrame:
    """All button presses in the last 30 minutes, newest first."""
    cutoff = (datetime.datetime.now() - datetime.timedelta(minutes=30)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, timestamp, stop_id, stop_name FROM button_presses "
            "WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 500",
            conn,
            params=(cutoff,),
        )
    return df


def get_today_presses() -> pd.DataFrame:
    """All button presses since midnight today."""
    today_start = datetime.date.today().isoformat() + "T00:00:00"
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, timestamp, stop_id, stop_name FROM button_presses "
            "WHERE timestamp >= ? ORDER BY timestamp DESC",
            conn,
            params=(today_start,),
        )
    return df


# ============================================================================
# Demo mode — in-memory only, never touches the real database
# ============================================================================

def make_demo_presses(n: int = 30) -> pd.DataFrame:
    """Generate synthetic button presses for testing. The real DB is untouched."""
    stop_ids = list(STOPS.keys())
    now      = datetime.datetime.now()
    rows     = []
    for i in range(n):
        sid = random.choice(stop_ids)
        rows.append({
            "id":        i + 1,
            "timestamp": (
                now - datetime.timedelta(minutes=random.uniform(0, 29))
            ).strftime("%Y-%m-%dT%H:%M:%S"),
            "stop_id":   sid,
            "stop_name": STOPS[sid]["name_en"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("timestamp", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================================
# Page 1 — Student PWA
# ============================================================================

def page_student() -> None:
    # Large, touch-friendly button for mobile — no color override, uses theme primary
    st.markdown("""
    <style>
        .stButton > button {
            height: 4rem;
            width: 100%;
            font-size: 1.25rem;
            font-weight: 600;
            border-radius: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Eco-Ring")
    st.subheader("Hop on the next bus")
    st.markdown("---")

    # Step 1 — stop selection
    st.markdown("**Step 1 — Where are you?**")
    stop_id = st.selectbox(
        "Select your stop",
        options=list(STOPS.keys()),
        format_func=lambda sid: STOPS[sid]["name_en"],
        label_visibility="collapsed",
    )

    st.write("")  # vertical breathing room

    # Step 2 — call button
    st.markdown("**Step 2 — Request a bus**")
    if st.button("🚌  I'm at the stop — call a bus"):
        stop_name = STOPS[stop_id]["name_en"]
        log_press(stop_id, stop_name)  # DB write always first, independent of AI

        try:
            wait_min = wait_estimate_for_stop(stop_id)
        except Exception:
            wait_min = None

        if wait_min is None:
            st.success(
                f"Got it! Your request is recorded.\n\n"
                f"**Stop:** {stop_name}\n\n"
                f"Estimated wait: **~5 min**"
            )
            st.caption("Live estimate temporarily unavailable.")
        else:
            msg = (
                f"Got it! Your request is recorded.\n\n"
                f"**Stop:** {stop_name}\n\n"
                f"Estimated wait: **~{wait_min} min**"
            )
            if wait_min <= 3:
                st.success(msg)    # green  — bus coming very soon
            elif wait_min <= 6:
                st.info(msg)       # blue   — moderate wait
            else:
                st.warning(msg)    # amber  — longer wait, low demand


# ============================================================================
# Page 2 — Operator Dashboard
# ============================================================================

def page_operator() -> None:
    # --- Auth gate: nothing renders until the correct password is entered ---
    if not st.session_state.get("auth", False):
        st.title("Eco-Ring — Operator Dashboard")
        pwd = st.text_input(
            "Admin password", type="password", placeholder="Enter password"
        )
        if st.button("Login"):
            if pwd == OPERATOR_PASSWORD:
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Access restricted.")
        st.stop()  # nothing below this line renders until auth passes

    # --- Authenticated dashboard ---
    st.title("Eco-Ring — Operator Dashboard")

    col_refresh, col_demo = st.columns([3, 1])
    with col_refresh:
        if st.button("↻ Refresh"):
            st.rerun()
    with col_demo:
        demo_on = st.toggle(
            "Demo mode", value=st.session_state.get("demo_mode", False)
        )
        st.session_state["demo_mode"] = demo_on

    if demo_on:
        st.warning("⚠ Demo mode ON — showing synthetic data, not live database.")

    # --- Live AI predictions (always real — unaffected by demo mode) ---
    st.markdown("### Live AI Predictions — Next 5 minutes")
    now_real = datetime.datetime.now()
    mapped   = map_to_training_date(now_real)
    st.caption(
        f"Wall clock: **{now_real.strftime('%H:%M')}** "
        f"→ mapped to **{mapped.strftime('%Y-%m-%d %H:%M')}** for the model"
    )

    try:
        per_stop_preds = predict_for_all_active_rings(now_real)
        if not per_stop_preds:
            st.info("No rings are currently active at this time of day.")
        else:
            # Fresh DispatchEngine (wait_minutes = 0) gives a demand-based
            # snapshot decision — WAIT_OVERRIDE never fires without history,
            # but DISPATCH / MERGE / HOLD reflect live demand correctly.
            decisions = DispatchEngine().predict_and_decide(now_real, per_stop_preds)
            pred_rows = [
                {
                    "ring":             d["ring_id"],
                    "active stops":     len(per_stop_preds.get(d["ring_id"], {})),
                    "predicted demand": d["predicted_total"],
                    "peak stop":        d["predicted_max_stop"],
                    "decision":         d["decision"],
                    "reason":           d["reason"],
                }
                for d in decisions
            ]
            st.dataframe(
                pd.DataFrame(pred_rows),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.warning(f"AI predictions unavailable: {exc}")

    st.markdown("---")

    # --- Load button-press data (real or synthetic) ---
    if demo_on:
        recent_df = make_demo_presses(n=random.randint(20, 40))
        today_df  = make_demo_presses(n=random.randint(60, 120))
    else:
        recent_df = get_recent_presses()
        today_df  = get_today_presses()

    # --- Summary stats ---
    st.markdown("### Summary — Today")

    total_today = len(today_df)
    if not today_df.empty:
        top_stop_id   = today_df["stop_id"].value_counts().index[0]
        busiest_label = STOPS.get(top_stop_id, {}).get("name_en", top_stop_id)
    else:
        busiest_label = "—"

    c1, c2 = st.columns(2)
    c1.metric("Total presses today", total_today)
    c2.metric("Busiest stop", busiest_label)

    # Presses per ring — one press at a shared stop counts toward all its rings
    if not today_df.empty:
        ring_counter = Counter()
        for sid in today_df["stop_id"]:
            for ring in get_rings_serving_stop(sid):
                ring_counter[ring] += 1
        ring_df = pd.DataFrame(
            ring_counter.most_common(), columns=["ring_id", "presses"]
        )
        st.markdown("**Presses per ring (today)**")
        st.dataframe(ring_df, use_container_width=True, hide_index=True)

    # --- Recent presses table ---
    st.markdown("### Recent button presses (last 30 min)")

    if recent_df.empty:
        st.info("No button presses recorded in the last 30 minutes.")
    else:
        display_df          = recent_df.copy()
        display_df["rings"] = display_df["stop_id"].apply(
            lambda sid: ", ".join(get_rings_serving_stop(sid))
        )
        st.dataframe(
            display_df[["timestamp", "stop_name", "stop_id", "rings"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(recent_df)} request(s) in the last 30 minutes.")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Eco-Ring",
        page_icon="🚌",
        layout="centered",
        initial_sidebar_state="auto",
    )
    init_db()

    page = st.sidebar.radio(
        "Navigate",
        ["Student", "Operator Dashboard"],
        index=0,
    )

    if page == "Student":
        page_student()
    else:
        page_operator()


main()
