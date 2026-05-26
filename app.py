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

import base64
import datetime
import os
import random
import sqlite3
from collections import Counter
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from metu_rings_data import STOPS, get_rings_serving_stop, get_stops_for_ring
from ai_model import map_to_training_date, predict_for_all_active_rings, wait_estimate_for_stop
from decision_engine import DispatchEngine

# ============================================================================
# Configuration
# ============================================================================

OPERATOR_PASSWORD = os.environ.get("ECORING_ADMIN_PW", "ecoring2026")
DB_PATH           = "database.db"

RING_COLORS = {
    "yellow_red":  "#E53935",
    "light_brown": "#A1887F",
    "turquoise":   "#00ACC1",
    "orange":      "#FB8C00",
    "navy":        "#1A237E",
    "purple":      "#7B1FA2",
    "gray_day":    "#9E9E9E",
    "gray_night":  "#424242",
}
RING_DISPLAY_NAMES = {
    "yellow_red":  "Yellow-Red Ring",
    "light_brown": "Light Brown Ring",
    "turquoise":   "Turquoise Ring",
    "orange":      "Orange Ring",
    "navy":        "Navy Ring",
    "purple":      "Purple Ring",
    "gray_day":    "Gray Day Ring",
    "gray_night":  "Gray Night Ring",
}

# ============================================================================
# Timezone helper — Streamlit Cloud runs on UTC; ring schedules and display
# times must reflect Europe/Istanbul (UTC+3).
# ============================================================================

TR_TZ = ZoneInfo("Europe/Istanbul")


def now_tr() -> datetime.datetime:
    """Current time in Türkiye timezone, tzinfo stripped for compatibility
    with existing timedelta arithmetic and naive-datetime comparisons."""
    return datetime.datetime.now(TR_TZ).replace(tzinfo=None)


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
    """Record a button press in session_state (in-memory, cloud-safe)."""
    st.session_state.presses.append({
        "timestamp": now_tr(),
        "stop_id":   stop_id,
        "stop_name": stop_name,
    })
    # SQLite fallback — commented out (unreliable on Streamlit Community Cloud):
    # ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.execute(
    #         "INSERT INTO button_presses (timestamp, stop_id, stop_name) VALUES (?, ?, ?)",
    #         (ts, stop_id, stop_name),
    #     )
    #     conn.commit()


def get_recent_presses() -> pd.DataFrame:
    """All button presses in the last 30 minutes, newest first."""
    cutoff = (now_tr() - datetime.timedelta(minutes=30)).strftime(
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
    today_start = now_tr().date().isoformat() + "T00:00:00"
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
    now      = now_tr()
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

def _img_b64(path: str, mime: str) -> str:
    """Return a base64 data-URI string for the image at path."""
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def page_student() -> None:
    st.markdown("""
    <style>
        /* ── Constrain and truly center the main content block ───────────────
           Streamlit centers within (viewport - sidebar). Overriding with a
           fixed max-width + margin:auto re-centers within the full viewport. */
        section.main > div.block-container,
        .main .block-container {
            max-width: 600px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* ── Hide Streamlit fullscreen button ── */
        button[title="View fullscreen"] { display: none !important; }

        /* ── Button: METU red livery, full container width ── */
        .stButton > button {
            height: 6rem;
            width: 100% !important;
            font-size: 1.6rem !important;
            font-weight: 600 !important;
            border-radius: 999px;
            background-color: #C8102E !important;
            color: #ffffff !important;
            border: none !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            box-shadow: 0 6px 16px rgba(200, 16, 46, 0.35);
            transition: transform 0.15s ease, background-color 0.15s ease;
        }
        .stButton > button:hover {
            background-color: #A00D24 !important;
            transform: scale(1.02);
        }

        /* ── Selectbox fills its container (identical width to button) ── */
        div[data-testid="stSelectbox"] > div { width: 100% !important; }
        div[data-testid="stButton"] > button { width: 100% !important; }

        /* ── Reset Streamlit default left margins/padding on text tags ── */
        section.main h1, section.main h2, section.main h3, section.main p {
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        /* ── st.markdown wrapper divs: full-width, centered ── */
        section.main div[data-testid="stMarkdown"] {
            width: 100% !important;
            text-align: center !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── 1–4. Logo + title + subtitle + date — ONE block, ONE alignment context ─
    _date_str = now_tr().strftime("%A, %B %d · %H:%M")
    st.markdown(
        f"""
        <div style="
            width: 100%;
            text-align: center;
            margin: 0;
            padding: 2rem 0 1rem 0;
        ">
            <img src="{_img_b64('odtu_logo.png', 'image/png')}"
                 style="width:180px; height:auto; display:block; margin:0 auto 1.5rem auto;" />
            <h1 style="
                color: #2E7D32;
                font-weight: 700;
                font-size: 2.5rem;
                margin: 0 auto 0.5rem auto;
                padding: 0;
                text-align: center;
                line-height: 1.2;
            ">Eco-Ring</h1>
            <p style="
                color: #6C757D;
                font-size: 1rem;
                margin: 0 auto 0.4rem auto;
                padding: 0;
                text-align: center;
            ">Hop on the next bus</p>
            <p style="
                color: #6C757D;
                font-size: 0.9rem;
                margin: 0 auto;
                padding: 0;
                text-align: center;
            ">{_date_str}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 5. Divider ────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='margin:1rem 0; border:none; border-top:1px solid #dee2e6;'>",
        unsafe_allow_html=True,
    )

    # ── 6. Ring bus photo ─────────────────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center; margin:1rem 0;'>"
        f"<img src='{_img_b64('ring.png', 'image/png')}' "
        f"style='width:100%; height:auto; display:block; "
        f"margin:0 auto; border-radius:12px;' /></div>",
        unsafe_allow_html=True,
    )

    # ── 7. Divider ────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='margin:1rem 0; border:none; border-top:1px solid #dee2e6;'>",
        unsafe_allow_html=True,
    )

    # ── 8. "Where are you?" label ─────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center; width:100%; margin:1rem 0 0.5rem 0;'>"
        "<p style='font-weight:600; font-size:1.05rem; margin:0;'>Where are you?</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 9. Dropdown ───────────────────────────────────────────────────────────
    stop_id = st.selectbox(
        "Select your stop",
        options=list(STOPS.keys()),
        format_func=lambda sid: STOPS[sid]["name_en"],
        label_visibility="collapsed",
    )

    st.write("")  # breathing room

    # ── 10. Action button ─────────────────────────────────────────────────────
    if st.button("🚌  I'm at the stop — call a bus", use_container_width=True):
        stop_name = STOPS[stop_id]["name_en"]
        log_press(stop_id, stop_name)

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
                st.success(msg)
            elif wait_min <= 6:
                st.info(msg)
            else:
                st.warning(msg)


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
        st.stop()

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

    # ── Gather AI predictions (needed for banner, KPI cards, and ring cards) ──
    now_real: datetime.datetime = now_tr()
    mapped                      = map_to_training_date(now_real)
    ai_error: str | None        = None
    per_stop_preds: dict        = {}
    decisions: list[dict]       = []

    try:
        per_stop_preds = predict_for_all_active_rings(now_real)
        if per_stop_preds:
            decisions = DispatchEngine().predict_and_decide(now_real, per_stop_preds)
    except Exception as exc:
        ai_error = str(exc)

    # ── STATUS BANNER ────────────────────────────────────────────────────────
    active_count   = len(decisions)
    wall_clock     = now_real.strftime("%H:%M")
    mapped_str     = mapped.strftime("%Y-%m-%d %H:%M")
    banner_date    = now_real.strftime("%A, %B %d, %Y · %H:%M")

    st.markdown(f"""
    <div style="
        background: #F0F4F8;
        border: 1px solid #DDE3EA;
        border-radius: 10px;
        padding: 0.85rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    ">
        <span style="font-weight:600; color:#2E7D32;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%;
                         background:#4CAF50; margin-right:6px; vertical-align:middle;"></span>
            System Live
        </span>
        <span style="color:#444; font-weight:600;">{banner_date}</span>
        <span style="color:#444; font-weight:500;">{active_count} of 8 rings active</span>
        <span style="color:#666; font-size:0.9rem;">
            Model mapped to <strong>{mapped_str}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Load button-press data ────────────────────────────────────────────────
    _EMPTY_DF = pd.DataFrame(columns=["timestamp", "stop_id", "stop_name"])
    if demo_on:
        recent_df = make_demo_presses(n=random.randint(20, 40))
        today_df  = make_demo_presses(n=random.randint(60, 120))
    else:
        _now          = now_tr()
        _cutoff_30    = _now - datetime.timedelta(minutes=30)
        _today_start  = _now.replace(hour=0, minute=0, second=0, microsecond=0)
        _all          = st.session_state.presses

        _recent_rows  = [p for p in _all if p["timestamp"] >= _cutoff_30]
        _today_rows   = [p for p in _all if p["timestamp"] >= _today_start]

        recent_df = (
            pd.DataFrame(_recent_rows)
            .sort_values("timestamp", ascending=False)
            .reset_index(drop=True)
        ) if _recent_rows else _EMPTY_DF.copy()

        today_df = (
            pd.DataFrame(_today_rows)
            .sort_values("timestamp", ascending=False)
            .reset_index(drop=True)
        ) if _today_rows else _EMPTY_DF.copy()

        # SQLite fallback — commented out:
        # recent_df = get_recent_presses()
        # today_df  = get_today_presses()

    # ── KPI CARDS ────────────────────────────────────────────────────────────
    total_today = len(today_df)

    if not today_df.empty:
        top_stop_id   = today_df["stop_id"].value_counts().index[0]
        busiest_label = STOPS.get(top_stop_id, {}).get("name_en", top_stop_id)
    else:
        busiest_label = "—"

    bus_capacity = 60
    if decisions:
        avg_occ     = sum(d["predicted_total"] for d in decisions) / len(decisions) / bus_capacity * 100
        avg_occ_str = f"{avg_occ:.1f}%"
    else:
        avg_occ_str = "—"

    st.markdown("""
    <style>
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
    }
    .kpi-number {
        font-size: 2rem;
        font-weight: 700;
        color: #003366;
        line-height: 1.2;
        word-break: break-word;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-number">{total_today}</div>
            <div class="kpi-label">Total presses today</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-number" style="font-size:1.25rem;">{busiest_label}</div>
            <div class="kpi-label">Busiest stop today</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-number">{avg_occ_str}</div>
            <div class="kpi-label">Avg. estimated occupancy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.75rem;'></div>", unsafe_allow_html=True)

    # ── ACTIVE RINGS — LIVE STATUS ────────────────────────────────────────────
    st.markdown("### Active Rings — Live Status")

    if ai_error:
        st.warning(f"AI predictions unavailable: {ai_error}")
    elif not decisions:
        st.info("No rings are currently active at this time of day.")
    else:
        REASON_TEXT = {
            "low_demand":        "Not enough passengers yet",
            "waiting":           "Waiting for more passengers",
            "threshold_met":     "Demand reached — ready to send",
            "max_wait_exceeded": "Waited too long — must send",
        }
        DECISION_ORDER = {"DISPATCH": 0, "MERGE": 1, "HOLD": 2}
        sorted_decisions = sorted(decisions, key=lambda d: DECISION_ORDER.get(d["decision"], 99))

        for d in sorted_decisions:
            ring_id    = d["ring_id"]
            color      = RING_COLORS.get(ring_id, "#888888")
            name       = RING_DISPLAY_NAMES.get(ring_id, ring_id.replace("_", " ").title())
            decision   = d["decision"]
            reason_raw = d["reason"]
            predicted  = d["predicted_total"]
            peak_stop  = d["predicted_max_stop"]
            n_stops    = len(per_stop_preds.get(ring_id, {}))
            peak_name  = STOPS.get(peak_stop, {}).get("name_en", peak_stop) if peak_stop else "—"

            if reason_raw.startswith("merge_with_"):
                partner_id   = reason_raw.replace("merge_with_", "")
                partner_name = RING_DISPLAY_NAMES.get(partner_id, partner_id.replace("_", " ").title())
                reason_text  = f"Combine with {partner_name}"
            else:
                reason_text = REASON_TEXT.get(reason_raw, reason_raw)

            ring_row = f"""
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.35rem;">
                    <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
                                 background:{color}; flex-shrink:0;"></span>
                    <strong style="font-size:1rem;">{name}</strong>
                </div>"""
            details_rows = f"""
                <div style="color:#555; font-size:0.9rem; margin-bottom:0.2rem;">
                    {n_stops} active stops &nbsp;&middot;&nbsp; predicted demand: <strong>{predicted}</strong>
                </div>
                <div style="color:#555; font-size:0.9rem; margin-bottom:0.2rem;">
                    Peak stop: <strong>{peak_stop}</strong> &mdash; {peak_name}
                </div>
                <div style="color:#777; font-size:0.85rem;">
                    Reason: {reason_text}
                </div>"""

            if decision == "DISPATCH":
                card_html = f"""
                <div style="
                    border-left: 6px solid #C8102E;
                    background: #FFEBEE;
                    border-radius: 8px;
                    padding: 1.1rem 1.3rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 10px rgba(200,16,46,0.10);
                ">
                    <div style="text-align:center; margin-bottom:0.75rem;">
                        <span style="
                            background: #C8102E;
                            color: white;
                            padding: 0.45rem 1.4rem;
                            border-radius: 8px;
                            font-size: 1.4rem;
                            font-weight: 700;
                            display: inline-block;
                            letter-spacing: 0.02em;
                        ">🚌 SEND BUS NOW</span>
                    </div>
                    {ring_row}
                    {details_rows}
                </div>"""

            elif decision == "MERGE":
                merge_badge = f"🔀 COMBINE RINGS WITH {partner_name.upper()}" if reason_raw.startswith("merge_with_") else "🔀 COMBINE RINGS"
                card_html = f"""
                <div style="
                    border-left: 5px solid #1976D2;
                    background: #E3F2FD;
                    border-radius: 8px;
                    padding: 1.1rem 1.3rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 2px 8px rgba(25,118,210,0.08);
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:0.5rem; flex-wrap:wrap; gap:0.4rem;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
                                         background:{color}; flex-shrink:0;"></span>
                            <strong style="font-size:1rem;">{name}</strong>
                        </div>
                        <span style="
                            background: #1976D2; color: white;
                            padding: 0.3rem 0.9rem; border-radius: 6px;
                            font-size: 0.85rem; font-weight: 600; white-space: nowrap;
                        ">{merge_badge}</span>
                    </div>
                    {details_rows}
                </div>"""

            else:  # HOLD
                card_html = f"""
                <div style="
                    border-left: 4px solid #BDBDBD;
                    background: white;
                    border-radius: 8px;
                    padding: 1rem 1.2rem;
                    margin-bottom: 1rem;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:0.45rem; flex-wrap:wrap; gap:0.4rem;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="display:inline-block; width:12px; height:12px; border-radius:50%;
                                         background:{color}; flex-shrink:0;"></span>
                            <strong style="font-size:1rem;">{name}</strong>
                        </div>
                        <span style="
                            background: #FFF3CD; color: #856404;
                            padding: 0.3rem 0.8rem; border-radius: 999px;
                            font-size: 0.85rem; font-weight: 600; white-space: nowrap;
                        ">⏸️ WAIT</span>
                    </div>
                    {details_rows}
                </div>"""

            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander("View route stops"):
                for sid in get_stops_for_ring(ring_id):
                    st.write(f"**{sid}** — {STOPS[sid]['name_en']}")

    # ── INACTIVE RINGS ────────────────────────────────────────────────────────
    active_ring_ids = {d["ring_id"] for d in decisions}
    inactive_rings  = [
        RING_DISPLAY_NAMES.get(rid, rid.replace("_", " ").title())
        for rid in RING_COLORS
        if rid not in active_ring_ids
    ]

    if inactive_rings:
        inactive_names = ", ".join(inactive_rings)
        st.markdown(f"""
        <div style="
            margin-top: 0.5rem;
            padding: 0.75rem 1rem;
            background: #FAFAFA;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
        ">
            <span style="color:#666; font-size:0.9rem;">
                <strong>Currently inactive ({len(inactive_rings)} rings):</strong>
                {inactive_names}
            </span><br>
            <span style="color:#aaa; font-size:0.82rem;">
                These rings will become active during their scheduled operating hours.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── GLOSSARY ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #444;
    ">
        <strong>ℹ️ How to read this dashboard:</strong><br><br>
        🚌 <strong>SEND BUS NOW</strong> — AI predicts enough demand. Operator should dispatch a bus.<br>
        ⏸️ <strong>WAIT</strong> — Not enough passengers yet. System keeps monitoring.<br>
        🔀 <strong>COMBINE RINGS</strong> — Two rings can share one bus to save resources.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── RECENT BUTTON PRESSES TABLE ───────────────────────────────────────────
    st.markdown("### Recent button presses (last 30 min)")

    if recent_df.empty:
        st.info("No button presses yet. Tap the button on the Ring User page to log a press.")
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

    # In-memory press log — persists across page switches within the same browser session.
    if "presses" not in st.session_state:
        st.session_state.presses = []

    # init_db()  # SQLite disabled — unreliable on Streamlit Community Cloud

    st.sidebar.markdown("### Choose Dashboard")
    page = st.sidebar.radio(
        "View",
        ["Ring User", "Operator Dashboard"],
        index=0,
        label_visibility="collapsed",
    )

    if page == "Ring User":
        page_student()
    else:
        page_operator()


main()
