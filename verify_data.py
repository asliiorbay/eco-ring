# verify_data.py — Eco-Ring demand data validation
#
# Run AFTER simulator.py:
#   python verify_data.py
#
# Produces 6 plots and saves them to data/demand_validation.png

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

CSV_PATH = "data/demand_log.csv"
PNG_PATH = "data/demand_validation.png"

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: '{path}' not found. Run simulator.py first.")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["timestamp", "date"])

    # CSV stores booleans as the strings "True"/"False" — convert back.
    for col in ["is_weekend", "is_exam_week", "is_holiday"]:
        df[col] = df[col].astype(str).map({"True": True, "False": False})

    print(f"Loaded {len(df):,} rows  |  "
          f"rings: {sorted(df['ring_id'].unique())}  |  "
          f"stops: {df['stop_id'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_hourly(ax, df):
    hourly = df.groupby("hour")["actual_demand"].mean()
    ax.bar(hourly.index, hourly.values, color="steelblue", width=0.8)
    ax.set_title("Mean Demand by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean Passengers / 5-min Slot")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.set_xlim(-0.5, 23.5)


def plot_day_of_week(ax, df):
    daily = (
        df.groupby("day_of_week")["actual_demand"]
        .mean()
        .reindex(range(7), fill_value=0)
    )
    colors = ["steelblue"] * 5 + ["salmon"] * 2
    ax.bar(range(7), daily.values, color=colors, width=0.7)
    ax.set_title("Mean Demand by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Passengers / 5-min Slot")
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_LABELS)


def plot_weekday_vs_weekend(ax, df):
    wk = df.groupby("is_weekend")["actual_demand"].mean()
    weekday_val = wk.get(False, 0.0)
    weekend_val = wk.get(True, 0.0)
    labels = ["Weekday", "Weekend"]
    values = [weekday_val, weekend_val]
    bars = ax.bar(labels, values, color=["steelblue", "salmon"], width=0.45)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(values) * 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("Mean Demand: Weekday vs Weekend")
    ax.set_ylabel("Mean Passengers / 5-min Slot")
    ax.set_ylim(0, max(values) * 1.2)


def plot_exam_vs_normal(ax, df):
    ew = df.groupby("is_exam_week")["actual_demand"].mean()
    normal_val = ew.get(False, 0.0)
    exam_val   = ew.get(True,  0.0)
    labels = ["Normal Week", "Exam Week"]
    values = [normal_val, exam_val]
    bars = ax.bar(labels, values, color=["steelblue", "darkorange"], width=0.45)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(values) * 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("Mean Demand: Exam Week vs Normal")
    ax.set_ylabel("Mean Passengers / 5-min Slot")
    ax.set_ylim(0, max(values) * 1.2)


def plot_total_by_ring(ax, df):
    ring_totals = (
        df.groupby("ring_id")["actual_demand"]
        .sum()
        .sort_values(ascending=False)
    )
    bars = ax.bar(ring_totals.index, ring_totals.values, color="teal", width=0.6)
    peak = ring_totals.max()
    for bar, val in zip(bars, ring_totals.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + peak * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_title("Total Demand by Ring (All Days)")
    ax.set_xlabel("Ring")
    ax.set_ylabel("Total Passengers")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, peak * 1.15)


def plot_daily_totals(ax, df):
    # Shows 90-day pattern — exam week spikes and weekend dips are visible here.
    daily = df.groupby("date")["actual_demand"].sum()
    ax.plot(daily.index, daily.values, color="steelblue", linewidth=0.9)
    ax.fill_between(daily.index, daily.values, alpha=0.18, color="steelblue")
    ax.set_title("Total Daily Demand — 90-Day View")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Passengers")
    ax.tick_params(axis="x", rotation=30)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = load_data(CSV_PATH)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Eco-Ring — Demand Validation Dashboard", fontsize=14, fontweight="bold")

    plot_hourly(           axes[0, 0], df)
    plot_day_of_week(      axes[0, 1], df)
    plot_weekday_vs_weekend(axes[0, 2], df)
    plot_exam_vs_normal(   axes[1, 0], df)
    plot_total_by_ring(    axes[1, 1], df)
    plot_daily_totals(     axes[1, 2], df)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
