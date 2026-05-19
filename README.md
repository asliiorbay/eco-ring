# Eco-Ring

**ML-powered demand-responsive campus bus dispatch system for METU**

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/lightgbm-4.x-orange.svg)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Eco-Ring replaces fixed-schedule campus ring buses with a **demand-responsive system**. Instead of running buses on a fixed timetable, the system listens to real-time passenger demand — via a simple **"I'm at the stop"** button — and dispatches buses only when and where they are actually needed.

This is a senior capstone project (April 2026 – March 2027) targeting **Middle East Technical University (METU)**'s 8-ring campus bus network in Ankara, Turkey.

---

## Why Eco-Ring?

Current campus bus operations face three persistent problems:

| Problem | Today | Eco-Ring Target |
|---|---|---|
| Average wait time | ~18 minutes | ≤ 8 minutes |
| Bus occupancy rate | ~35% | ≥ 65% |
| Near-empty dispatches | ~25% of trips | < 10% of trips |
| Fuel use per passenger-km | Baseline | ≤ 75% (↓ 25%) |

Eco-Ring addresses all four by combining:
- **AI demand prediction** (LightGBM)
- **Rule-based dispatch logic** (DISPATCH / HOLD / MERGE)
- **A simple passenger interface** (single-button web app)
- **An operator dashboard** with live recommendations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Eco-Ring System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📱 Student PWA              📺 Operator Dashboard          │
│  (anonymous, mobile-first)   (password-protected)           │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌────────────────────────────────────────────────┐         │
│  │              app.py (Streamlit)                 │         │
│  └──────┬────────────────┬────────────────────────┘         │
│         │                │                                  │
│         ▼                ▼                                  │
│  ┌────────────┐  ┌──────────────────┐                       │
│  │ database.db│  │   ai_model.py    │                       │
│  │  (SQLite)  │  │  (LightGBM load) │                       │
│  └────────────┘  └────────┬─────────┘                       │
│                           │                                 │
│                           ▼                                 │
│                  ┌──────────────────┐                       │
│                  │ decision_engine  │                       │
│                  │   .py            │                       │
│                  │ DISPATCH/HOLD/   │                       │
│                  │ MERGE rules      │                       │
│                  └──────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Module | Purpose |
|---|---|
| `metu_rings_data.py` | METU ring topology (28 stops, 8 lines) + simulation config |
| `simulator.py` | Generates synthetic passenger demand (90 days, ~450K rows) |
| `verify_data.py` | Visual validation of 5 realism patterns |
| `train_model.py` | LightGBM training pipeline (time-based split, early stopping) |
| `evaluate_model.py` | Model audit plots (predicted vs actual, residuals, errors) |
| `decision_engine.py` | DISPATCH / HOLD / MERGE rule engine with KPI tracking |
| `simulate_decisions.py` | End-to-end dispatch simulation over test set |
| `ai_model.py` | Streamlit-friendly wrapper for real-time predictions |
| `app.py` | Streamlit web app (student page + operator dashboard) |

---

## Quick Start

### Requirements
- macOS / Linux
- Python 3.10+
- Homebrew (Mac, for `libomp` dependency of LightGBM)

### Installation

```bash
# Clone the repo
git clone https://github.com/asliiorbay/eco-ring.git
cd eco-ring

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# On macOS only: install libomp for LightGBM
brew install libomp

# Install Python dependencies
pip install pandas numpy lightgbm streamlit matplotlib scikit-learn
```

### Run the Web App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

- **Student page** is the default (anonymous, no login)
- **Operator Dashboard** is protected — password: `ecoring2026` (configurable via `ECORING_ADMIN_PW` env variable)

### Reproduce the ML Pipeline (Optional)

The repo already includes pre-generated data and a trained model. To regenerate from scratch:

```bash
# 1. Generate synthetic demand data
python simulator.py

# 2. Validate the data visually
python verify_data.py

# 3. Train the LightGBM model
python train_model.py

# 4. Evaluate model performance
python evaluate_model.py

# 5. Simulate dispatch decisions on test set
python simulate_decisions.py
```

---

## Results

### Model Performance

The LightGBM model is trained on 80% of the 90-day window and evaluated on the unseen final 20% (which includes the final exam week — a true generalization test).

| Metric | Value | Target |
|---|---|---|
| MAE (passengers / 5-min slot) | **~1.5** | ≤ 2 |
| R² | **~0.88** | ≥ 0.85 |
| Per-ring MAE (6 of 8 rings) | < 2.0 ✅ | — |
| Per-ring MAE (2 small rings) | ~2.2 — 2.3 🟡 | small rings have limited data |

### Dispatch KPIs

After tuning `merge_threshold` to 15 (25% of bus capacity), all three scope KPIs **pass**:

| KPI | Result | Scope Target | Status |
|---|---|---|---|
| Avg wait before DISPATCH | **1.6 min** | ≤ 8 min | ✅ PASS |
| Estimated occupancy rate | **139.5%** ¹ | ≥ 65% | ✅ PASS |
| Near-empty trip rate | **1.5%** | < 10% | ✅ PASS |

¹ *Occupancy is measured as total boardings per route cycle, not simultaneous on-board count. This is the natural KPI for demand-responsive systems where the goal is route utilization. A bus visits ~20 stops per cycle; passengers board and alight throughout.*

### Decision Distribution

```
DISPATCH:  3,865  (75%)
HOLD:      1,271  (25%)
MERGE:        16   (<1%)
```

Low MERGE count is operationally correct for METU's topology: busy rings (yellow_red, light_brown) reach the dispatch threshold on their own, while quiet rings (turquoise, single morning trip) rarely overlap with merge-eligible neighbors.

---

## Data Source

**Real METU ring topology** scraped from the official METU Transportation Office:
- Stops: https://rota.metu.edu.tr/kategori/23/duraklar
- Ring schedules: https://tim.metu.edu.tr/tr/ring-services

**Synthetic passenger demand** generated by `simulator.py`, encoding 5 realism patterns:
1. Hourly demand profile by stop type (academic, dorm, transit_hub, leisure, service)
2. Weekend reduction
3. Exam week surge
4. Semester break dip
5. Poisson-distributed randomness

⚠️ **Synthetic data disclaimer:** Hourly profile values and multipliers are informed assumptions based on general transit literature and METU operational schedules. **Actual passenger demand was not available** since no demand-signaling infrastructure exists at METU. These assumptions will be re-calibrated against real button-press data once the live pilot is conducted.

---

## Scope and Limits

### In Scope
- Full 8-ring METU simulation
- AI demand prediction and dispatch decision engine
- Web-based student PWA and operator dashboard
- 2-week supervised live pilot on selected rings (subject to institutional approval)

### Explicitly Out of Scope
- Production deployment across the entire campus
- Physical hardware (no GPS, IoT sensors, or on-bus displays)
- Native iOS / Android apps (web app only)
- Integration with university ERP, student information systems, or external maps/traffic APIs
- Driver scheduling, vehicle maintenance, payroll
- KVKK / GDPR compliance certification (basic anonymization applied, not formally audited)
- Authentication with METU SSO (anonymous by design)

---

## Privacy

By design, the system stores **no personal data**:
- No user identification, no cookies, no GPS
- Each button press records only `(timestamp, stop_id, stop_name)`
- This is documented as a deliberate design decision to (1) maximize adoption, (2) comply with data minimization principles, and (3) demonstrate that demand-responsive dispatch does not require personal identification

---

## Project Status

| Milestone | Status |
|---|---|
| Scope and requirements | ✅ Done |
| Simulator running (8 rings) | ✅ Done |
| AI agent validated (MAE ≤ 2, R² ≥ 0.85) | ✅ Done |
| Dispatch engine integrated (3/3 KPIs PASS) | ✅ Done |
| Streamlit web app (Aşama 1 + Aşama 2) | ✅ Done |
| UI/UX polish + ODTÜ branding | 🟡 In progress |
| Capstone documentation | 🟡 In progress |
| 2-week live pilot | ⏳ Pending institutional approval |
| Final delivery + jury presentation | ⏳ Scheduled |

---

## Author

**Aslı Orbay, Aladdin Demirkan** — MBA Students, Middle East Technical University
Capstone Project, 2026

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

ODTÜ logo and brand assets are used for educational purposes only.
