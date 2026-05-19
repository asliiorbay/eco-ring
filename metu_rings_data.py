"""
METU Campus Ring System — Real Topology Data
============================================

Source: https://tim.metu.edu.tr/tr/ring-services (Official METU Transportation Office)
Compiled: 2026-05

This file contains the authoritative topology of the METU campus ring bus system:
- 28 unique stops across the campus
- 8 ring lines with their official routes, operating hours, and schedules
- Stop types categorized by typical demand pattern

This is the ground truth that drives the Eco-Ring simulator, AI model,
and dispatch engine. All other modules import from here.
"""

# ============================================================================
# UNIQUE STOPS — 28 unique locations across all 8 ring lines
# ============================================================================
# Stop types drive the hourly demand profile:
#   academic   → peaks around class transitions (8-9am, 12-1pm, 4-5pm)
#   dorm       → peaks morning (rush to class) and evening (return)
#   transit_hub→ broad demand all day (metro entry/exit)
#   leisure    → lunch + evening peaks (cafes, shops, sports)
#   service    → steady moderate demand (admin, services)

STOPS = {
    "S01": {"name_en": "A1 Metro Gate",                "name_tr": "A-1 Metro Kapısı",                 "type": "transit_hub"},
    "S02": {"name_en": "A2 Metro Gate",                "name_tr": "A-2 Metro Kapısı",                 "type": "transit_hub"},
    "S03": {"name_en": "College / Garages",            "name_tr": "Kolej / Garajlar",                 "type": "service"},
    "S04": {"name_en": "BOTE-Vocational School",       "name_tr": "BÖTE-MYO",                         "type": "academic"},
    "S05": {"name_en": "Faculty of Education",         "name_tr": "Eğitim Fakültesi",                 "type": "academic"},
    "S06": {"name_en": "Teknokent",                    "name_tr": "Teknokent",                        "type": "service"},
    "S07": {"name_en": "METU Sports Center",           "name_tr": "ODTÜ Spor Merkezi",                "type": "leisure"},
    "S08": {"name_en": "METU KENT Junction",           "name_tr": "ODTÜ KENT Kavşağı",                "type": "service"},
    "S09": {"name_en": "West Dormitories",             "name_tr": "Batı Yurtlar",                     "type": "dorm"},
    "S10": {"name_en": "Aerospace Engineering",        "name_tr": "Havacılık ve Uzay Mühendisliği",   "type": "academic"},
    "S11": {"name_en": "Food Engineering",             "name_tr": "Gıda Mühendisliği",                "type": "academic"},
    "S12": {"name_en": "Geological Engineering",       "name_tr": "Jeoloji Mühendisliği",             "type": "academic"},
    "S13": {"name_en": "Mechanical Engineering",       "name_tr": "Makine Mühendisliği",              "type": "academic"},
    "S14": {"name_en": "Industrial Engineering",       "name_tr": "Endüstri Mühendisliği",            "type": "academic"},
    "S15": {"name_en": "METU Nursery",                 "name_tr": "ODTÜ Yuva",                        "type": "service"},
    "S16": {"name_en": "Faculty of Architecture",      "name_tr": "Mimarlık Fakültesi",               "type": "academic"},
    "S17": {"name_en": "School of Foreign Languages",  "name_tr": "YDYO",                             "type": "academic"},
    "S18": {"name_en": "Faculty of Economics & Admin.","name_tr": "İİBF",                             "type": "academic"},
    "S19": {"name_en": "Rectorate",                    "name_tr": "Rektörlük",                        "type": "service"},
    "S20": {"name_en": "Ziraat Bank (Shopping Area)",  "name_tr": "Ziraat Bankası (Çarşı)",           "type": "leisure"},
    "S21": {"name_en": "East Dormitories",             "name_tr": "Doğu Yurtlar",                     "type": "dorm"},
    "S22": {"name_en": "Is Bank",                      "name_tr": "İş Bankası",                       "type": "leisure"},
    "S23": {"name_en": "Culture & Congress Center",    "name_tr": "Kültür Kongre Merkezi",            "type": "leisure"},
    "S24": {"name_en": "Civil Engineering",            "name_tr": "İnşaat Mühendisliği",              "type": "academic"},
    "S25": {"name_en": "Chemical Engineering",         "name_tr": "Kimya Mühendisliği",               "type": "academic"},
    "S26": {"name_en": "Teknokent Satellite Parking",  "name_tr": "Teknokent Uydu Otopark",           "type": "service"},
    "S27": {"name_en": "Stadium Junction",             "name_tr": "Stadyum Kavşağı",                  "type": "leisure"},
    "S28": {"name_en": "Aviation Engineering",         "name_tr": "Havacılık Mühendisliği",           "type": "academic"},
}


# ============================================================================
# 8 RING LINES — Real METU routes with operating hours
# ============================================================================
# Each ring has:
#   color           : official name
#   route           : ordered list of stop IDs (the bus visits them in this order)
#   operating_days  : weekday / weekend / both
#   start_time      : earliest scheduled departure (HH:MM)
#   end_time        : latest scheduled departure (HH:MM)
#   headway_minutes : approximate gap between consecutive buses
#   schedule_type   : "interval" (regular gaps) or "fixed" (specific times)

RINGS = {
    "yellow_red": {
        "color": "Yellow-Red",
        "name_tr": "Sarı-Kırmızı Ring",
        "route": ["S02", "S03", "S04", "S05", "S26", "S07", "S08", "S09", "S28", "S11",
                  "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21",
                  "S22", "S23", "S24", "S25", "S13", "S14", "S15", "S16", "S05", "S04",
                  "S03", "S02"],
        "operating_days": "weekday",
        "start_time": "09:00",
        "end_time": "17:35",
        "headway_minutes": 20,
        "schedule_type": "interval",
    },
    "light_brown": {
        "color": "Light Brown",
        "name_tr": "Açık Kahverengi Ring (A-1)",
        "route": ["S01", "S17", "S18", "S19", "S23", "S24", "S25", "S13", "S14", "S15",
                  "S16", "S17", "S01"],
        "operating_days": "weekday",
        "start_time": "08:00",
        "end_time": "20:00",
        "headway_minutes": 5,    # 5 min in morning, 12 min midday, 15 min evening (using rush hour as base)
        "schedule_type": "interval",
    },
    "turquoise": {
        "color": "Turquoise",
        "name_tr": "Turkuaz Ring",
        "route": ["S09", "S06", "S05", "S04", "S03"],
        "operating_days": "weekday",
        "start_time": "08:25",
        "end_time": "08:25",
        "headway_minutes": None,  # single trip per day
        "schedule_type": "fixed",
    },
    "orange": {
        "color": "Orange",
        "name_tr": "Turuncu Ring",
        "route": ["S09", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19",
                  "S20", "S21", "S22", "S23", "S24", "S25", "S13", "S14", "S15", "S16",
                  "S05", "S04", "S03"],
        "operating_days": "weekday",
        "start_time": "08:05",
        "end_time": "08:40",
        "headway_minutes": 5,    # 8 trips in 35 minutes
        "schedule_type": "interval",
    },
    "navy": {
        "color": "Navy",
        "name_tr": "Lacivert Ring",
        "route": ["S21", "S22", "S23", "S24", "S25", "S13", "S14", "S15", "S16", "S06",
                  "S07", "S26", "S09", "S10", "S09", "S08", "S11", "S12", "S25", "S24",
                  "S20", "S21"],
        "operating_days": "weekday",
        "start_time": "18:00",
        "end_time": "20:00",
        "headway_minutes": 30,
        "schedule_type": "interval",
    },
    "purple": {
        "color": "Purple",
        "name_tr": "Mor Ring",
        "route": ["S01", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S13",
                  "S14", "S15", "S16", "S06", "S07", "S26", "S09", "S10", "S09", "S08",
                  "S11", "S12", "S25", "S24", "S27", "S19", "S01"],
        "operating_days": "weekday",
        "start_time": "20:30",
        "end_time": "00:30",
        "headway_minutes": 40,
        "schedule_type": "interval",
    },
    "gray_day": {
        "color": "Gray Day",
        "name_tr": "Gri Ring Sabah",
        "route": ["S02", "S03", "S04", "S05", "S06", "S07", "S09", "S07", "S06", "S17",
                  "S18", "S19", "S20", "S21", "S22", "S01", "S17", "S18", "S19", "S24",
                  "S25", "S11", "S09", "S07", "S06", "S05", "S17", "S03", "S02"],
        "operating_days": "weekend",
        "start_time": "08:30",
        "end_time": "18:30",
        "headway_minutes": 60,   # 11 trips across 10 hours
        "schedule_type": "interval",
    },
    "gray_night": {
        "color": "Gray Night",
        "name_tr": "Gri Ring Akşam",
        "route": ["S02", "S03", "S04", "S05", "S06", "S07", "S09", "S11", "S25", "S24",
                  "S20", "S21", "S22", "S23", "S01", "S17", "S18", "S19", "S24", "S25",
                  "S11", "S08", "S09", "S10", "S07", "S06", "S05", "S17", "S03", "S02"],
        "operating_days": "weekend",
        "start_time": "19:30",
        "end_time": "23:30",
        "headway_minutes": 60,   # 5 trips across 4 hours
        "schedule_type": "interval",
    },
}


# ============================================================================
# HELPER FUNCTIONS — convenient access patterns
# ============================================================================

def get_stop_ids():
    """Return list of all stop IDs."""
    return list(STOPS.keys())

def get_ring_ids():
    """Return list of all ring IDs."""
    return list(RINGS.keys())

def get_stops_for_ring(ring_id):
    """Return ordered list of unique stop IDs served by a ring."""
    return list(dict.fromkeys(RINGS[ring_id]["route"]))  # preserves order, removes duplicates

def get_rings_serving_stop(stop_id):
    """Return list of ring IDs that include this stop."""
    return [rid for rid, r in RINGS.items() if stop_id in r["route"]]

def get_stop_type(stop_id):
    """Return type category (academic/dorm/transit_hub/leisure/service)."""
    return STOPS[stop_id]["type"]

def is_ring_active_on(ring_id, weekday):
    """Check if a ring operates on a given weekday (0=Monday, 6=Sunday)."""
    days = RINGS[ring_id]["operating_days"]
    is_weekend = weekday >= 5
    if days == "weekday":
        return not is_weekend
    elif days == "weekend":
        return is_weekend
    return True  # "both"


# ============================================================================
# CONFIG — Tunable simulation parameters
# ============================================================================

SIMULATION_CONFIG = {
    "time_granularity_minutes": 5,         # demand sampled every 5 minutes
    "simulation_start_date": "2025-09-15", # Fall 2025 semester start (Monday)
    "simulation_days": 90,                  # 90-day window
    "random_seed": 42,                      # reproducibility
}

# Demand multipliers — start values, easily tunable
DEMAND_MULTIPLIERS = {
    "weekend":   0.30,   # weekday → weekend reduction
    "exam_week": 1.80,   # midterm / final week surge
    "break":     0.10,   # semester break dip
    "holiday":   0.15,   # public holidays (e.g., Oct 29)
}

# Semester calendar relative to start date
SEMESTER_CALENDAR = {
    "exam_weeks": [
        ("2025-10-20", "2025-10-26"),  # midterm week (week 6)
        ("2025-12-08", "2025-12-13"),  # finals approach (last week of window)
    ],
    "break_weeks": [],                  # no major breaks within 90-day fall window
    "holidays": [
        "2025-10-29",  # Republic Day
    ],
}
