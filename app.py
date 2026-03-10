import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import math
import requests

# ─────────────────────────────────────────────
# UI CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="T.G. | Tactical Flight Debrief",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #050505; color: #C0C0C0; font-family: 'JetBrains Mono', monospace; }
    h1, h2, h3 { color: #FF9F1C !important; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px solid #1A1A1A; padding-bottom: 5px; }
    div[data-testid="metric-container"] { background-color: #0F0F0F; border: 1px solid #333333; padding: 20px; border-radius: 0px; border-left: 5px solid #FF9F1C; }
    div[data-testid="metric-container"] label { color: #888888 !important; font-size: 0.8rem !important; }
    div[data-testid="metric-container"] div { color: #00FF41 !important; font-weight: 700 !important; }
    .streamlit-expanderHeader { background-color: #0F0F0F !important; color: #FF9F1C !important; border: 1px solid #333333 !important; border-radius: 0px !important; }
    .stFileUploader { border: 1px dashed #FF9F1C; background-color: #0F0F0F; }
    div[data-baseweb="select"] > div { background-color: #0F0F0F !important; border: 1px solid #FF9F1C !important; color: #00FF41 !important; }
    table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    th { background-color: #1A1A1A; color: #FF9F1C; padding: 8px 12px; text-align: left; border: 1px solid #333; }
    td { padding: 7px 12px; border: 1px solid #222; color: #C0C0C0; }
    tr:nth-child(even) { background-color: #0A0A0A; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TURN_RATE_THRESHOLD    = 1.8     # deg/s — minimum to classify as in-maneuver
MIN_AIRSPEED_KTS       = 40      # kts — maneuver detection floor
AGL_TOUCHDOWN_FT       = 75      # ft AGL — runway contact threshold
MIN_AIRBORNE_AGL_FT    = 200     # ft AGL — minimum to be considered airborne
MIN_MANEUVER_DUR_S     = 15      # seconds — minimum maneuver duration
MIN_MANEUVER_TURN_DEG  = 100     # degrees — minimum heading sweep to log
ALT_PERCENTILE_FIELD   = 0.02    # field elevation estimation percentile
METAR_TTL_S            = 600     # METAR cache TTL in seconds
GS_TAXI_THRESHOLD      = 35      # kts — below this = taxiing (3D view filter)
STALL_SPEED_KTS        = 55      # kts GS — stall detection threshold
STALL_VSI_THRESH       = -300    # fpm — minimum sink rate to flag as stall entry
EMERG_DESCENT_VSI      = -1500   # fpm — threshold for emergency descent detection
EMERG_DESCENT_MIN_S    = 10      # seconds — minimum duration to log an ED event
PATTERN_ALT_AGL        = 800     # ft AGL — nominal pattern altitude (user-adjustable in Tab 8)
NAVAID_RADIUS_NM       = 40      # nm — radius for airport/navaid fetch

# ─────────────────────────────────────────────
# FAA ACS STANDARDS REFERENCE TABLE
# Each entry: ppl_tol = PPL ACS tolerance, cpl_tol = CPL ACS tolerance
# Lower = tighter for deviations; higher = better for minimum rates.
# ─────────────────────────────────────────────
ACS_STANDARDS = {
    "360° STEEP TURN": {
        "ref": "ACS PA.VI.A / CA.VI.A",
        "params": {
            "Altitude Deviation":      {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Max deviation from entry altitude"},
            "Bank Dev from 45°":       {"unit": "°",   "ppl_tol": 5,   "cpl_tol": 5,   "lower_better": True,  "desc": "Median bank angle deviation from 45°"},
            "Rollout Heading Error":   {"unit": "°",   "ppl_tol": 10,  "cpl_tol": 10,  "lower_better": True,  "desc": "Heading error at 360° completion"},
        }
    },
    "180° COURSE REVERSAL": {
        "ref": "ACS PA.VI.A / CA.VI.A",
        "params": {
            "Altitude Deviation":      {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Max deviation from entry altitude"},
            "Rollout Heading Error":   {"unit": "°",   "ppl_tol": 10,  "cpl_tol": 10,  "lower_better": True,  "desc": "Heading error from reciprocal"},
        }
    },
    "GROUND REFERENCE / S-TURNS": {
        "ref": "ACS PA.VI.B-C / CA.VI.B-C",
        "params": {
            "Altitude Deviation":      {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Max altitude deviation from entry"},
        }
    },
    "STALL": {
        "ref": "ACS PA.IV.C-D / CA.IV.C-D",
        "params": {
            "Altitude Loss":           {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Total altitude lost from stall entry to recovery"},
        }
    },
    "EMERGENCY DESCENT": {
        "ref": "ACS PA.IX.A / CA.IX.A",
        "params": {
            "Peak Descent Rate":       {"unit": "fpm", "ppl_tol": 1500, "cpl_tol": 1500, "lower_better": False, "desc": "Peak descent rate achieved (higher = better)"},
        }
    },
    "TRAFFIC PATTERN": {
        "ref": "ACS PA.VII.B / CA.VII.B",
        "params": {
            "Pattern Altitude Dev":    {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Deviation from pattern altitude on downwind"},
            "Final Heading Error":     {"unit": "°",   "ppl_tol": 10,  "cpl_tol": 5,   "lower_better": True,  "desc": "Alignment error with runway heading on final"},
        }
    },
}

# ─────────────────────────────────────────────
# MATH & AERODYNAMIC HELPERS
# ─────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi    = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def heading_diff(a, b):
    """Shortest angular distance between two headings (0–180)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

# ─────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=METAR_TTL_S)
def fetch_metar(lat, lon):
    try:
        url = f"https://aviationweather.gov/api/data/metar?lat={lat}&lon={lon}&distance=25&format=json"
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and res.json():
            return res.json()[0]
    except (requests.RequestException, ValueError, KeyError):
        return None

@st.cache_data(ttl=3600)
def fetch_airports_navaids(lat, lon, radius_nm=NAVAID_RADIUS_NM):
    """Fetch nearby airports and navaids from aviationweather.gov within radius_nm."""
    lat_offset = radius_nm / 60.0
    lon_offset = radius_nm / (60.0 * math.cos(math.radians(lat)))
    bbox = f"{lat - lat_offset},{lon - lon_offset},{lat + lat_offset},{lon + lon_offset}"

    airports, navaids = [], []
    try:
        r = requests.get(
            f"https://aviationweather.gov/api/data/airport?bbox={bbox}&format=json", timeout=6
        )
        if r.status_code == 200:
            airports = r.json() or []
    except (requests.RequestException, ValueError):
        pass

    try:
        r = requests.get(
            f"https://aviationweather.gov/api/data/navaid?bbox={bbox}&format=json", timeout=6
        )
        if r.status_code == 200:
            navaids = r.json() or []
    except (requests.RequestException, ValueError):
        pass

    return airports, navaids

# ─────────────────────────────────────────────
# CORE KML PIPELINE
# ─────────────────────────────────────────────
@st.cache_data
def process_kml(file_content):
    soup = BeautifulSoup(file_content, 'xml')
    times  = soup.find_all('when')
    coords = soup.find_all('gx:coord')

    if not times or not coords:
        return pd.DataFrame(), "KML is missing <when> or <gx:coord> tags. Is this a Google Earth track log?"

    data = []
    for t, c in zip(times, coords):
        parts = c.text.split()
        if len(parts) == 3:
            data.append({
                'Time':    t.text.replace('Z', ''),
                'Lon':     float(parts[0]),
                'Lat':     float(parts[1]),
                'Alt_Raw': float(parts[2]) * 3.28084,
            })

    df = pd.DataFrame(data)
    if df.empty:
        return df, "No valid coordinate records found in KML."

    df['Time'] = pd.to_datetime(df['Time'])
    df['Dt']   = df['Time'].diff().dt.total_seconds().fillna(1)
    dt_safe    = df['Dt'].replace(0, np.nan)

    df['Cumulative_Min'] = df['Dt'].cumsum() / 60.0
    df['Alt_Smooth']     = df['Alt_Raw'].rolling(window=7, center=True, min_periods=1).mean()
    df['VSI']            = (df['Alt_Smooth'].diff() / (dt_safe / 60.0)).fillna(0).rolling(5).mean()

    field_elev   = df['Alt_Smooth'].quantile(ALT_PERCENTILE_FIELD)
    df['Alt_AGL'] = df['Alt_Smooth'] - field_elev

    dist, bear = [0], [0]
    for i in range(1, len(df)):
        dist.append(haversine_distance(
            df.iloc[i-1]['Lat'], df.iloc[i-1]['Lon'],
            df.iloc[i]['Lat'],   df.iloc[i]['Lon']
        ))
        bear.append(calculate_bearing(
            df.iloc[i-1]['Lat'], df.iloc[i-1]['Lon'],
            df.iloc[i]['Lat'],   df.iloc[i]['Lon']
        ))

    df['GS']    = (pd.Series(dist) / (dt_safe / 3600.0)).fillna(0).rolling(5).mean()
    df['Track'] = bear

    df['Track_Delta'] = df['Track'].diff().abs()
    df['Track_Delta'] = df['Track_Delta'].apply(lambda x: 360 - x if x > 180 else x).fillna(0)
    df['Turn_Rate']   = (df['Track_Delta'] / dt_safe).rolling(window=3).mean().fillna(0)

    g              = 32.174
    df['Vel_fps']  = df['GS'] * 1.68781
    turn_rate_rad  = np.radians(df['Turn_Rate'].fillna(0))
    df['Bank_Angle'] = np.degrees(np.arctan((turn_rate_rad * df['Vel_fps']) / g)).fillna(0)
    df['G_Load']     = (1 / np.cos(np.radians(df['Bank_Angle']))).clip(1, 3)
    df['Specific_Energy'] = df['Alt_Smooth'] + ((df['Vel_fps'] ** 2) / (2 * g))

    return df, None

# ─────────────────────────────────────────────
# ACS GRADING ENGINE
# ─────────────────────────────────────────────
def grade_maneuver(label, row, mdata):
    """
    Returns a list of dicts: {param, value, unit, ppl_tol, cpl_tol, ppl_pass, cpl_pass, desc}
    for display in the ACS Standards panel.
    """
    entry_alt = row['entry_alt']
    max_dev   = max(row['alt_max'] - entry_alt, entry_alt - row['alt_min'])
    results   = []

    def add(param, value, std_key):
        s = ACS_STANDARDS.get(std_key, {}).get("params", {}).get(param)
        if not s:
            return
        lb = s["lower_better"]
        ppl_pass = (value <= s["ppl_tol"]) if lb else (value >= s["ppl_tol"])
        cpl_pass = (value <= s["cpl_tol"]) if lb else (value >= s["cpl_tol"])
        results.append({
            "Parameter":  param,
            "Your Value": f"{round(value, 1)} {s['unit']}",
            "PPL Tol":    f"{'≤' if lb else '≥'}{s['ppl_tol']} {s['unit']}",
            "CPL Tol":    f"{'≤' if lb else '≥'}{s['cpl_tol']} {s['unit']}",
            "PPL":        "✅ PASS" if ppl_pass else "❌ BUST",
            "CPL":        "✅ PASS" if cpl_pass else "❌ BUST",
            "Description": s["desc"],
            "_ppl_pass":  ppl_pass,
            "_cpl_pass":  cpl_pass,
        })

    std_key = "360° STEEP TURN" if "360°" in label else \
              "180° COURSE REVERSAL" if "180°" in label else \
              "GROUND REFERENCE / S-TURNS"

    add("Altitude Deviation", max_dev, std_key)

    if "360°" in label:
        avg_bank = mdata['Bank_Angle'].median()
        bank_dev = abs(avg_bank - 45)
        add("Bank Dev from 45°", bank_dev, std_key)

        entry_track = mdata['Track'].iloc[0]
        exit_track  = mdata['Track'].iloc[-1]
        hdg_err     = heading_diff(entry_track, exit_track)
        add("Rollout Heading Error", hdg_err, std_key)

    elif "180°" in label:
        entry_track   = mdata['Track'].iloc[0]
        exit_track    = mdata['Track'].iloc[-1]
        expected_exit = (entry_track + 180) % 360
        hdg_err       = heading_diff(exit_track, expected_exit)
        add("Rollout Heading Error", hdg_err, std_key)

    return results

def render_acs_table(grade_rows):
    """Render ACS grading results as an HTML table."""
    rows_html = ""
    for r in grade_rows:
        ppl_color = "#00FF41" if r["_ppl_pass"] else "#FF4444"
        cpl_color = "#00FF41" if r["_cpl_pass"] else "#FF4444"
        rows_html += f"""
        <tr>
            <td>{r['Parameter']}</td>
            <td style='color:#00FFFF;font-weight:700'>{r['Your Value']}</td>
            <td>{r['PPL Tol']}</td>
            <td>{r['CPL Tol']}</td>
            <td style='color:{ppl_color};font-weight:700'>{r['PPL']}</td>
            <td style='color:{cpl_color};font-weight:700'>{r['CPL']}</td>
            <td style='color:#888;font-size:0.8rem'>{r['Description']}</td>
        </tr>"""
    return f"""
    <table>
        <thead><tr>
            <th>Parameter</th><th>Your Result</th>
            <th>PPL Tolerance</th><th>CPL Tolerance</th>
            <th>PPL Grade</th><th>CPL Grade</th><th>Description</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""

# ─────────────────────────────────────────────
# STALL & SLOW FLIGHT DETECTOR
# ─────────────────────────────────────────────
def detect_stalls(df):
    """
    Flag segments where GS < STALL_SPEED_KTS AND VSI < STALL_VSI_THRESH while airborne.
    Returns list of event dicts.
    """
    df = df.copy()
    df['Stall_Flag'] = (
        (df['GS'] < STALL_SPEED_KTS) &
        (df['GS'] > 15) &
        (df['VSI'] < STALL_VSI_THRESH) &
        (df['Alt_AGL'] > 300)
    )
    df['Stall_Event_ID'] = (df['Stall_Flag'] != df['Stall_Flag'].shift()).cumsum()

    events = []
    for eid, edata in df[df['Stall_Flag']].groupby('Stall_Event_ID'):
        duration = edata['Dt'].sum()
        if duration < 4:
            continue
        entry_alt    = edata['Alt_Smooth'].iloc[0]
        recovery_alt = edata['Alt_Smooth'].iloc[-1]
        alt_loss     = max(0, entry_alt - edata['Alt_Smooth'].min())
        events.append({
            'time':         edata['Time'].iloc[0],
            'entry_alt':    entry_alt,
            'recovery_alt': recovery_alt,
            'alt_loss':     alt_loss,
            'min_gs':       edata['GS'].min(),
            'min_vsi':      edata['VSI'].min(),
            'duration':     duration,
            'data':         edata,
        })
    return events

# ─────────────────────────────────────────────
# EMERGENCY DESCENT PROFILER
# ─────────────────────────────────────────────
def detect_emergency_descents(df):
    """
    Flag segments with sustained VSI < EMERG_DESCENT_VSI fpm while airborne and flying.
    Returns list of event dicts.
    """
    df = df.copy()
    df['ED_Flag'] = (
        (df['VSI'] < EMERG_DESCENT_VSI) &
        (df['Alt_AGL'] > 500) &
        (df['GS'] > MIN_AIRSPEED_KTS)
    )
    df['ED_Event_ID'] = (df['ED_Flag'] != df['ED_Flag'].shift()).cumsum()

    events = []
    for eid, edata in df[df['ED_Flag']].groupby('ED_Event_ID'):
        duration = edata['Dt'].sum()
        if duration < EMERG_DESCENT_MIN_S:
            continue
        entry_alt = edata['Alt_Smooth'].iloc[0]
        exit_alt  = edata['Alt_Smooth'].iloc[-1]
        avg_vs    = edata['VSI'].mean()
        peak_vs   = edata['VSI'].min()  # most negative
        alt_loss  = entry_alt - exit_alt

        # ACS grade: achieved peak rate must be ≥ 1500 fpm
        acs_pass = abs(peak_vs) >= 1500
        events.append({
            'time':      edata['Time'].iloc[0],
            'entry_alt': entry_alt,
            'exit_alt':  exit_alt,
            'alt_loss':  alt_loss,
            'avg_vs':    avg_vs,
            'peak_vs':   peak_vs,
            'duration':  duration,
            'acs_pass':  acs_pass,
            'data':      edata,
        })
    return events

# ─────────────────────────────────────────────
# PATTERN WORK GRADER
# ─────────────────────────────────────────────
MAX_CIRCUIT_MIN = 20  # minutes — circuits longer than this are straight-in approaches, not patterns

def detect_pattern_legs(df, td_row, pattern_alt_agl=PATTERN_ALT_AGL, right_hand=False):
    """
    Isolate a single circuit by finding the most recent departure before this
    touchdown, then match leg headings within that window.

    Returns (legs_dict, runway_hdg, skip_reason) where skip_reason is a string
    if the circuit was skipped (too long = straight-in), or None if graded.
    """
    td_time = td_row['Time']

    # ── Find circuit start: most recent liftoff before this touchdown ──
    df_before = df[df['Time'] < td_time].copy()
    df_before['_on_gnd']    = df_before['Alt_AGL'] < AGL_TOUCHDOWN_FT
    df_before['_departure'] = (df_before['_on_gnd'] == False) & \
                               (df_before['_on_gnd'].shift(1) == True)
    departures = df_before[df_before['_departure']]

    if not departures.empty:
        circuit_start = departures['Time'].iloc[-1]
    else:
        circuit_start = td_time - pd.Timedelta(minutes=MAX_CIRCUIT_MIN)

    circuit_dur_min = (td_time - circuit_start).total_seconds() / 60.0

    # ── KEY GATE: skip circuits that are too long to be a pattern ──
    # A normal T&G circuit is 4–12 min. Anything longer is a cross-country
    # or practice-area flight returning to land straight-in — not gradeable.
    if circuit_dur_min > MAX_CIRCUIT_MIN:
        return {}, None, f"STRAIGHT-IN / LONG APPROACH ({circuit_dur_min:.0f} min since last departure — not a circuit)"

    window   = df[(df['Time'] >= circuit_start) & (df['Time'] <= td_time)].copy()
    airborne = window[(window['GS'] > MIN_AIRSPEED_KTS) & (window['Alt_AGL'] > 50)]

    if airborne.empty or len(airborne) < 10:
        return {}, None, "INSUFFICIENT AIRBORNE DATA IN CIRCUIT WINDOW"

    # ── Runway heading: median track in last 45 s while still flying ──
    final_flying = airborne[airborne['Time'] >= td_time - pd.Timedelta(seconds=45)]
    if final_flying.empty:
        return {}, None, "NO FLYING DATA IN FINAL 45s"

    runway_hdg = final_flying['Track'].median() % 360

    # ── Leg headings: left-hand (standard) or right-hand ──
    turn = -1 if right_hand else 1   # +1 = left turns add 90° per leg, -1 = right
    leg_headings = {
        "CROSSWIND": (runway_hdg - turn * 90)  % 360,
        "DOWNWIND":  (runway_hdg + 180)         % 360,
        "BASE":      (runway_hdg + turn * 90)   % 360,
        "FINAL":     runway_hdg,
    }

    HDG_TOL = 35  # degrees

    legs = {}
    for leg_name, exp_hdg in leg_headings.items():
        mask     = airborne['Track'].apply(lambda hdg: heading_diff(hdg, exp_hdg)) < HDG_TOL
        leg_data = airborne[mask]
        if not leg_data.empty and leg_data['Dt'].sum() > 5:
            legs[leg_name] = leg_data

    return legs, runway_hdg, None



# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def acs_status_color(value, ppl_tol, cpl_tol, lower_better=True):
    if lower_better:
        if value <= cpl_tol: return "#00FF41", "CPL GRADE"
        if value <= ppl_tol: return "#FF9F1C", "PPL PASS"
        return "#FF4444", "ACS BUST"
    else:
        if value >= cpl_tol: return "#00FF41", "CPL GRADE"
        if value >= ppl_tol: return "#FF9F1C", "PPL PASS"
        return "#FF4444", "ACS BUST"

# ─────────────────────────────────────────────
# APP ENTRY POINT
# ─────────────────────────────────────────────
st.title("🛰️ T.G. TACTICAL FLIGHT DEBRIEF")
st.markdown("`SYSTEM STATUS: ONLINE | ADVANCED AERO ENGINE ARMED`")

uploaded = st.file_uploader("", type=['kml'])

if uploaded:
    raw_content = uploaded.getvalue().decode('utf-8')
    df, parse_error = process_kml(raw_content)

    if parse_error:
        st.error(f"⚠️ **KML PARSE FAILURE:** {parse_error}")
        st.stop()

    if df.empty:
        st.warning("No data found in file.")
        st.stop()

    # Use median position for METAR — more representative than first point on
    # cross-country flights that land somewhere other than departure.
    metar          = fetch_metar(df['Lat'].median(), df['Lon'].median())
    total_mins     = int(df['Dt'].sum() / 60)
    field_elevation = df['Alt_Smooth'].quantile(ALT_PERCENTILE_FIELD)

    # ── TOP METRICS ──────────────────────────
    st.markdown("### 📡 INITIAL CONDITIONS & TELEMETRY")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PEAK ALTITUDE",   f"{int(df['Alt_Smooth'].max())} FT MSL")
    c2.metric("MAX GROUNDSPEED", f"{int(df['GS'].max())} KTS")
    c3.metric("MAX G-LOAD",      f"+{df['G_Load'].max():.1f} G")
    c4.metric("TOTAL SORTIE",    f"{total_mins} MIN")

    if metar:
        st.info(f"📍 **SURFACE WX ({metar.get('icaoId')}):** `{metar.get('rawOb')}`")

    # ── MANEUVER ANALYSIS ────────────────────
    st.markdown("### 🎯 ACS MANEUVER ANALYSIS")

    df['In_Maneuver'] = (df['Turn_Rate'] > TURN_RATE_THRESHOLD) & (df['GS'] > MIN_AIRSPEED_KTS)
    df['Maneuver_ID'] = (df['In_Maneuver'] != df['In_Maneuver'].shift()).cumsum()

    maneuver_df  = df[df['In_Maneuver']]
    maneuver_agg = maneuver_df.groupby('Maneuver_ID').agg(
        total_turn=('Track_Delta', 'sum'),
        duration=  ('Dt',          'sum'),
        entry_alt= ('Alt_Smooth',  'first'),
        alt_max=   ('Alt_Smooth',  'max'),
        alt_min=   ('Alt_Smooth',  'min'),
        peak_g=    ('G_Load',      'max'),
        gs_max=    ('GS',          'max'),
        gs_min=    ('GS',          'min'),
    ).reset_index()

    # Store graded maneuvers for ACS panel
    all_graded_maneuvers = []
    found_mnvrs = 0

    for _, row in maneuver_agg.iterrows():
        total_turn = row['total_turn']
        duration   = row['duration']
        if duration < MIN_MANEUVER_DUR_S or total_turn < MIN_MANEUVER_TURN_DEG:
            continue

        found_mnvrs += 1
        entry_alt = row['entry_alt']
        max_dev   = max(row['alt_max'] - entry_alt, entry_alt - row['alt_min'])
        mdata     = maneuver_df[maneuver_df['Maneuver_ID'] == row['Maneuver_ID']]

        if total_turn >= 700:
            label, color, status = "EXTENDED CIRCLING / HOLD", "#888888", "UNGRADED"
        elif 320 <= total_turn <= 400:
            label  = "360° STEEP TURN"
            color, status = acs_status_color(max_dev, 100, 50)
        elif 150 <= total_turn <= 210:
            label  = "180° COURSE REVERSAL"
            color, status = acs_status_color(max_dev, 100, 50)
        else:
            label  = "GROUND REFERENCE / S-TURNS"
            color, status = acs_status_color(max_dev, 100, 50)

        # Grade and store
        grade_rows = grade_maneuver(label, row, mdata)
        all_graded_maneuvers.append({
            'id': found_mnvrs, 'label': label, 'color': color, 'status': status,
            'total_turn': total_turn, 'row': row, 'mdata': mdata,
            'grade_rows': grade_rows, 'max_dev': max_dev
        })

        with st.expander(f"MNVR {found_mnvrs} | {label} | TURN: {int(total_turn)}°"):
            st.markdown(
                f"**STATUS:** <span style='color:{color}'>{status}</span> (DEV: {int(max_dev)}FT)",
                unsafe_allow_html=True
            )
            st.write(f"`ENTRY ALT: {int(entry_alt)} FT | DURATION: {int(duration)}s | PEAK G: +{row['peak_g']:.1f}G`")

            if "360°" in label:
                wind_est = (row['gs_max'] - row['gs_min']) / 2
                st.write(f"`~EST WINDS ALOFT: {int(wind_est)} KTS (valid only for complete circular turns)`")

            fig_mnvr = px.line_mapbox(mdata, lat="Lat", lon="Lon", zoom=14.5, height=300)
            fig_mnvr.update_traces(line=dict(color='#00FF41', width=4))
            fig_mnvr.update_layout(
                mapbox_style="carto-darkmatter", template="plotly_dark",
                margin=dict(l=0, r=0, b=0, t=0),
                mapbox=dict(center=dict(lat=mdata['Lat'].mean(), lon=mdata['Lon'].mean()))
            )
            st.plotly_chart(fig_mnvr, use_container_width=True, key=f"mnvr_map_{row['Maneuver_ID']}_{found_mnvrs}")

    if found_mnvrs == 0:
        st.info("`NO GRADABLE MANEUVERS DETECTED — minimum 15s duration and 100° heading sweep required`")

    # ── Pre-compute touchdown data once (used by Tab 4 and Tab 8) ──────────
    df['On_Ground']         = df['Alt_AGL'] < AGL_TOUCHDOWN_FT
    df['Touchdown_Trigger'] = (df['On_Ground'] == True) & (df['On_Ground'].shift(1) == False)
    touchdowns_all          = df[df['Touchdown_Trigger']]

    # ── TABS ─────────────────────────────────
    st.markdown("### 🗺️ SPATIAL TELEMETRY, PHYSICS & ADVANCED ANALYSIS")
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "2D DYNAMIC MAP",
        "3D AIRWAY CORRIDOR",
        "AERODYNAMICS",
        "TOUCH & GO PROFILER",
        "ACS STANDARDS PANEL",
        "STALL & SLOW FLIGHT",
        "EMERGENCY DESCENT",
        "PATTERN WORK GRADER",
    ])

    # ── TAB 1: 2D MAP ──────────────────────────
    with t1:
        st.write("`SELECT AVIONICS OVERLAY METRIC:`")
        gs_flying = df.loc[df['GS'] > GS_TAXI_THRESHOLD, 'GS']
        gs_range  = [gs_flying.quantile(0.05), gs_flying.quantile(0.99)] if not gs_flying.empty else [0, df['GS'].max()]
        map_metrics = {
            "ALTITUDE (AGL)":       ["Alt_AGL",    "Viridis", [0, 3000]],
            "VERTICAL SPEED (FPM)": ["VSI",        "RdBu_r",  [-1000, 1000]],
            "GROUNDSPEED (KTS)":    ["GS",         "Inferno", gs_range],
            "BANK ANGLE (°)":       ["Bank_Angle", "Plasma",  [0, 60]],
            "G-LOAD (G)":           ["G_Load",     "Turbo",   [1, 2]],
            "TURN RATE (°/SEC)":    ["Turn_Rate",  "Plasma",  [0, 4]],
        }
        selected_metric = st.selectbox("", list(map_metrics.keys()))
        active_col, active_cs, active_range = map_metrics[selected_metric]

        show_navaids = st.checkbox("🔵 OVERLAY AIRPORTS & NAVAIDS", value=True)

        fig_map = px.scatter_mapbox(
            df, lat="Lat", lon="Lon", color=active_col,
            color_continuous_scale=active_cs, range_color=active_range,
            zoom=10, height=680,
            hover_data=["Alt_AGL", "GS", "VSI", "Bank_Angle"]
        )

        if show_navaids:
            airports, navaids = fetch_airports_navaids(df['Lat'].mean(), df['Lon'].mean())

            if airports:
                ap_lats = [a.get('lat') for a in airports if a.get('lat') and a.get('lon')]
                ap_lons = [a.get('lon') for a in airports if a.get('lat') and a.get('lon')]
                ap_names = [
                    f"✈ {a.get('icaoId','?')} | {a.get('name','')}<br>ELEV: {a.get('elev','?')} FT"
                    for a in airports if a.get('lat') and a.get('lon')
                ]
                fig_map.add_trace(go.Scattermapbox(
                    lat=ap_lats, lon=ap_lons, mode='markers+text',
                    marker=dict(size=14, color='#FF9F1C', symbol='airport'),
                    text=[a.get('icaoId', '') for a in airports if a.get('lat') and a.get('lon')],
                    textposition="top center",
                    textfont=dict(color='#FF9F1C', size=11),
                    hovertext=ap_names,
                    hoverinfo='text',
                    name="AIRPORTS",
                ))

            if navaids:
                # Separate VORs from NDBs
                for nav_type, symbol, color in [("VOR", "marker", "#00FFFF"), ("NDB", "circle", "#FF44FF")]:
                    filtered = [n for n in navaids if nav_type in (n.get('type') or '').upper()
                                and n.get('lat') and n.get('lon')]
                    if filtered:
                        fig_map.add_trace(go.Scattermapbox(
                            lat=[n['lat'] for n in filtered],
                            lon=[n['lon'] for n in filtered],
                            mode='markers+text',
                            marker=dict(size=10, color=color),
                            text=[n.get('navId', '') for n in filtered],
                            textposition="top right",
                            textfont=dict(color=color, size=10),
                            hovertext=[
                                f"🔷 {n.get('navId','?')} {n.get('type','')}<br>{n.get('name','')}<br>FREQ: {n.get('freq','?')}"
                                for n in filtered
                            ],
                            hoverinfo='text',
                            name=f"{nav_type}s",
                        ))

        fig_map.update_layout(
            mapbox_style="carto-darkmatter", template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color='#C0C0C0'))
        )
        st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})

    # ── TAB 2: 3D MAP ──────────────────────────
    with t2:
        st.write("`USE SLIDERS TO CROP OUT TAXI, CLIMB, AND HOLDING PATTERN RUBBISH`")
        trim_start, trim_end = st.slider(
            "SELECT PRACTICE AREA TIMEFRAME (MINUTES)",
            0, total_mins, (int(total_mins * 0.15), int(total_mins * 0.85))
        )
        airborne_df = df[
            (df['GS'] > GS_TAXI_THRESHOLD) &
            (df['Alt_Smooth'] > (field_elevation + MIN_AIRBORNE_AGL_FT)) &
            (df['Cumulative_Min'] >= trim_start) &
            (df['Cumulative_Min'] <= trim_end)
        ]
        if not airborne_df.empty:
            fig_3d = go.Figure(data=go.Scatter3d(
                x=airborne_df['Lon'], y=airborne_df['Lat'], z=airborne_df['Alt_Smooth'],
                mode='lines',
                line=dict(color=airborne_df['GS'], colorscale='Inferno', width=6, colorbar=dict(title="KTS")),
                text=[f"ALT: {alt:.0f} FT<br>GS: {gs:.0f} KTS<br>BANK: {bk:.0f}°"
                      for alt, gs, bk in zip(airborne_df['Alt_Smooth'], airborne_df['GS'], airborne_df['Bank_Angle'])],
                hoverinfo="text"
            ))
            fig_3d.update_layout(
                title="TRIMMED 3D TRAJECTORY (COLOR=SPEED)",
                template="plotly_dark", height=700, margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis_title="LONGITUDE", yaxis_title="LATITUDE", zaxis_title="ALTITUDE (FT)",
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.4)
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True, config={
                'scrollZoom': True, 'displayModeBar': True,
                'displaylogo': False, 'modeBarButtonsToRemove': ['resetCameraDefault3d']
            })
        else:
            st.warning("`WARNING: NO DATA REMAINS AFTER CURRENT CROP SELECTION.`")

    # ── TAB 3: AERODYNAMICS ────────────────────
    with t3:
        st.write("`SPECIFIC ENERGY STATE & ESTIMATED BANK ANGLES`")
        fig_aero = go.Figure()
        fig_aero.add_trace(go.Scatter(x=df['Time'], y=df['Specific_Energy'], name="SPECIFIC ENERGY", line=dict(color="#FF9F1C", width=2)))
        fig_aero.add_trace(go.Scatter(x=df['Time'], y=df['Alt_Smooth'], name="POTENTIAL ENERGY (ALT)", line=dict(color="#00FF41", width=2, dash='dot')))
        fig_aero.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="ENERGY STATE (ft equivalent)", height=400)
        st.plotly_chart(fig_aero, use_container_width=True, config={'scrollZoom': True})

        fig_bank = go.Figure()
        fig_bank.add_trace(go.Scatter(x=df['Time'], y=df['Bank_Angle'], name="ESTIMATED BANK (°)", line=dict(color="#00FFFF", width=2)))
        fig_bank.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="BANK ANGLE (°)", height=300)
        st.plotly_chart(fig_bank, use_container_width=True, config={'scrollZoom': True})

    # ── TAB 4: TOUCH & GO ─────────────────────
    with t4:
        st.write("`TOUCH & GO DETECTOR: 90-SECOND GLIDEPATH ISOLATION`")
        touchdowns = touchdowns_all

        if not touchdowns.empty:
            st.info(f"🛬 **DETECTED {len(touchdowns)} RUNWAY CONTACT(S)**")
            fig_glide = go.Figure()
            fig_speed = go.Figure()
            for idx, (td_index, td_row) in enumerate(touchdowns.iterrows()):
                start_time    = td_row['Time'] - pd.Timedelta(seconds=90)
                approach_data = df[(df['Time'] >= start_time) & (df['Time'] <= td_row['Time'])].copy()
                if len(approach_data) > 10:
                    approach_data['Sec_To_TD'] = (approach_data['Time'] - td_row['Time']).dt.total_seconds()
                    app_name = f"APPROACH {idx+1}"
                    fig_glide.add_trace(go.Scatter(x=approach_data['Sec_To_TD'], y=approach_data['Alt_AGL'], mode='lines', name=app_name, line=dict(width=3)))
                    fig_speed.add_trace(go.Scatter(x=approach_data['Sec_To_TD'], y=approach_data['GS'],     mode='lines', name=app_name, line=dict(width=3)))

            fig_glide.update_layout(template="plotly_dark", title="GLIDEPATH PROFILE (AGL)",     xaxis_title="SECONDS TO TOUCHDOWN", yaxis_title="ALTITUDE (FT AGL)",   hovermode="x unified", height=400)
            fig_speed.update_layout(template="plotly_dark", title="AIRSPEED DECAY PROFILE",      xaxis_title="SECONDS TO TOUCHDOWN", yaxis_title="GROUNDSPEED (KTS)",    hovermode="x unified", height=300)
            st.plotly_chart(fig_glide, use_container_width=True, config={'scrollZoom': True})
            st.plotly_chart(fig_speed, use_container_width=True, config={'scrollZoom': True})
        else:
            st.warning("`NO RUNWAY CONTACT DETECTED IN LOG.`")

    # ── TAB 5: ACS STANDARDS PANEL ────────────
    with t5:
        st.write("`FAA ACS STANDARDS REFERENCE — FULL TOLERANCE TABLE WITH YOUR DEVIATIONS`")

        # Reference table (all standards, no flight data)
        st.markdown("#### 📋 ACS TOLERANCE REFERENCE")
        ref_rows = []
        for maneuver_name, maneuver_data in ACS_STANDARDS.items():
            for param, s in maneuver_data["params"].items():
                sign = "≤" if s["lower_better"] else "≥"
                ref_rows.append({
                    "Maneuver": maneuver_name,
                    "ACS Ref":  maneuver_data["ref"],
                    "Parameter": param,
                    "PPL Tolerance": f"{sign}{s['ppl_tol']} {s['unit']}",
                    "CPL Tolerance": f"{sign}{s['cpl_tol']} {s['unit']}",
                    "Notes": s["desc"],
                })
        ref_df = pd.DataFrame(ref_rows)
        st.dataframe(
            ref_df, use_container_width=True, hide_index=True,
            column_config={
                "Maneuver":      st.column_config.TextColumn(width="medium"),
                "ACS Ref":       st.column_config.TextColumn(width="small"),
                "Parameter":     st.column_config.TextColumn(width="medium"),
                "PPL Tolerance": st.column_config.TextColumn(width="small"),
                "CPL Tolerance": st.column_config.TextColumn(width="small"),
                "Notes":         st.column_config.TextColumn(width="large"),
            }
        )

        # Per-maneuver detailed grading
        if all_graded_maneuvers:
            st.markdown("#### 🎯 YOUR FLIGHT — MANEUVER-BY-MANEUVER ACS BREAKDOWN")
            for gm in all_graded_maneuvers:
                if gm['label'] == "EXTENDED CIRCLING / HOLD":
                    continue
                with st.expander(f"MNVR {gm['id']} | {gm['label']} | Overall: {gm['status']}"):
                    if gm['grade_rows']:
                        st.markdown(render_acs_table(gm['grade_rows']), unsafe_allow_html=True)
                    else:
                        st.write("`No ACS parameters computed for this maneuver type.`")
        else:
            st.info("`No gradable maneuvers detected in this log.`")

    # ── TAB 6: STALL & SLOW FLIGHT ────────────
    with t6:
        st.write(f"`STALL DETECTOR — FLAGS GS < {STALL_SPEED_KTS} KTS WITH VSI < {STALL_VSI_THRESH} FPM WHILE AIRBORNE`")
        stall_events = detect_stalls(df)

        if stall_events:
            st.info(f"⚠️ **{len(stall_events)} STALL / SLOW FLIGHT EVENT(S) DETECTED**")

            for i, ev in enumerate(stall_events):
                alt_loss     = ev['alt_loss']
                color, grade = acs_status_color(alt_loss, 100, 50)

                with st.expander(f"STALL {i+1} | {ev['time'].strftime('%H:%M:%S')} UTC | ALT LOSS: {int(alt_loss)} FT | {grade}"):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("ENTRY ALT",    f"{int(ev['entry_alt'])} FT MSL")
                    col_b.metric("RECOVERY ALT", f"{int(ev['recovery_alt'])} FT MSL")
                    col_c.metric("ALT LOSS",     f"{int(alt_loss)} FT")
                    col_d.metric("MIN GS",       f"{ev['min_gs']:.0f} KTS")

                    # ACS grading table
                    ppl_pass = alt_loss <= 100
                    cpl_pass = alt_loss <= 50
                    st.markdown(render_acs_table([{
                        "Parameter":  "Altitude Loss",
                        "Your Value": f"{int(alt_loss)} ft",
                        "PPL Tol":    "≤100 ft",
                        "CPL Tol":    "≤50 ft",
                        "PPL":        "✅ PASS" if ppl_pass else "❌ BUST",
                        "CPL":        "✅ PASS" if cpl_pass else "❌ BUST",
                        "Description": "Total altitude lost from stall entry to recovery",
                        "_ppl_pass": ppl_pass, "_cpl_pass": cpl_pass,
                    }]), unsafe_allow_html=True)

                    # Profile chart
                    edata = ev['data']
                    fig_stall = go.Figure()
                    fig_stall.add_trace(go.Scatter(x=edata['Time'], y=edata['Alt_Smooth'], name="ALT (FT MSL)", line=dict(color="#00FF41", width=2)))
                    fig_stall.add_trace(go.Scatter(x=edata['Time'], y=edata['GS'],         name="GS (KTS)",    line=dict(color="#FF9F1C", width=2), yaxis="y2"))
                    fig_stall.add_trace(go.Scatter(x=edata['Time'], y=edata['VSI'],         name="VSI (FPM)",   line=dict(color="#00FFFF", width=1, dash='dot'), yaxis="y3"))
                    fig_stall.update_layout(
                        template="plotly_dark", height=350,
                        xaxis=dict(title="TIME (UTC)", domain=[0, 0.82]),
                        yaxis=dict(title="ALT (FT)", color="#00FF41"),
                        yaxis2=dict(title="GS (KTS)", overlaying='y', side='right', color="#FF9F1C"),
                        yaxis3=dict(title="VSI (FPM)", overlaying='y', side='right',
                                    anchor='free', position=0.92, color="#00FFFF"),
                        legend=dict(bgcolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_stall, use_container_width=True, config={'scrollZoom': True})
        else:
            st.success(f"`✅ NO STALL EVENTS DETECTED (GS never dropped below {STALL_SPEED_KTS} KTS with high sink rate while airborne)`")

        # Slow-flight timeline: show GS across the whole flight
        st.markdown("#### 📉 FULL SORTIE GROUNDSPEED TIMELINE")
        fig_gs = go.Figure()
        fig_gs.add_trace(go.Scatter(x=df['Time'], y=df['GS'], name="GS (KTS)", line=dict(color="#FF9F1C", width=1.5)))
        fig_gs.add_hline(y=STALL_SPEED_KTS, line=dict(color="#FF4444", dash="dash", width=1), annotation_text=f"Stall Threshold ({STALL_SPEED_KTS} KTS)", annotation_font_color="#FF4444")
        fig_gs.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="GROUNDSPEED (KTS)", height=300)
        st.plotly_chart(fig_gs, use_container_width=True, config={'scrollZoom': True})

    # ── TAB 7: EMERGENCY DESCENT ──────────────
    with t7:
        st.write(f"`EMERGENCY DESCENT PROFILER — FLAGS SUSTAINED VSI < {EMERG_DESCENT_VSI} FPM FOR > {EMERG_DESCENT_MIN_S}s WHILE AIRBORNE`")
        ed_events = detect_emergency_descents(df)

        if ed_events:
            st.info(f"🔻 **{len(ed_events)} EMERGENCY DESCENT EVENT(S) DETECTED**")
            for i, ev in enumerate(ed_events):
                peak_vs_abs  = abs(ev['peak_vs'])
                color, grade = acs_status_color(peak_vs_abs, 1500, 1500, lower_better=False)

                with st.expander(f"ED {i+1} | {ev['time'].strftime('%H:%M:%S')} UTC | PEAK {int(peak_vs_abs)} FPM | {grade}"):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("ENTRY ALT",   f"{int(ev['entry_alt'])} FT MSL")
                    col_b.metric("EXIT ALT",    f"{int(ev['exit_alt'])} FT MSL")
                    col_c.metric("ALT LOST",    f"{int(ev['alt_loss'])} FT")
                    col_d.metric("PEAK RATE",   f"{int(peak_vs_abs)} FPM")

                    acs_pass = peak_vs_abs >= 1500
                    st.markdown(render_acs_table([{
                        "Parameter":  "Peak Descent Rate",
                        "Your Value": f"{int(peak_vs_abs)} fpm",
                        "PPL Tol":    "≥1500 fpm",
                        "CPL Tol":    "≥1500 fpm",
                        "PPL":        "✅ PASS" if acs_pass else "❌ BUST",
                        "CPL":        "✅ PASS" if acs_pass else "❌ BUST",
                        "Description": "ACS requires establishing maximum safe descent rate",
                        "_ppl_pass": acs_pass, "_cpl_pass": acs_pass,
                    }]), unsafe_allow_html=True)

                    st.write(f"`AVG DESCENT RATE: {int(abs(ev['avg_vs']))} FPM | DURATION: {int(ev['duration'])}s`")

                    edata = ev['data']
                    fig_ed = go.Figure()
                    fig_ed.add_trace(go.Scatter(x=edata['Time'], y=edata['Alt_AGL'], name="ALT AGL (FT)", line=dict(color="#00FF41", width=2)))
                    fig_ed.add_trace(go.Scatter(x=edata['Time'], y=edata['VSI'],     name="VSI (FPM)",   line=dict(color="#FF4444", width=2), yaxis="y2"))
                    fig_ed.add_trace(go.Scatter(x=edata['Time'], y=edata['GS'],      name="GS (KTS)",    line=dict(color="#FF9F1C", width=1.5, dash='dot'), yaxis="y3"))
                    fig_ed.update_layout(
                        template="plotly_dark", height=350,
                        xaxis=dict(title="TIME (UTC)", domain=[0, 0.82]),
                        yaxis=dict(title="ALT AGL (FT)", color="#00FF41"),
                        yaxis2=dict(title="VSI (FPM)", overlaying='y', side='right', color="#FF4444"),
                        yaxis3=dict(title="GS (KTS)", overlaying='y', side='right',
                                    anchor='free', position=0.92, color="#FF9F1C"),
                        legend=dict(bgcolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_ed, use_container_width=True, config={'scrollZoom': True})
        else:
            st.info(f"`NO EMERGENCY DESCENT EVENTS DETECTED (no sustained VSI below {EMERG_DESCENT_VSI} FPM)`")

        # VSI timeline
        st.markdown("#### 📉 FULL SORTIE VSI TIMELINE")
        fig_vsi = go.Figure()
        fig_vsi.add_trace(go.Scatter(x=df['Time'], y=df['VSI'], name="VSI (FPM)", line=dict(color="#00FF41", width=1.5)))
        fig_vsi.add_hline(y=EMERG_DESCENT_VSI, line=dict(color="#FF4444", dash="dash", width=1), annotation_text=f"ED Threshold ({EMERG_DESCENT_VSI} FPM)", annotation_font_color="#FF4444")
        fig_vsi.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="VERTICAL SPEED (FPM)", height=300)
        st.plotly_chart(fig_vsi, use_container_width=True, config={'scrollZoom': True})

    # ── TAB 8: PATTERN WORK GRADER ────────────
    with t8:
        st.write("`PATTERN WORK GRADER — AUTO-DETECTS RECTANGULAR TRAFFIC PATTERN LEGS FROM TOUCHDOWN EVENTS`")
        touchdowns = touchdowns_all

        if touchdowns.empty:
            st.warning("`NO RUNWAY CONTACTS DETECTED — PATTERN ANALYSIS REQUIRES AT LEAST ONE TOUCHDOWN`")
        else:
            pattern_alt_input = st.number_input(
                "PATTERN ALTITUDE AGL (FT) — adjust to match your airport's published TPA",
                min_value=400, max_value=2000, value=PATTERN_ALT_AGL, step=50
            )

            right_hand = st.checkbox("🔄 RIGHT-HAND PATTERN", value=False)

            for t_idx, (td_index, td_row) in enumerate(touchdowns.iterrows()):
                legs, runway_hdg, skip_reason = detect_pattern_legs(
                    df, td_row, pattern_alt_agl=pattern_alt_input, right_hand=right_hand
                )

                if skip_reason:
                    st.info(f"`RUNWAY CONTACT {t_idx+1} ({td_row['Time'].strftime('%H:%M:%S')} UTC): {skip_reason}`")
                    continue

                if not legs:
                    st.write(f"`CIRCUIT {t_idx+1}: insufficient pattern data found`")
                    continue

                rwy_num = int(round(runway_hdg / 10)) % 36
                if rwy_num == 0:
                    rwy_num = 36

                st.markdown(f"#### ✈️ CIRCUIT {t_idx+1} — RWY {rwy_num:02d} (HDG {runway_hdg:.0f}°)")

                leg_order = ["CROSSWIND", "DOWNWIND", "BASE", "FINAL"]
                leg_colors = {"CROSSWIND": "#FF9F1C", "DOWNWIND": "#00FF41", "BASE": "#00FFFF", "FINAL": "#FF4444"}
                cols = st.columns(len(leg_order))

                for col, leg_name in zip(cols, leg_order):
                    with col:
                        if leg_name in legs:
                            leg_data  = legs[leg_name]
                            avg_agl   = leg_data['Alt_AGL'].mean()
                            avg_gs    = leg_data['GS'].mean()
                            duration  = leg_data['Dt'].sum()

                            if leg_name == "DOWNWIND":
                                dev = abs(avg_agl - pattern_alt_input)
                                color, grade = acs_status_color(dev, 100, 50)
                                grade_str = f"{grade} ({int(dev)} FT DEV)"
                            elif leg_name == "FINAL":
                                exp_hdg   = runway_hdg
                                hdg_err   = leg_data['Track'].apply(lambda hdg: heading_diff(hdg, exp_hdg)).mean()
                                color, grade = acs_status_color(hdg_err, 10, 5)
                                grade_str = f"{grade} ({hdg_err:.1f}° HDG ERR)"
                            else:
                                color, grade = "#888888", "DATA"
                                grade_str = ""

                            st.markdown(
                                f"<div style='border:1px solid {leg_colors[leg_name]};padding:10px;background:#0F0F0F'>"
                                f"<span style='color:{leg_colors[leg_name]};font-weight:700;font-size:1.1rem'>{leg_name}</span><br>"
                                f"<span style='color:#888;font-size:0.8rem'>AVG ALT: </span><span style='color:#00FFFF'>{int(avg_agl)} FT AGL</span><br>"
                                f"<span style='color:#888;font-size:0.8rem'>AVG GS: </span><span style='color:#FF9F1C'>{int(avg_gs)} KTS</span><br>"
                                f"<span style='color:#888;font-size:0.8rem'>DURATION: </span><span style='color:#C0C0C0'>{int(duration)}s</span><br>"
                                + (f"<span style='color:{color};font-weight:700;font-size:0.85rem'>{grade_str}</span>" if grade_str else "") +
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div style='border:1px dashed #333;padding:10px;background:#080808'>"
                                f"<span style='color:#555;font-weight:700'>{leg_name}</span><br>"
                                f"<span style='color:#444;font-size:0.8rem'>NOT DETECTED</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                # Pattern map for this circuit
                if legs:
                    st.markdown("**PATTERN TRACK MAP:**")
                    fig_pat = go.Figure()
                    for leg_name, leg_data in legs.items():
                        fig_pat.add_trace(go.Scattermapbox(
                            lat=leg_data['Lat'], lon=leg_data['Lon'],
                            mode='lines+markers',
                            line=dict(width=4, color=leg_colors.get(leg_name, "#888")),
                            marker=dict(size=4),
                            name=leg_name,
                        ))
                    all_points  = pd.concat(legs.values())
                    center_lat  = all_points['Lat'].mean()
                    center_lon  = all_points['Lon'].mean()
                    fig_pat.update_layout(
                        mapbox=dict(
                            style="carto-darkmatter",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=13
                        ),
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=0, r=0, b=0, t=0),
                        legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color='#C0C0C0'))
                    )
                    st.plotly_chart(fig_pat, use_container_width=True, key=f"pattern_map_{t_idx}")

                st.markdown("---")
