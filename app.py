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
PATTERN_ALT_AGL        = 800     # ft AGL — nominal pattern altitude (user-adjustable in sidebar)
NAVAID_RADIUS_NM       = 40      # nm — radius for airport/navaid fetch
SLOW_FLIGHT_MARGIN_KTS = 20      # kts — band above stall speed considered "slow flight"
MIN_SLOW_FLIGHT_DUR_S  = 20      # seconds — minimum sustained duration to log slow flight
SLOW_FLIGHT_VSI_BAND   = 200     # fpm — max |VSI| to count as "maintaining altitude"
MIN_TRACK_POINTS       = 20      # minimum coordinate records required for reliable analysis

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
    "CHANDELLE": {
        "ref": "ACS PA.VIII.A / CA.VIII.A",
        "params": {
            "Rollout Heading Error":   {"unit": "°",   "ppl_tol": 10,  "cpl_tol": 5,   "lower_better": True,  "desc": "Heading error from reciprocal at rollout"},
            "Altitude Gain":           {"unit": "ft",  "ppl_tol": 0,   "cpl_tol": 50,  "lower_better": False, "desc": "Net altitude gained through the climbing turn (approximate — ACS also requires rollout just above stall speed, which cannot be verified from GPS track alone)"},
        }
    },
    "LAZY EIGHT": {
        "ref": "ACS PA.VIII.B / CA.VIII.B",
        "params": {
            "Heading Return Error":    {"unit": "°",   "ppl_tol": 15,  "cpl_tol": 10,  "lower_better": True,  "desc": "Heading error vs. entry heading after the full figure-eight"},
            "Altitude Return Dev":     {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Altitude deviation vs. entry altitude after the full figure-eight (approximate — ACS grades each 90°/180° point individually)"},
        }
    },
    "STEEP SPIRAL": {
        "ref": "ACS PA.VIII.C / CA.VIII.C",
        "params": {
            "Rollout Heading Error":   {"unit": "°",   "ppl_tol": 10,  "cpl_tol": 10,  "lower_better": True,  "desc": "Heading error vs. entry heading at rollout"},
        }
    },
    "SLOW FLIGHT": {
        "ref": "ACS PA.V.A / CA.V.A",
        "params": {
            "Altitude Deviation":      {"unit": "ft",  "ppl_tol": 100, "cpl_tol": 50,  "lower_better": True,  "desc": "Max deviation from segment entry altitude while slow"},
            "Airspeed Control Range":  {"unit": "kt",  "ppl_tol": 10,  "cpl_tol": 5,   "lower_better": True,  "desc": "Spread between max and min groundspeed during the segment"},
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
    """
    Parse a Google Earth KML track log into a fully-derived telemetry DataFrame.
    Never raises: on any failure returns (empty_df, human_readable_error_string).
    """
    # ── Parse XML, trying the best available parser ────────────────────
    soup = None
    parse_errors = []
    for parser_name in ('xml', 'lxml-xml', 'html.parser'):
        try:
            soup = BeautifulSoup(file_content, parser_name)
            break
        except Exception as e:  # missing optional parser lib, malformed doc, etc.
            parse_errors.append(f"{parser_name}: {e}")
            soup = None

    if soup is None:
        return pd.DataFrame(), (
            "Could not parse this file as XML/KML. "
            f"Details: {'; '.join(parse_errors)}"
        )

    times  = soup.find_all('when')
    coords = soup.find_all('gx:coord')
    # html.parser strips namespace prefixes — retry without the "gx:" prefix
    if not coords:
        coords = soup.find_all('coord')

    if not times or not coords:
        return pd.DataFrame(), "KML is missing <when> or <gx:coord> tags. Is this a Google Earth track log?"

    if len(times) != len(coords):
        n = min(len(times), len(coords))
    else:
        n = len(times)

    data = []
    missing_alt_count = 0
    for t, c in zip(times[:n], coords[:n]):
        try:
            parts = c.text.split()
            if len(parts) == 3:
                lon, lat, alt_m = float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                lon, lat, alt_m = float(parts[0]), float(parts[1]), 0.0
                missing_alt_count += 1
            else:
                continue
            data.append({
                'Time':    t.text.replace('Z', ''),
                'Lon':     lon,
                'Lat':     lat,
                'Alt_Raw': alt_m * 3.28084,
            })
        except (ValueError, TypeError):
            continue  # skip malformed record rather than crashing the whole parse

    df = pd.DataFrame(data)
    if df.empty:
        return df, "No valid coordinate records found in KML."

    try:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    except Exception:
        return pd.DataFrame(), "Could not parse timestamps in this KML — <when> tags may be malformed."

    df = df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
    # Duplicate timestamps produce zero-duration steps that corrupt rate math downstream.
    df = df.drop_duplicates(subset=['Time'], keep='first').reset_index(drop=True)

    if len(df) < MIN_TRACK_POINTS:
        return pd.DataFrame(), (
            f"Only {len(df)} usable track point(s) found — need at least "
            f"{MIN_TRACK_POINTS} for reliable analysis. Is this a short or corrupted log?"
        )

    if missing_alt_count:
        df.attrs['missing_alt_count'] = missing_alt_count

    df['Dt']   = df['Time'].diff().dt.total_seconds().fillna(1)
    dt_safe    = df['Dt'].replace(0, np.nan)

    try:
        df['Cumulative_Min'] = df['Dt'].cumsum() / 60.0
        df['Alt_Smooth']     = df['Alt_Raw'].rolling(window=7, center=True, min_periods=1).mean()
        df['VSI']            = (df['Alt_Smooth'].diff() / (dt_safe / 60.0)).fillna(0).rolling(5, min_periods=1).mean()

        field_elev   = df['Alt_Smooth'].quantile(ALT_PERCENTILE_FIELD)
        df['Alt_AGL'] = df['Alt_Smooth'] - field_elev

        lat_arr, lon_arr = df['Lat'].to_numpy(), df['Lon'].to_numpy()
        dist, bear = [0.0], [0.0]
        for i in range(1, len(df)):
            dist.append(haversine_distance(lat_arr[i-1], lon_arr[i-1], lat_arr[i], lon_arr[i]))
            bear.append(calculate_bearing(lat_arr[i-1], lon_arr[i-1], lat_arr[i], lon_arr[i]))

        df['GS']    = (pd.Series(dist) / (dt_safe / 3600.0)).fillna(0).rolling(5, min_periods=1).mean()
        df['Track'] = bear

        df['Track_Delta'] = df['Track'].diff().abs()
        df['Track_Delta'] = df['Track_Delta'].apply(lambda x: 360 - x if x > 180 else x).fillna(0)

        # Signed heading change: positive = right/clockwise, negative = left/counterclockwise
        # Normalised to [-180, 180] per step so wrap-arounds don't corrupt the sign.
        raw_delta = df['Track'].diff()
        df['Track_Delta_Signed'] = raw_delta.apply(
            lambda x: ((x + 180) % 360) - 180 if pd.notna(x) else 0
        ).fillna(0)

        df['Turn_Rate'] = (df['Track_Delta'] / dt_safe).rolling(window=3, min_periods=1).mean().fillna(0)

        g              = 32.174
        df['Vel_fps']  = df['GS'] * 1.68781
        turn_rate_rad  = np.radians(df['Turn_Rate'].fillna(0))
        df['Bank_Angle'] = np.degrees(np.arctan((turn_rate_rad * df['Vel_fps']) / g)).fillna(0).clip(0, 89)
        df['G_Load']     = (1 / np.cos(np.radians(df['Bank_Angle']))).clip(1, 3)
        df['Specific_Energy'] = df['Alt_Smooth'] + ((df['Vel_fps'] ** 2) / (2 * g))
    except Exception as e:
        return pd.DataFrame(), f"Unexpected error while computing flight parameters: {e}"

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

    std_key = "360° STEEP TURN"       if "360° STEEP TURN" in label else \
              "CHANDELLE"             if "CHANDELLE" in label else \
              "180° COURSE REVERSAL"  if "180° COURSE REVERSAL" in label else \
              "LAZY EIGHT"            if "LAZY EIGHT" in label else \
              "STEEP SPIRAL"          if "STEEP SPIRAL" in label else \
              "GROUND REFERENCE / S-TURNS"

    entry_track = mdata['Track'].iloc[0]
    exit_track  = mdata['Track'].iloc[-1]

    if std_key in ("360° STEEP TURN", "180° COURSE REVERSAL", "GROUND REFERENCE / S-TURNS"):
        add("Altitude Deviation", max_dev, std_key)

    if std_key == "360° STEEP TURN":
        avg_bank = mdata['Bank_Angle'].median()
        bank_dev = abs(avg_bank - 45)
        add("Bank Dev from 45°", bank_dev, std_key)
        hdg_err = heading_diff(entry_track, exit_track)
        add("Rollout Heading Error", hdg_err, std_key)

    elif std_key == "180° COURSE REVERSAL":
        expected_exit = (entry_track + 180) % 360
        hdg_err       = heading_diff(exit_track, expected_exit)
        add("Rollout Heading Error", hdg_err, std_key)

    elif std_key == "CHANDELLE":
        expected_exit = (entry_track + 180) % 360
        hdg_err       = heading_diff(exit_track, expected_exit)
        add("Rollout Heading Error", hdg_err, std_key)
        alt_gain = row.get('alt_last', row['entry_alt']) - row['entry_alt']
        add("Altitude Gain", alt_gain, std_key)

    elif std_key == "LAZY EIGHT":
        hdg_err = heading_diff(entry_track, exit_track)
        add("Heading Return Error", hdg_err, std_key)
        alt_dev = abs(row.get('alt_last', row['entry_alt']) - row['entry_alt'])
        add("Altitude Return Dev", alt_dev, std_key)

    elif std_key == "STEEP SPIRAL":
        hdg_err = heading_diff(entry_track, exit_track)
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
# SLOW FLIGHT DETECTOR
# ─────────────────────────────────────────────
def detect_slow_flight(df):
    """
    Flag sustained segments where GS sits just above stall speed while the
    aircraft holds altitude (|VSI| stays small) — i.e. actual slow flight,
    as distinct from a stall entry (which has a large negative VSI).
    Returns list of event dicts.
    """
    df = df.copy()
    upper = STALL_SPEED_KTS + SLOW_FLIGHT_MARGIN_KTS
    df['SlowFlight_Flag'] = (
        (df['GS'] >= STALL_SPEED_KTS) &
        (df['GS'] <= upper) &
        (df['VSI'].abs() < SLOW_FLIGHT_VSI_BAND) &
        (df['Alt_AGL'] > 300)
    )
    df['SlowFlight_Event_ID'] = (df['SlowFlight_Flag'] != df['SlowFlight_Flag'].shift()).cumsum()

    events = []
    for eid, edata in df[df['SlowFlight_Flag']].groupby('SlowFlight_Event_ID'):
        duration = edata['Dt'].sum()
        if duration < MIN_SLOW_FLIGHT_DUR_S:
            continue
        entry_alt = edata['Alt_Smooth'].iloc[0]
        alt_dev   = (edata['Alt_Smooth'] - entry_alt).abs().max()
        gs_range  = edata['GS'].max() - edata['GS'].min()
        events.append({
            'time':      edata['Time'].iloc[0],
            'entry_alt': entry_alt,
            'alt_dev':   alt_dev,
            'gs_range':  gs_range,
            'avg_gs':    edata['GS'].mean(),
            'duration':  duration,
            'data':      edata,
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
# MANEUVER CLASSIFIER
# ─────────────────────────────────────────────
def classify_maneuver(row):
    """
    Multi-factor maneuver classifier. Returns (label, is_gradeable, caveats_list).

    Old approach: bucket purely on total_turn degree ranges.
    Problem: total_turn is the SUM OF ABSOLUTE heading deltas — a winding path
    accumulates large values even if the aircraft never actually circles.

    New approach:
      - net_turn  = signed sum of heading deltas → actual net heading swept
      - total_turn = absolute sum → how much turning was done regardless of direction
      - consistency = |net| / total → 1.0 = clean single-direction, 0.0 = pure S-turn

    Each maneuver type requires ALL of its gates to pass.
    Anything that doesn't meet a clear threshold gets labelled honestly
    as UNCLASSIFIED rather than force-fitted to a misleading name.
    """
    total     = row['total_turn']
    net       = row['net_turn']
    abs_net   = abs(net)
    duration  = row['duration']
    max_bank  = row['max_bank']
    mean_agl  = row['mean_agl']
    entry_alt = row['entry_alt']
    alt_last  = row.get('alt_last', entry_alt)
    gs_first  = row.get('gs_first', row['gs_max'])
    gs_last   = row.get('gs_last', row['gs_min'])
    consistency = abs_net / total if total > 0 else 0
    direction   = "RIGHT" if net > 0 else "LEFT"

    alt_change  = alt_last - entry_alt      # + climb, - descent, over the whole maneuver
    speed_loss  = gs_first - gs_last         # + decelerating, - accelerating

    caveats = []

    # ── STEEP SPIRAL ─────────────────────────────────────────────────────
    # Multiple consistent-direction circles (≥ ~2.5 turns) while descending.
    # Checked before the generic "extended circling" catch-all since a spiral
    # is a single clean direction (high consistency), not a wandering hold.
    if consistency >= 0.75 and total >= 900:
        if alt_change <= -300:
            return f"STEEP SPIRAL ({direction})", True, caveats
        else:
            caveats.append(f"ALT CHANGE {alt_change:+.0f}ft — steep spiral requires a sustained descent")
            return f"EXTENDED TURN, NO DESCENT ({direction})", False, caveats

    # ── EXTENDED CIRCLING / HOLD ────────────────────────────────────────
    # High total turn but near-zero consistency = back-and-forth = circling/holding.
    if total >= 700 and consistency < 0.20:
        return "EXTENDED CIRCLING / HOLD", False, []

    # ── 360° STEEP TURN ─────────────────────────────────────────────────
    # Must be one clean circle (consistency ≥ 0.80), net heading swept
    # 300–420°, AND bank must actually be steep (≥ 40°).
    if consistency >= 0.80 and 300 <= abs_net <= 420:
        if max_bank >= 40:
            return f"360° STEEP TURN ({direction})", True, caveats
        else:
            # Circle confirmed but bank is shallow — label honestly
            caveats.append(f"MAX BANK {max_bank:.0f}° — ACS requires 45°, not graded as steep turn")
            return f"360° TURN SHALLOW BANK ({direction})", False, caveats

    # ── CHANDELLE ────────────────────────────────────────────────────────
    # A 180° reversal that also climbs and bleeds airspeed toward stall —
    # distinguishes it from a flat 180° course reversal in the same net-turn band.
    if consistency >= 0.65 and 145 <= abs_net <= 215 and alt_change >= 50 and speed_loss >= 10:
        return f"CHANDELLE ({direction})", True, caveats

    # ── 180° COURSE REVERSAL ────────────────────────────────────────────
    # Single clean reversal: consistency ≥ 0.75, net heading 145–215°.
    if consistency >= 0.75 and 145 <= abs_net <= 215:
        return f"180° COURSE REVERSAL ({direction})", True, caveats

    # ── LAZY EIGHT ───────────────────────────────────────────────────────
    # Alternating-direction turns (low consistency, like S-turns) but flown
    # at altitude rather than over the ground — the key ACS distinguisher.
    if consistency < 0.40 and 500 <= total <= 950 and mean_agl > 1500:
        return "LAZY EIGHT", True, caveats

    # ── S-TURNS / GROUND REFERENCE ──────────────────────────────────────
    # Low consistency (alternating turns) AND low altitude.
    # High altitude with low consistency is just maneuvering, not ground ref.
    if consistency < 0.40 and total >= 150:
        if mean_agl <= 1500:
            return "S-TURNS / GROUND REFERENCE", True, caveats
        else:
            caveats.append(f"MEAN ALT {mean_agl:.0f}ft AGL — ground reference is flown below 1500ft AGL")
            return "MANEUVERING (HIGH ALT S-TURN)", False, caveats

    # ── PARTIAL STEEP TURN ──────────────────────────────────────────────
    # High consistency, significant sweep, but not a full circle.
    # Could be an aborted steep turn, chandelle entry, or spiral.
    if consistency >= 0.75 and 215 < abs_net < 300 and max_bank >= 40:
        caveats.append(f"NET TURN {abs_net:.0f}° — not a complete circle, cannot grade as 360°")
        return f"PARTIAL STEEP TURN ({direction})", False, caveats

    # ── HIGH-ALT SINGLE DIRECTION TURN ─────────────────────────────────
    # Consistent single-direction but doesn't fit any standard maneuver shape.
    if consistency >= 0.75 and abs_net > 80:
        caveats.append(f"NET {abs_net:.0f}°, BANK {max_bank:.0f}° — does not match any ACS maneuver profile")
        return f"TURN {abs_net:.0f}° ({direction})", False, caveats

    # ── CATCH-ALL ───────────────────────────────────────────────────────
    caveats.append(f"CONSISTENCY {consistency:.2f}, NET {abs_net:.0f}° — mixed direction, no clear maneuver type")
    return "UNCLASSIFIED MANEUVERING", False, caveats


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

with st.sidebar:
    st.markdown("### ⚙️ GLOBAL SETTINGS")
    st.number_input(
        "PATTERN ALTITUDE AGL (FT)",
        min_value=400, max_value=2000, value=PATTERN_ALT_AGL, step=50,
        key="pattern_alt_agl",
        help="Adjust to match your airport's published traffic pattern altitude — used by the Pattern Work Grader."
    )
    st.checkbox("🔄 RIGHT-HAND PATTERN", value=False, key="right_hand_pattern")
    st.markdown("---")
    st.caption("`T.G. TACTICAL FLIGHT DEBRIEF`\n\nUpload a Google Earth (.kml) GPS track log to begin.")

uploaded = st.file_uploader("", type=['kml'])


def run_analysis(uploaded):
    # Robust decode: some KML exports aren't strict UTF-8.
    raw_bytes = uploaded.getvalue()
    try:
        raw_content = raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        st.warning("`FILE IS NOT VALID UTF-8 — DECODING WITH ERRORS IGNORED (some characters may be dropped)`")
        raw_content = raw_bytes.decode('utf-8', errors='ignore')

    with st.spinner("PROCESSING TELEMETRY..."):
        df, parse_error = process_kml(raw_content)

    if parse_error:
        st.error(f"⚠️ **KML PARSE FAILURE:** {parse_error}")
        st.stop()

    if df.empty:
        st.warning("No data found in file.")
        st.stop()

    if df.attrs.get('missing_alt_count'):
        st.warning(
            f"`{df.attrs['missing_alt_count']} COORDINATE RECORD(S) HAD NO ALTITUDE DATA — "
            f"DEFAULTED TO 0m. AGL/VSI VALUES MAY BE LESS RELIABLE.`"
        )

    # Use median position for METAR — more representative than first point on
    # cross-country flights that land somewhere other than departure.
    with st.spinner("FETCHING SURFACE WX & NAVAIDS..."):
        metar = fetch_metar(df['Lat'].median(), df['Lon'].median())
    total_mins     = int(df['Dt'].sum() / 60)
    field_elevation = df['Alt_Smooth'].quantile(ALT_PERCENTILE_FIELD)

    # ── TOP METRICS ──────────────────────────
    st.markdown("### 📡 INITIAL CONDITIONS & TELEMETRY")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PEAK ALTITUDE",   f"{int(df['Alt_Smooth'].max())} FT MSL")
    c2.metric("MAX GROUNDSPEED", f"{int(df['GS'].max())} KTS")
    c3.metric("MAX G-LOAD",      f"+{df['G_Load'].max():.1f} G")
    c4.metric("TOTAL SORTIE",    f"{total_mins} MIN")

    if isinstance(metar, dict) and metar.get('rawOb'):
        st.info(f"📍 **SURFACE WX ({metar.get('icaoId', '?')}):** `{metar.get('rawOb')}`")
    else:
        st.caption("`NO SURFACE WX AVAILABLE FOR THIS TRACK'S POSITION/TIME`")

    # ── MANEUVER ANALYSIS ────────────────────
    st.markdown("### 🎯 ACS MANEUVER ANALYSIS")

    df['In_Maneuver'] = (df['Turn_Rate'] > TURN_RATE_THRESHOLD) & (df['GS'] > MIN_AIRSPEED_KTS)
    df['Maneuver_ID'] = (df['In_Maneuver'] != df['In_Maneuver'].shift()).cumsum()

    maneuver_df  = df[df['In_Maneuver']]
    maneuver_agg = maneuver_df.groupby('Maneuver_ID').agg(
        total_turn=('Track_Delta',        'sum'),
        net_turn=  ('Track_Delta_Signed', 'sum'),   # signed: + = right, - = left
        duration=  ('Dt',                 'sum'),
        entry_alt= ('Alt_Smooth',         'first'),
        alt_last=  ('Alt_Smooth',         'last'),
        alt_max=   ('Alt_Smooth',         'max'),
        alt_min=   ('Alt_Smooth',         'min'),
        peak_g=    ('G_Load',             'max'),
        gs_max=    ('GS',                 'max'),
        gs_min=    ('GS',                 'min'),
        gs_first=  ('GS',                 'first'),
        gs_last=   ('GS',                 'last'),
        mean_bank= ('Bank_Angle',         'mean'),
        max_bank=  ('Bank_Angle',         'max'),
        mean_agl=  ('Alt_AGL',            'mean'),
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
        entry_alt  = row['entry_alt']
        max_dev    = max(row['alt_max'] - entry_alt, entry_alt - row['alt_min'])
        mdata      = maneuver_df[maneuver_df['Maneuver_ID'] == row['Maneuver_ID']]

        label, is_gradeable, caveats = classify_maneuver(row)

        # Grade and store (only for labelled gradeable maneuvers)
        grade_rows = grade_maneuver(label, row, mdata) if is_gradeable else []

        if not is_gradeable:
            color, status = "#888888", "UNGRADED"
        elif grade_rows:
            # Overall status reflects the WORST-performing graded parameter,
            # rather than always keying off altitude deviation — this generalizes
            # correctly to maneuvers (Chandelle, Lazy Eight, Steep Spiral) whose
            # primary ACS parameter isn't altitude.
            if all(r['_cpl_pass'] for r in grade_rows):
                color, status = "#00FF41", "CPL GRADE"
            elif all(r['_ppl_pass'] for r in grade_rows):
                color, status = "#FF9F1C", "PPL PASS"
            else:
                color, status = "#FF4444", "ACS BUST"
        else:
            color, status = acs_status_color(max_dev, 100, 50)
        all_graded_maneuvers.append({
            'id': found_mnvrs, 'label': label, 'color': color, 'status': status,
            'total_turn': total_turn, 'row': row, 'mdata': mdata,
            'grade_rows': grade_rows, 'max_dev': max_dev,
            'is_gradeable': is_gradeable, 'caveats': caveats,
        })

        consistency = abs(row['net_turn']) / total_turn if total_turn > 0 else 0

        with st.expander(f"MNVR {found_mnvrs} | {label} | NET: {abs(row['net_turn']):.0f}°"):
            col_s, col_b = st.columns([3, 1])
            with col_s:
                st.markdown(
                    f"**STATUS:** <span style='color:{color}'>{status}</span>"
                    + (f" — ALT DEV: {int(max_dev)} FT" if is_gradeable else ""),
                    unsafe_allow_html=True
                )
                st.write(
                    f"`ENTRY: {int(entry_alt)} FT MSL | DUR: {int(duration)}s | "
                    f"PEAK G: +{row['peak_g']:.1f}G | MAX BANK: {row['max_bank']:.0f}° | "
                    f"CONSISTENCY: {consistency:.2f}`"
                )
                for caveat in caveats:
                    st.warning(f"⚠️ {caveat}")
            with col_b:
                st.metric("NET TURN",   f"{abs(row['net_turn']):.0f}°")
                st.metric("TOTAL TURN", f"{total_turn:.0f}°")
                st.metric("MAX BANK",   f"{row['max_bank']:.0f}°")

            if "360°" in label and is_gradeable:
                wind_est = (row['gs_max'] - row['gs_min']) / 2
                st.write(f"`~EST WINDS ALOFT: {int(wind_est)} KTS (valid only for complete circular turns)`")

            fig_mnvr = px.line_map(mdata, lat="Lat", lon="Lon", zoom=14.5, height=300)
            fig_mnvr.update_traces(line=dict(color=color if is_gradeable else '#888888', width=4))
            fig_mnvr.update_layout(
                map_style="carto-darkmatter", template="plotly_dark",
                margin=dict(l=0, r=0, b=0, t=0),
                map=dict(center=dict(lat=mdata['Lat'].mean(), lon=mdata['Lon'].mean()))
            )
            st.plotly_chart(fig_mnvr, width='stretch', key=f"mnvr_map_{row['Maneuver_ID']}_{found_mnvrs}")

    if found_mnvrs == 0:
        st.info("`NO GRADABLE MANEUVERS DETECTED — minimum 15s duration and 100° heading sweep required`")

    # ── Pre-compute touchdown data once (used by Tab 4 and Tab 8) ──────────
    df['On_Ground']         = df['Alt_AGL'] < AGL_TOUCHDOWN_FT
    df['Touchdown_Trigger'] = (df['On_Ground'] == True) & (df['On_Ground'].shift(1) == False)
    touchdowns_all          = df[df['Touchdown_Trigger']]

    # ── Pre-compute all event detectors once (used by Mission Summary + Tabs 6/7) ──
    stall_events      = detect_stalls(df)
    slow_flight_events = detect_slow_flight(df)
    ed_events         = detect_emergency_descents(df)

    # ── MISSION SUMMARY SCORECARD ─────────────
    st.markdown("### 🧾 MISSION SUMMARY")
    graded = [gm for gm in all_graded_maneuvers if gm['is_gradeable']]
    cpl_count = sum(1 for gm in graded if gm['status'] == "CPL GRADE")
    ppl_count = sum(1 for gm in graded if gm['status'] == "PPL PASS")
    bust_count = sum(1 for gm in graded if gm['status'] == "ACS BUST")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("MANEUVERS GRADED", f"{len(graded)}", help="Total ACS-gradeable maneuvers detected")
    sc2.metric("CPL / PPL / BUST", f"{cpl_count} / {ppl_count} / {bust_count}")
    sc3.metric("STALL / SLOW-FLT EVENTS", f"{len(stall_events)} / {len(slow_flight_events)}")
    sc4.metric("RUNWAY CONTACTS", f"{len(touchdowns_all)}")

    if bust_count > 0:
        st.warning(f"`⚠️ {bust_count} MANEUVER(S) OUTSIDE PPL ACS TOLERANCE — see ACS STANDARDS PANEL tab for detail`")
    elif len(graded) > 0:
        st.success("`✅ ALL GRADED MANEUVERS WITHIN AT LEAST PPL ACS TOLERANCE`")

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

        fig_map = px.scatter_map(
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
                fig_map.add_trace(go.Scattermap(
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
                        fig_map.add_trace(go.Scattermap(
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
            map_style="carto-darkmatter", template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color='#C0C0C0'))
        )
        st.plotly_chart(fig_map, width='stretch', config={'scrollZoom': True})

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
            st.plotly_chart(fig_3d, width='stretch', config={
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
        st.plotly_chart(fig_aero, width='stretch', config={'scrollZoom': True})

        fig_bank = go.Figure()
        fig_bank.add_trace(go.Scatter(x=df['Time'], y=df['Bank_Angle'], name="ESTIMATED BANK (°)", line=dict(color="#00FFFF", width=2)))
        fig_bank.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="BANK ANGLE (°)", height=300)
        st.plotly_chart(fig_bank, width='stretch', config={'scrollZoom': True})

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
            st.plotly_chart(fig_glide, width='stretch', config={'scrollZoom': True})
            st.plotly_chart(fig_speed, width='stretch', config={'scrollZoom': True})
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
            ref_df, width='stretch', hide_index=True,
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
                if not gm['is_gradeable']:
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

        if stall_events:
            st.info(f"⚠️ **{len(stall_events)} STALL EVENT(S) DETECTED**")

            for i, ev in enumerate(stall_events):
                alt_loss     = ev['alt_loss']
                color, grade = acs_status_color(alt_loss, 100, 50)

                with st.expander(f"STALL {i+1} | {ev['time'].strftime('%H:%M:%S')} UTC | ALT LOSS: {int(alt_loss)} FT | {grade}"):
                    col_a, col_b = st.columns(2)
                    col_a.metric("ENTRY ALT",    f"{int(ev['entry_alt'])} FT MSL")
                    col_b.metric("RECOVERY ALT", f"{int(ev['recovery_alt'])} FT MSL")
                    col_c, col_d = st.columns(2)
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
                    st.plotly_chart(fig_stall, width='stretch', config={'scrollZoom': True})
        else:
            st.success(f"`✅ NO STALL EVENTS DETECTED (GS never dropped below {STALL_SPEED_KTS} KTS with high sink rate while airborne)`")

        st.markdown("---")
        st.write(
            f"`SLOW FLIGHT DETECTOR — FLAGS ≥{MIN_SLOW_FLIGHT_DUR_S}s SUSTAINED GS BETWEEN "
            f"{STALL_SPEED_KTS}–{STALL_SPEED_KTS + SLOW_FLIGHT_MARGIN_KTS} KTS WHILE HOLDING ALTITUDE`"
        )

        if slow_flight_events:
            st.info(f"🐢 **{len(slow_flight_events)} SLOW FLIGHT SEGMENT(S) DETECTED**")

            for i, ev in enumerate(slow_flight_events):
                alt_dev  = ev['alt_dev']
                gs_range = ev['gs_range']
                color_alt, grade_alt = acs_status_color(alt_dev, 100, 50)
                color_gs,  grade_gs  = acs_status_color(gs_range, 10, 5)
                overall = "CPL GRADE" if (grade_alt == "CPL GRADE" and grade_gs == "CPL GRADE") else \
                          "ACS BUST" if (grade_alt == "ACS BUST" or grade_gs == "ACS BUST") else "PPL PASS"

                with st.expander(f"SLOW FLIGHT {i+1} | {ev['time'].strftime('%H:%M:%S')} UTC | {int(ev['duration'])}s | {overall}"):
                    col_a, col_b = st.columns(2)
                    col_a.metric("ENTRY ALT",  f"{int(ev['entry_alt'])} FT MSL")
                    col_b.metric("AVG GS",     f"{ev['avg_gs']:.0f} KTS")
                    col_c, col_d = st.columns(2)
                    col_c.metric("ALT DEVIATION", f"{int(alt_dev)} FT")
                    col_d.metric("GS RANGE",      f"{gs_range:.0f} KTS")

                    ppl_alt, cpl_alt = alt_dev <= 100, alt_dev <= 50
                    ppl_gs,  cpl_gs  = gs_range <= 10, gs_range <= 5
                    st.markdown(render_acs_table([
                        {
                            "Parameter": "Altitude Deviation", "Your Value": f"{int(alt_dev)} ft",
                            "PPL Tol": "≤100 ft", "CPL Tol": "≤50 ft",
                            "PPL": "✅ PASS" if ppl_alt else "❌ BUST", "CPL": "✅ PASS" if cpl_alt else "❌ BUST",
                            "Description": "Max deviation from segment entry altitude while slow",
                            "_ppl_pass": ppl_alt, "_cpl_pass": cpl_alt,
                        },
                        {
                            "Parameter": "Airspeed Control Range", "Your Value": f"{gs_range:.1f} kt",
                            "PPL Tol": "≤10 kt", "CPL Tol": "≤5 kt",
                            "PPL": "✅ PASS" if ppl_gs else "❌ BUST", "CPL": "✅ PASS" if cpl_gs else "❌ BUST",
                            "Description": "Spread between max and min groundspeed during the segment",
                            "_ppl_pass": ppl_gs, "_cpl_pass": cpl_gs,
                        },
                    ]), unsafe_allow_html=True)

                    edata = ev['data']
                    fig_sf = go.Figure()
                    fig_sf.add_trace(go.Scatter(x=edata['Time'], y=edata['Alt_AGL'], name="ALT AGL (FT)", line=dict(color="#00FF41", width=2)))
                    fig_sf.add_trace(go.Scatter(x=edata['Time'], y=edata['GS'],      name="GS (KTS)",     line=dict(color="#FF9F1C", width=2), yaxis="y2"))
                    fig_sf.update_layout(
                        template="plotly_dark", height=300,
                        xaxis=dict(title="TIME (UTC)", domain=[0, 0.9]),
                        yaxis=dict(title="ALT AGL (FT)", color="#00FF41"),
                        yaxis2=dict(title="GS (KTS)", overlaying='y', side='right', color="#FF9F1C"),
                        legend=dict(bgcolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_sf, width='stretch', config={'scrollZoom': True})
        else:
            st.success("`✅ NO SUSTAINED SLOW FLIGHT SEGMENTS DETECTED`")

        # Slow-flight timeline: show GS across the whole flight
        st.markdown("#### 📉 FULL SORTIE GROUNDSPEED TIMELINE")
        fig_gs = go.Figure()
        fig_gs.add_trace(go.Scatter(x=df['Time'], y=df['GS'], name="GS (KTS)", line=dict(color="#FF9F1C", width=1.5)))
        fig_gs.add_hline(y=STALL_SPEED_KTS, line=dict(color="#FF4444", dash="dash", width=1), annotation_text=f"Stall Threshold ({STALL_SPEED_KTS} KTS)", annotation_font_color="#FF4444")
        fig_gs.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="GROUNDSPEED (KTS)", height=300)
        st.plotly_chart(fig_gs, width='stretch', config={'scrollZoom': True})

    # ── TAB 7: EMERGENCY DESCENT ──────────────
    with t7:
        st.write(f"`EMERGENCY DESCENT PROFILER — FLAGS SUSTAINED VSI < {EMERG_DESCENT_VSI} FPM FOR > {EMERG_DESCENT_MIN_S}s WHILE AIRBORNE`")

        if ed_events:
            st.info(f"🔻 **{len(ed_events)} EMERGENCY DESCENT EVENT(S) DETECTED**")
            for i, ev in enumerate(ed_events):
                peak_vs_abs  = abs(ev['peak_vs'])
                color, grade = acs_status_color(peak_vs_abs, 1500, 1500, lower_better=False)

                with st.expander(f"ED {i+1} | {ev['time'].strftime('%H:%M:%S')} UTC | PEAK {int(peak_vs_abs)} FPM | {grade}"):
                    col_a, col_b = st.columns(2)
                    col_a.metric("ENTRY ALT",   f"{int(ev['entry_alt'])} FT MSL")
                    col_b.metric("EXIT ALT",    f"{int(ev['exit_alt'])} FT MSL")
                    col_c, col_d = st.columns(2)
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
                    st.plotly_chart(fig_ed, width='stretch', config={'scrollZoom': True})
        else:
            st.info(f"`NO EMERGENCY DESCENT EVENTS DETECTED (no sustained VSI below {EMERG_DESCENT_VSI} FPM)`")

        # VSI timeline
        st.markdown("#### 📉 FULL SORTIE VSI TIMELINE")
        fig_vsi = go.Figure()
        fig_vsi.add_trace(go.Scatter(x=df['Time'], y=df['VSI'], name="VSI (FPM)", line=dict(color="#00FF41", width=1.5)))
        fig_vsi.add_hline(y=EMERG_DESCENT_VSI, line=dict(color="#FF4444", dash="dash", width=1), annotation_text=f"ED Threshold ({EMERG_DESCENT_VSI} FPM)", annotation_font_color="#FF4444")
        fig_vsi.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="VERTICAL SPEED (FPM)", height=300)
        st.plotly_chart(fig_vsi, width='stretch', config={'scrollZoom': True})

    # ── TAB 8: PATTERN WORK GRADER ────────────
    with t8:
        st.write("`PATTERN WORK GRADER — AUTO-DETECTS RECTANGULAR TRAFFIC PATTERN LEGS FROM TOUCHDOWN EVENTS`")
        touchdowns = touchdowns_all

        if touchdowns.empty:
            st.warning("`NO RUNWAY CONTACTS DETECTED — PATTERN ANALYSIS REQUIRES AT LEAST ONE TOUCHDOWN`")
        else:
            st.caption("`Pattern altitude & turn direction are set in the sidebar ⟵`")
            pattern_alt_input = st.session_state.get('pattern_alt_agl', PATTERN_ALT_AGL)
            right_hand        = st.session_state.get('right_hand_pattern', False)

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
                        fig_pat.add_trace(go.Scattermap(
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
                        map=dict(
                            style="carto-darkmatter",
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=13
                        ),
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=0, r=0, b=0, t=0),
                        legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color='#C0C0C0'))
                    )
                    st.plotly_chart(fig_pat, width='stretch', key=f"pattern_map_{t_idx}")

                st.markdown("---")


# ─────────────────────────────────────────────
# DRIVER — runs the analysis with a safety net so a single malformed
# track or an unexpected data shape shows a friendly message instead
# of a raw traceback filling the page.
# ─────────────────────────────────────────────
if uploaded:
    try:
        run_analysis(uploaded)
    except Exception as e:
        st.error(
            "⚠️ **UNEXPECTED ERROR DURING ANALYSIS** — this flight log may contain "
            "data this build doesn't handle gracefully yet."
        )
        with st.expander("Technical details"):
            st.exception(e)
else:
    st.info("`AWAITING KML UPLOAD — DRAG A GOOGLE EARTH TRACK LOG ABOVE TO BEGIN DEBRIEF`")
