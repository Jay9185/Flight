import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import math
import requests
from datetime import datetime

# --- UI Configuration: Tactical Industrial ---
st.set_page_config(
    page_title="T.G. | Tactical Flight Debrief",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="collapsed"
)

# Custom MFD-Style Styling
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
    div[data-testid="stNotification"] { background-color: #0F0F0F !important; border-radius: 0px !important; }
    div[data-baseweb="select"] > div { background-color: #0F0F0F !important; border: 1px solid #FF9F1C !important; color: #00FF41 !important; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
TURN_RATE_THRESHOLD   = 1.8    # deg/s — minimum to classify as in-maneuver
MIN_AIRSPEED_KTS      = 40     # kts — below this, maneuver detection is suppressed
AGL_TOUCHDOWN_FT      = 75     # ft AGL — considered runway contact
MIN_AIRBORNE_AGL_FT   = 200    # ft AGL above field to be considered airborne
MIN_MANEUVER_DUR_S    = 15     # seconds — minimum duration to log a maneuver
MIN_MANEUVER_TURN_DEG = 100    # degrees — minimum heading change to log a maneuver
ALT_PERCENTILE_FIELD  = 0.02   # percentile used to estimate field elevation
METAR_TTL_S           = 600    # seconds — how long to cache METAR data
GS_TAXI_THRESHOLD     = 35     # kts — below this counts as taxiing for 3D view

# --- Math & Aerodynamic Logic ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

@st.cache_data(ttl=METAR_TTL_S)  # FIX #5: TTL prevents stale METARs being served indefinitely
def fetch_metar(lat, lon):
    try:
        url = f"https://aviationweather.gov/api/data/metar?lat={lat}&lon={lon}&distance=25&format=json"
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and res.json():
            return res.json()[0]
    except (requests.RequestException, ValueError, KeyError):  # FIX #11: explicit exception types
        return None

# --- Core Data Pipeline ---
@st.cache_data
def process_kml(file_content):
    soup = BeautifulSoup(file_content, 'xml')
    times = soup.find_all('when')
    coords = soup.find_all('gx:coord')

    # FIX #12: Validate KML structure before processing
    if not times or not coords:
        return pd.DataFrame(), "KML file is missing <when> or <gx:coord> tags. Is this a Google Earth track log?"

    data = []
    for t, c in zip(times, coords):
        c_parts = c.text.split()
        if len(c_parts) == 3:
            data.append({
                'Time': t.text.replace('Z', ''),
                'Lon': float(c_parts[0]),
                'Lat': float(c_parts[1]),
                'Alt_Raw': float(c_parts[2]) * 3.28084
            })

    df = pd.DataFrame(data)
    if df.empty:
        return df, "No valid coordinate records found in KML file."

    df['Time'] = pd.to_datetime(df['Time'])
    df['Dt'] = df['Time'].diff().dt.total_seconds().fillna(1)

    # FIX #1: Replace zero Dt with NaN before division to avoid inf groundspeed/VSI
    dt_safe = df['Dt'].replace(0, np.nan)

    df['Cumulative_Min'] = df['Dt'].cumsum() / 60.0

    df['Alt_Smooth'] = df['Alt_Raw'].rolling(window=7, center=True, min_periods=1).mean()
    df['VSI'] = (df['Alt_Smooth'].diff() / (dt_safe / 60.0)).fillna(0).rolling(5).mean()

    # FIX #2: Use low percentile instead of .min() for more robust field elevation
    field_elev = df['Alt_Smooth'].quantile(ALT_PERCENTILE_FIELD)
    df['Alt_AGL'] = df['Alt_Smooth'] - field_elev

    dist, bear = [0], [0]
    for i in range(1, len(df)):
        dist.append(haversine_distance(
            df.iloc[i - 1]['Lat'], df.iloc[i - 1]['Lon'],
            df.iloc[i]['Lat'], df.iloc[i]['Lon']
        ))
        bear.append(calculate_bearing(
            df.iloc[i - 1]['Lat'], df.iloc[i - 1]['Lon'],
            df.iloc[i]['Lat'], df.iloc[i]['Lon']
        ))

    # FIX #1 (continued): Use dt_safe in GS calculation
    df['GS'] = (pd.Series(dist) / (dt_safe / 3600.0)).fillna(0).rolling(5).mean()
    df['Track'] = bear

    df['Track_Delta'] = df['Track'].diff().abs()
    df['Track_Delta'] = df['Track_Delta'].apply(lambda x: 360 - x if x > 180 else x).fillna(0)

    # FIX #3 (continued): Use dt_safe in Turn_Rate
    df['Turn_Rate'] = (df['Track_Delta'] / dt_safe).rolling(window=3).mean().fillna(0)

    g = 32.174
    df['Vel_fps'] = df['GS'] * 1.68781

    # FIX #4: Ensure Turn_Rate NaNs are filled before arctan to prevent silent NaN propagation
    turn_rate_rad = np.radians(df['Turn_Rate'].fillna(0))
    df['Bank_Angle'] = np.degrees(np.arctan((turn_rate_rad * df['Vel_fps']) / g)).fillna(0)

    df['G_Load'] = (1 / np.cos(np.radians(df['Bank_Angle']))).clip(1, 3)

    # FIX #9: Units are consistent (ft + ft) — label will clarify in chart
    df['Specific_Energy'] = df['Alt_Smooth'] + ((df['Vel_fps'] ** 2) / (2 * g))

    return df, None  # Return df + no error

# --- UI Layout ---
st.title("🛰️ T.G. TACTICAL FLIGHT DEBRIEF")
st.markdown("`SYSTEM STATUS: ONLINE | ADVANCED AERO ENGINE ARMED`")

uploaded = st.file_uploader("", type=['kml'])

if uploaded:
    raw_content = uploaded.getvalue().decode('utf-8')

    # FIX #12: Unpack the error string returned from process_kml
    result = process_kml(raw_content)
    df, parse_error = result

    if parse_error:
        st.error(f"⚠️ **KML PARSE FAILURE:** {parse_error}")
        st.stop()

    if not df.empty:
        metar = fetch_metar(df['Lat'].iloc[0], df['Lon'].iloc[0])
        total_mins = int(df['Dt'].sum() / 60)
        field_elevation = df['Alt_Smooth'].quantile(ALT_PERCENTILE_FIELD)

        st.markdown("### 📡 INITIAL CONDITIONS & TELEMETRY")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PEAK ALTITUDE", f"{int(df['Alt_Smooth'].max())} FT MSL")
        col2.metric("MAX GROUNDSPEED", f"{int(df['GS'].max())} KTS")
        col3.metric("MAX G-LOAD", f"+{df['G_Load'].max():.1f} G")
        col4.metric("TOTAL SORTIE", f"{total_mins} MIN")

        if metar:
            st.info(f"📍 **SURFACE WX ({metar.get('icaoId')}):** `{metar.get('rawOb')}`")

        st.markdown("### 🎯 ACS MANEUVER ANALYSIS")

        # FIX #8: Gate maneuver detection on minimum airspeed to suppress taxi/GPS jitter
        df['In_Maneuver'] = (df['Turn_Rate'] > TURN_RATE_THRESHOLD) & (df['GS'] > MIN_AIRSPEED_KTS)
        df['Maneuver_ID'] = (df['In_Maneuver'] != df['In_Maneuver'].shift()).cumsum()

        # FIX #6: Pre-aggregate maneuver stats once instead of hitting pandas per-loop
        maneuver_df = df[df['In_Maneuver']]
        maneuver_agg = maneuver_df.groupby('Maneuver_ID').agg(
            total_turn=('Track_Delta', 'sum'),
            duration=('Dt', 'sum'),
            entry_alt=('Alt_Smooth', 'first'),
            alt_max=('Alt_Smooth', 'max'),
            alt_min=('Alt_Smooth', 'min'),
            peak_g=('G_Load', 'max'),
            gs_max=('GS', 'max'),
            gs_min=('GS', 'min'),
        ).reset_index()

        found_mnvrs = 0
        for _, row in maneuver_agg.iterrows():
            total_turn = row['total_turn']
            duration   = row['duration']

            if duration < MIN_MANEUVER_DUR_S or total_turn < MIN_MANEUVER_TURN_DEG:
                continue

            found_mnvrs += 1
            entry_alt = row['entry_alt']
            max_dev = max(row['alt_max'] - entry_alt, entry_alt - row['alt_min'])
            mdata = maneuver_df[maneuver_df['Maneuver_ID'] == row['Maneuver_ID']]

            if total_turn >= 700:
                label, color, status = "EXTENDED CIRCLING / HOLD", "#888888", "UNGRADED"
            elif 320 <= total_turn <= 400:
                label = "360° STEEP TURN"
                color  = "#00FF41" if max_dev <= 50 else ("#FF9F1C" if max_dev <= 100 else "#FF0000")
                status = "CPL GRADE" if max_dev <= 50 else ("PPL PASS" if max_dev <= 100 else "ACS BUST")
            elif 150 <= total_turn <= 210:
                label = "180° COURSE REVERSAL"
                color  = "#00FF41" if max_dev <= 50 else ("#FF9F1C" if max_dev <= 100 else "#FF0000")
                status = "CPL GRADE" if max_dev <= 50 else ("PPL PASS" if max_dev <= 100 else "ACS BUST")
            else:
                label = "GROUND REFERENCE / S-TURNS"
                color  = "#00FF41" if max_dev <= 50 else ("#FF9F1C" if max_dev <= 100 else "#FF0000")
                status = "CPL GRADE" if max_dev <= 50 else ("PPL PASS" if max_dev <= 100 else "ACS BUST")

            with st.expander(f"MNVR {found_mnvrs} | {label} | TURN: {int(total_turn)}°"):
                st.markdown(
                    f"**STATUS:** <span style='color:{color}'>{status}</span> (DEV: {int(max_dev)}FT)",
                    unsafe_allow_html=True
                )
                st.write(f"`ENTRY ALT: {int(entry_alt)} FT | DURATION: {int(duration)}s | PEAK G: +{row['peak_g']:.1f}G`")

                if "360°" in label:
                    # FIX #7: Caveat the wind estimate clearly in the UI
                    wind_est = (row['gs_max'] - row['gs_min']) / 2
                    st.write(f"`~EST WINDS ALOFT: {int(wind_est)} KTS (valid only for complete circular turns)`")

                fig_mnvr = px.line_mapbox(mdata, lat="Lat", lon="Lon", zoom=14.5, height=300)
                fig_mnvr.update_traces(line=dict(color='#00FF41', width=4))
                fig_mnvr.update_layout(
                    mapbox_style="carto-darkmatter",
                    template="plotly_dark",
                    margin=dict(l=0, r=0, b=0, t=0),
                    mapbox=dict(center=dict(lat=mdata['Lat'].mean(), lon=mdata['Lon'].mean()))
                )
                st.plotly_chart(fig_mnvr, use_container_width=True, key=f"mnvr_map_{row['Maneuver_ID']}_{found_mnvrs}")

        st.markdown("### 🗺️ SPATIAL TELEMETRY & PHYSICS")
        t1, t2, t3, t4 = st.tabs(["2D DYNAMIC MAP", "3D AIRWAY CORRIDOR", "AERODYNAMICS", "TOUCH & GO PROFILER"])

        with t1:
            st.write("`SELECT AVIONICS OVERLAY METRIC:`")
            map_metrics = {
                "ALTITUDE (AGL)":      ["Alt_AGL",    "Viridis", [0, 3000]],
                "VERTICAL SPEED (FPM)":["VSI",        "RdBu_r",  [-1000, 1000]],
                "GROUNDSPEED (KTS)":   ["GS",         "Inferno", [df['GS'].min(), df['GS'].max()]],
                "BANK ANGLE (°)":      ["Bank_Angle", "Plasma",  [0, 60]],
                "G-LOAD (G)":          ["G_Load",     "Turbo",   [1, 2]],
                "TURN RATE (°/SEC)":   ["Turn_Rate",  "Plasma",  [0, 4]]
            }

            selected_metric = st.selectbox("", list(map_metrics.keys()))
            active_col, active_colorscale, active_range = map_metrics[selected_metric]

            fig_map = px.scatter_mapbox(
                df, lat="Lat", lon="Lon", color=active_col,
                color_continuous_scale=active_colorscale, range_color=active_range,
                zoom=10, height=650,
                hover_data=["Alt_AGL", "GS", "VSI", "Bank_Angle"]
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})

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
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['resetCameraDefault3d']
                })
            else:
                st.warning("`WARNING: NO DATA REMAINS AFTER CURRENT CROP SELECTION.`")

        with t3:
            # FIX #9: Clarified energy state chart axis label with units
            st.write("`SPECIFIC ENERGY STATE & ESTIMATED BANK ANGLES`")
            fig_aero = go.Figure()
            fig_aero.add_trace(go.Scatter(
                x=df['Time'], y=df['Specific_Energy'],
                name="SPECIFIC ENERGY",
                line=dict(color="#FF9F1C", width=2)
            ))
            fig_aero.add_trace(go.Scatter(
                x=df['Time'], y=df['Alt_Smooth'],
                name="POTENTIAL ENERGY (ALT)",
                line=dict(color="#00FF41", width=2, dash='dot')
            ))
            fig_aero.update_layout(
                template="plotly_dark",
                xaxis_title="TIME (UTC)",
                yaxis_title="ENERGY STATE (ft equivalent)",  # FIX #9: explicit units
                height=400
            )
            st.plotly_chart(fig_aero, use_container_width=True, config={'scrollZoom': True})

            fig_bank = go.Figure()
            fig_bank.add_trace(go.Scatter(
                x=df['Time'], y=df['Bank_Angle'],
                name="ESTIMATED BANK (°)",
                line=dict(color="#00FFFF", width=2)
            ))
            fig_bank.update_layout(
                template="plotly_dark",
                xaxis_title="TIME (UTC)",
                yaxis_title="BANK ANGLE (°)",
                height=300
            )
            st.plotly_chart(fig_bank, use_container_width=True, config={'scrollZoom': True})

        with t4:
            st.write("`TOUCH & GO DETECTOR: 90-SECOND GLIDEPATH ISOLATION`")

            df['On_Ground'] = df['Alt_AGL'] < AGL_TOUCHDOWN_FT
            df['Touchdown_Trigger'] = (df['On_Ground'] == True) & (df['On_Ground'].shift(1) == False)
            touchdowns = df[df['Touchdown_Trigger']]

            if not touchdowns.empty:
                st.info(f"🛬 **DETECTED {len(touchdowns)} RUNWAY CONTACT(S)**")

                fig_glide = go.Figure()
                fig_speed = go.Figure()

                for idx, (td_index, td_row) in enumerate(touchdowns.iterrows()):
                    start_time = td_row['Time'] - pd.Timedelta(seconds=90)
                    approach_data = df[(df['Time'] >= start_time) & (df['Time'] <= td_row['Time'])].copy()

                    if len(approach_data) > 10:
                        approach_data['Sec_To_TD'] = (approach_data['Time'] - td_row['Time']).dt.total_seconds()
                        app_name = f"APPROACH {idx + 1}"

                        fig_glide.add_trace(go.Scatter(
                            x=approach_data['Sec_To_TD'], y=approach_data['Alt_AGL'],
                            mode='lines', name=app_name, line=dict(width=3)
                        ))
                        fig_speed.add_trace(go.Scatter(
                            x=approach_data['Sec_To_TD'], y=approach_data['GS'],
                            mode='lines', name=app_name, line=dict(width=3)
                        ))

                fig_glide.update_layout(
                    template="plotly_dark", title="GLIDEPATH PROFILE (AGL)",
                    xaxis_title="SECONDS TO TOUCHDOWN", yaxis_title="ALTITUDE (FT AGL)",
                    hovermode="x unified", height=400
                )
                fig_speed.update_layout(
                    template="plotly_dark", title="AIRSPEED DECAY PROFILE",
                    xaxis_title="SECONDS TO TOUCHDOWN", yaxis_title="GROUNDSPEED (KTS)",
                    hovermode="x unified", height=300
                )

                st.plotly_chart(fig_glide, use_container_width=True, config={'scrollZoom': True})
                st.plotly_chart(fig_speed, use_container_width=True, config={'scrollZoom': True})
            else:
                st.warning("`NO RUNWAY CONTACT DETECTED IN LOG.`")
