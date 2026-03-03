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

    .stApp {
        background-color: #050505;
        color: #C0C0C0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    h1, h2, h3 {
        color: #FF9F1C !important; 
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 2px solid #1A1A1A;
        padding-bottom: 5px;
    }

    div[data-testid="metric-container"] {
        background-color: #0F0F0F;
        border: 1px solid #333333;
        padding: 20px;
        border-radius: 0px;
        border-left: 5px solid #FF9F1C;
    }
    div[data-testid="metric-container"] label {
        color: #888888 !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="metric-container"] div {
        color: #00FF41 !important; 
        font-weight: 700 !important;
    }

    .streamlit-expanderHeader {
        background-color: #0F0F0F !important;
        color: #FF9F1C !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }

    .stFileUploader {
        border: 1px dashed #FF9F1C;
        background-color: #0F0F0F;
    }

    div[data-testid="stNotification"] {
        background-color: #0F0F0F !important;
        border-radius: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Math & Aerodynamic Logic ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi, delta_lambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

@st.cache_data
def fetch_metar(lat, lon):
    try:
        url = f"https://aviationweather.gov/api/data/metar?lat={lat}&lon={lon}&distance=25&format=json"
        res = requests.get(url, timeout=5)
        if res.status_code == 200 and res.json():
            return res.json()[0]
    except: return None

# --- Core Data Pipeline ---
@st.cache_data
def process_kml(file_content):
    soup = BeautifulSoup(file_content, 'xml')
    times, coords = soup.find_all('when'), soup.find_all('gx:coord')
    
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
    if df.empty: return df

    df['Time'] = pd.to_datetime(df['Time'])
    df['Dt'] = df['Time'].diff().dt.total_seconds().fillna(1)
    
    df['Alt_Smooth'] = df['Alt_Raw'].rolling(window=7, center=True, min_periods=1).mean()
    df['VSI'] = (df['Alt_Smooth'].diff() / (df['Dt'] / 60.0)).fillna(0).rolling(5).mean()
    
    dist, bear = [0], [0]
    for i in range(1, len(df)):
        dist.append(haversine_distance(df.iloc[i-1]['Lat'], df.iloc[i-1]['Lon'], df.iloc[i]['Lat'], df.iloc[i]['Lon']))
        bear.append(calculate_bearing(df.iloc[i-1]['Lat'], df.iloc[i-1]['Lon'], df.iloc[i]['Lat'], df.iloc[i]['Lon']))
        
    df['GS'] = (pd.Series(dist) / (df['Dt'] / 3600.0)).fillna(0).rolling(5).mean()
    df['Track'] = bear
    
    df['Track_Delta'] = df['Track'].diff().abs()
    df['Track_Delta'] = df['Track_Delta'].apply(lambda x: 360 - x if x > 180 else x).fillna(0)
    df['Turn_Rate'] = (df['Track_Delta'] / df['Dt']).rolling(window=3).mean()
    
    return df

# --- UI Layout ---
st.title("🛰️ T.G. TACTICAL FLIGHT DEBRIEF")
st.markdown("`SYSTEM STATUS: ONLINE | AWAITING TELEMETRY INGESTION`")

uploaded = st.file_uploader("", type=['kml'])

if uploaded:
    raw_content = uploaded.getvalue().decode('utf-8')
    df = process_kml(raw_content)
    
    if not df.empty:
        metar = fetch_metar(df['Lat'].iloc[0], df['Lon'].iloc[0])
        
        st.markdown("### 📡 INITIAL CONDITIONS & TELEMETRY")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PEAK ALTITUDE", f"{int(df['Alt_Smooth'].max())} FT MSL")
        col2.metric("MAX GROUNDSPEED", f"{int(df['GS'].max())} KTS")
        col3.metric("MAX SINK RATE", f"{int(df['VSI'].min())} FPM")
        col4.metric("TOTAL SORTIE", f"{int(df['Dt'].sum()/60)} MIN")

        if metar:
            st.info(f"📍 **SURFACE WX ({metar.get('icaoId')}):** `{metar.get('rawOb')}`")

        st.markdown("### 🎯 ACS MANEUVER ANALYSIS")
        
        df['In_Maneuver'] = df['Turn_Rate'] > 1.8
        df['Maneuver_ID'] = (df['In_Maneuver'] != df['In_Maneuver'].shift()).cumsum()
        
        found_mnvrs = 0
        for mid, mdata in df[df['In_Maneuver']].groupby('Maneuver_ID'):
            total_turn = mdata['Track_Delta'].sum()
            duration = mdata['Dt'].sum()
            
            if duration > 15 and total_turn > 150:
                found_mnvrs += 1
                entry_alt = mdata['Alt_Smooth'].iloc[0]
                max_dev = max(mdata['Alt_Smooth'].max() - entry_alt, entry_alt - mdata['Alt_Smooth'].min())
                
                label = "360° STEEP TURN" if total_turn > 320 else "COURSE REVERSAL"
                color = "#00FF41" if max_dev <= 50 else ("#FF9F1C" if max_dev <= 100 else "#FF0000")
                status = "CPL GRADE" if max_dev <= 50 else ("PPL PASS" if max_dev <= 100 else "ACS BUST")

                with st.expander(f"MNVR {found_mnvrs} | {label} | DEV: {int(max_dev)}FT"):
                    st.markdown(f"**STATUS:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
                    st.write(f"`ENTRY ALT: {int(entry_alt)} FT | DURATION: {int(duration)}s | TURN: {int(total_turn)}°`")
                    if label == "360° STEEP TURN":
                        wind = (mdata['GS'].max() - mdata['GS'].min()) / 2
                        st.write(f"`ESTIMATED WINDS ALOFT: {int(wind)} KTS`")

        st.markdown("### 🗺️ SPATIAL TELEMETRY")
        t1, t2, t3 = st.tabs(["3D AIRWAY CORRIDOR", "TACTICAL MAP", "ALTITUDE PROFILE"])
        
        with t1:
            # 1. THE AIRBORNE FILTER: Strip out the taxiway spaghetti
            airborne_df = df[df['GS'] > 35]
            
            # 2. THE RIBBON UPGRADE: True 3D line instead of scatter dots
            fig_3d = go.Figure(data=go.Scatter3d(
                x=airborne_df['Lon'],
                y=airborne_df['Lat'],
                z=airborne_df['Alt_Smooth'],
                mode='lines',
                line=dict(
                    color=airborne_df['GS'],
                    colorscale='Inferno',
                    width=6,
                    colorbar=dict(title="KTS")
                ),
                text=[f"ALT: {alt:.0f} FT<br>GS: {gs:.0f} KTS" for alt, gs in zip(airborne_df['Alt_Smooth'], airborne_df['GS'])],
                hoverinfo="text"
            ))
            
            # 3. THE ASPECT RATIO FIX: Force the box to be wide and flat
            fig_3d.update_layout(
                title="3D TRAJECTORY (AIRBORNE ONLY)",
                template="plotly_dark", 
                height=700, 
                margin=dict(l=0,r=0,b=0,t=40),
                scene=dict(
                    xaxis_title="LONGITUDE",
                    yaxis_title="LATITUDE",
                    zaxis_title="ALTITUDE (FT)",
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.4) # Flattens the Z-axis distortion
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with t2:
            fig_map = px.scatter_mapbox(
                df, lat="Lat", lon="Lon", color="VSI",
                color_continuous_scale="RdBu_r", range_color=[-1000, 1000],
                zoom=11, height=600
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_map, use_container_width=True)

        with t3:
            fig_alt = go.Figure()
            fig_alt.add_trace(go.Scatter(x=df['Time'], y=df['Alt_Raw'], name="RAW SENSOR", line=dict(color="rgba(255,0,0,0.3)", width=1)))
            fig_alt.add_trace(go.Scatter(x=df['Time'], y=df['Alt_Smooth'], name="MFD SMOOTHED", line=dict(color="#00FF41", width=3)))
            fig_alt.update_layout(template="plotly_dark", xaxis_title="TIME (UTC)", yaxis_title="ALTITUDE (FT MSL)")
            st.plotly_chart(fig_alt, use_container_width=True)
