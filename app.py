import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import math
import requests

# --- UI Configuration (Industrial Mode) ---
st.set_page_config(page_title="Turbulence Gains | Pro Flight Analyzer", layout="wide", page_icon="✈️", initial_sidebar_state="collapsed")

# --- Custom CSS Injection for Glass Cockpit Aesthetic ---
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers & Accent Colors */
    h1, h2, h3 {
        color: #FF9F1C !important; 
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 1px solid #333333;
        padding-bottom: 10px;
    }
    
    /* Metric Cards - Industrial Box Look */
    div[data-testid="metric-container"] {
        background-color: #1A1C23;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 2px;
        border-left: 4px solid #FF9F1C;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    div[data-testid="metric-container"] label {
        color: #888888 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="metric-container"] div {
        color: #00FF41 !important; /* Radar Green */
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
    }
    
    /* Expanders & Information Boxes */
    .streamlit-expanderHeader {
        background-color: #1A1C23 !important;
        color: #FF9F1C !important;
        font-family: 'Courier New', Courier, monospace;
        border: 1px solid #333333;
    }
    div[data-testid="stInfo"] {
        background-color: rgba(0, 255, 65, 0.1);
        border-left: 4px solid #00FF41;
        color: #E0E0E0;
    }
    div[data-testid="stWarning"] {
        background-color: rgba(255, 159, 28, 0.1);
        border-left: 4px solid #FF9F1C;
        color: #E0E0E0;
    }
    div[data-testid="stError"] {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #FF0000;
        color: #E0E0E0;
    }
    
    /* Uploader */
    .stFileUploader {
        border: 1px dashed #FF9F1C;
        background-color: #1A1C23;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Math Helper Functions ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3440.065 # NM
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

# --- AWC Weather API Engine ---
@st.cache_data
def fetch_weather(lat, lon):
    try:
        url = f"https://aviationweather.gov/api/data/metar?lat={lat}&lon={lon}&distance=25&format=json&hours=48"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                station = data[0]
                return {
                    "raw": station.get("rawOb", "No raw data available."),
                    "id": station.get("icaoId", "Unknown"),
                    "temp": station.get("temp", "N/A"),
                    "wind_dir": station.get("wdir", "VRB"),
                    "wind_spd": station.get("wspd", 0)
                }
    except Exception as e:
        return None
    return None

# --- Core AI Processing Engine ---
@st.cache_data
def process_flight_data(file_content):
    soup = BeautifulSoup(file_content, 'xml')
    times, coords = soup.find_all('when'), soup.find_all('gx:coord')

    flight_data = [{'Timestamp': t.text, 'Longitude': float(c.text.split()[0]), 
                    'Latitude': float(c.text.split()[1]), 'Raw_Altitude': float(c.text.split()[2]) * 3.28084} 
                   for t, c in zip(times, coords) if len(c.text.split()) == 3]

    df = pd.DataFrame(flight_data)
    if df.empty: return df

    df['Timestamp'] = df['Timestamp'].str.replace('Z', '', regex=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Delta_Sec'] = df['Timestamp'].diff().dt.total_seconds().fillna(1)
    
    df['Smoothed_Altitude'] = df['Raw_Altitude'].rolling(window=7, center=True, min_periods=1).mean()
    df['VSI (FPM)'] = (df['Smoothed_Altitude'].diff() / (df['Time_Delta_Sec'] / 60.0)).fillna(0).rolling(3).mean()
    
    distances, courses = [0], [0]
    for i in range(1, len(df)):
        distances.append(haversine_distance(df.iloc[i-1]['Latitude'], df.iloc[i-1]['Longitude'], df.iloc[i]['Latitude'], df.iloc[i]['Longitude']))
        courses.append(calculate_bearing(df.iloc[i-1]['Latitude'], df.iloc[i-1]['Longitude'], df.iloc[i]['Latitude'], df.iloc[i]['Longitude']))
        
    df['Distance_NM'] = distances
    df['Course'] = courses
    df['Groundspeed (Kts)'] = (df['Distance_NM'] / (df['Time_Delta_Sec'] / 3600.0)).fillna(0).rolling(5).mean()
    
    df['Course_Change'] = df['Course'].diff().abs()
    df['Course_Change'] = df['Course_Change'].apply(lambda x: 360 - x if x > 180 else x).fillna(0)
    df['Smoothed_Turn_Rate'] = (df['Course_Change'] / df['Time_Delta_Sec']).rolling(window=3, center=True, min_periods=1).mean()

    return df

# --- Application UI Layout ---
st.title("✈️ T.G. TACTICAL DEBRIEF SYSTEM")
st.markdown("`SYSTEM ARMED. AWAITING KML TELEMETRY INGESTION.`")

uploaded_file = st.file_uploader("", type=['kml'])

if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode('utf-8')
    with st.spinner('PROCESSING FLIGHT DYNAMICS...'):
        df = process_flight_data(file_content)
        
    if not df.empty:
        start_lat = df['Latitude'].iloc[0]
        start_lon = df['Longitude'].iloc[0]
        weather_data = fetch_weather(start_lat, start_lon)
        
        st.markdown("### 📊 FLIGHT TELEMETRY")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PEAK ALT (MSL)", f"{int(df['Smoothed_Altitude'].max())} FT")
        col2.metric("MAX GS", f"{int(df['Groundspeed (Kts)'].max())} KTS")
        col3.metric("MAX SINK", f"{int(df['VSI (FPM)'].min())} FPM")
        col4.metric("TIME ON TARGET", f"{int(df['Time_Delta_Sec'].sum() / 60)} MIN")
        
        st.markdown("<br>", unsafe_allow_html=True)

        if weather_data:
            st.info(f"📍 **AWC SURFACE METAR ({weather_data['id']}):**\n\n`{weather_data['raw']}`")
        
        st.markdown("### 🎯 ACS MANEUVER GRADING")
        
        df['Is_Turning'] = df['Smoothed_Turn_Rate'] > 1.8 
        df['Maneuver_ID'] = (df['Is_Turning'] != df['Is_Turning'].shift()).cumsum()
        
        turns = df[df['Is_Turning']]
        graded_maneuvers = 0
        
        for maneuver_id, maneuver_data in turns.groupby('Maneuver_ID'):
            duration = maneuver_data['Time_Delta_Sec'].sum()
            total_heading_change = maneuver_data['Course_Change'].sum()
            
            if duration > 15 and total_heading_change > 150:
                graded_maneuvers += 1
                start_alt = maneuver_data['Smoothed_Altitude'].iloc[0]
                max_alt = maneuver_data['Smoothed_Altitude'].max()
                min_alt = maneuver_data['Smoothed_Altitude'].min()
                
                alt_loss, alt_gain = start_alt - min_alt, max_alt - start_alt
                max_deviation = max(alt_loss, alt_gain)
                
                if total_heading_change >= 320:
                    maneuver_type = "360° STEEP TURN"
                    max_gs = maneuver_data['Groundspeed (Kts)'].max()
                    min_gs = maneuver_data['Groundspeed (Kts)'].min()
                    estimated_wind = (max_gs - min_gs) / 2
                    wind_text = f"💨 DERIVED WINDS ALOFT: ~{int(estimated_wind)} KTS"
                elif 150 <= total_heading_change < 320:
                    maneuver_type = "COURSE REVERSAL"
                    wind_text = ""
                
                with st.expander(f"MNVR {graded_maneuvers} | {maneuver_type} | {int(total_heading_change)}°"):
                    st.markdown(f"`DURATION: {int(duration)}s | ENTRY ALT: {int(start_alt)}FT`")
                    
                    if wind_text:
                        st.info(f"`{wind_text}`")

                    if max_deviation <= 50:
                        st.success(f"**CPL STD:** ALT DEVIATION +{int(alt_gain)}FT / -{int(alt_loss)}FT.")
                    elif max_deviation <= 100:
                        st.warning(f"**PPL STD:** ALT DEVIATION +{int(alt_gain)}FT / -{int(alt_loss)}FT.")
                    else:
                        st.error(f"**ACS BUST:** ALT DEVIATION +{int(alt_gain)}FT / -{int(alt_loss)}FT. FIX: MAINTAIN OUTSIDE VISUAL REFERENCE.")
                        
        if graded_maneuvers == 0:
            st.write("`NO ACS MANEUVERS DETECTED IN LOG.`")
            
        st.markdown("### 🗺️ TACTICAL PLOTS")
        tab1, tab2, tab3 = st.tabs(["3D TRAJECTORY", "2D TRACK", "ALTITUDE PROFILE"])
        
        with tab1:
            
            fig_3d = px.line_3d(
                df, x="Longitude", y="Latitude", z="Smoothed_Altitude", 
                color="Groundspeed (Kts)", color_continuous_scale=px.colors.sequential.Plotly3,
                hover_data=["VSI (FPM)"]
            )
            fig_3d.update_traces(line=dict(width=5))
            fig_3d.update_layout(template="plotly_dark", scene=dict(zaxis=dict(title="Altitude (MSL)")), height=600, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_3d, use_container_width=True)

        with tab2:
            fig_map = px.scatter_mapbox(
                df, lat="Latitude", lon="Longitude", hover_name="Timestamp", 
                hover_data=["Smoothed_Altitude", "Groundspeed (Kts)", "VSI (FPM)"],
                color="VSI (FPM)", color_continuous_scale=px.colors.diverging.RdYlBu,
                zoom=10, height=600
            )
            # Switch to a dark map theme
            fig_map.update_layout(mapbox_style="carto-darkmatter", template="plotly_dark", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
        with tab3:
            fig_alt = go.Figure()
            fig_alt.add_trace(go.Scatter(x=df['Timestamp'], y=df['Raw_Altitude'], mode='lines', name='RAW SENSOR', line=dict(color='rgba(255, 0, 0, 0.4)', width=1)))
            fig_alt.add_trace(go.Scatter(x=df['Timestamp'], y=df['Smoothed_Altitude'], mode='lines', name='SMOOTHED', line=dict(color='#00FF41', width=3)))
            fig_alt.update_layout(template="plotly_dark", xaxis_title="TIME (Z)", yaxis_title="ALTITUDE (MSL)", hovermode="x unified")
            st.plotly_chart(fig_alt, use_container_width=True)
