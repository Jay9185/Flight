import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import math

# --- UI Configuration ---
st.set_page_config(page_title="Turbulence Gains | Flight Analyzer", layout="wide", page_icon="✈️")

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

# --- Core Engine ---
@st.cache_data
def process_flight_data(file_content):
    soup = BeautifulSoup(file_content, 'xml')
    times, coords = soup.find_all('when'), soup.find_all('gx:coord')

    flight_data = [{'Timestamp': t.text, 'Longitude': float(c.text.split()[0]), 
                    'Latitude': float(c.text.split()[1]), 'Raw_Altitude': float(c.text.split()[2]) * 3.28084} 
                   for t, c in zip(times, coords) if len(c.text.split()) == 3]

    df = pd.DataFrame(flight_data)
    if df.empty: return df

    # ForeFlight PMZ Fix & Time Deltas
    df['Timestamp'] = df['Timestamp'].str.replace('Z', '', regex=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Delta_Sec'] = df['Timestamp'].diff().dt.total_seconds().fillna(1)
    
    # Cellular GPS Smoothing 
    df['Smoothed_Altitude'] = df['Raw_Altitude'].rolling(window=5, center=True, min_periods=1).mean()
    df['VSI (FPM)'] = (df['Smoothed_Altitude'].diff() / (df['Time_Delta_Sec'] / 60.0)).fillna(0).rolling(3).mean()
    
    # Track & Speed Math
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

# --- Application Layout ---
st.title("✈️ Advanced Flight Debrief System")
st.markdown("Upload your ForeFlight KML track log. The AI will automatically isolate maneuvers and grade them against standard ACS tolerances.")

uploaded_file = st.file_uploader("Drop KML File Here", type=['kml'])

if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode('utf-8')
    with st.spinner('Applying hardware smoothing and maneuver recognition...'):
        df = process_flight_data(file_content)
        
    if not df.empty:
        # High-Level Metrics
        st.markdown("### 📊 Flight Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Altitude (MSL)", f"{int(df['Smoothed_Altitude'].max())} ft")
        col2.metric("Max Groundspeed", f"{int(df['Groundspeed (Kts)'].max())} kts")
        col3.metric("Max Sink Rate", f"{int(df['VSI (FPM)'].min())} FPM")
        col4.metric("Duration", f"{int(df['Time_Delta_Sec'].sum() / 60)} mins")
        
        st.markdown("---")
        
        # Smart Maneuver Debrief
        st.markdown("### 🎯 ACS Maneuver Grading (Steep Turns)")
        
        df['Is_Turning'] = df['Smoothed_Turn_Rate'] > 2.0
        df['Maneuver_ID'] = (df['Is_Turning'] != df['Is_Turning'].shift()).cumsum()
        
        turns = df[df['Is_Turning']]
        graded_maneuvers = 0
        
        for maneuver_id, maneuver_data in turns.groupby('Maneuver_ID'):
            duration = maneuver_data['Time_Delta_Sec'].sum()
            total_heading_change = maneuver_data['Course_Change'].sum()
            
            # THE NEW SMART FILTER: Must be > 15s AND > 150 degrees of turn
            if duration > 15 and total_heading_change > 150:
                graded_maneuvers += 1
                start_alt = maneuver_data['Smoothed_Altitude'].iloc[0]
                max_alt = maneuver_data['Smoothed_Altitude'].max()
                min_alt = maneuver_data['Smoothed_Altitude'].min()
                
                alt_loss, alt_gain = start_alt - min_alt, max_alt - start_alt
                max_deviation = max(alt_loss, alt_gain)
                
                with st.expander(f"Maneuver {graded_maneuvers} | {int(total_heading_change)}° Turn | Duration: {int(duration)}s"):
                    # Tiered Grading System
                    if max_deviation <= 50:
                        st.success(f"**🌟 EXCELLENT:** Altitude held within {int(max_deviation)}ft. Checkride ready.")
                    elif max_deviation <= 100:
                        st.info(f"**✅ PASS (ACS Standard):** Altitude deviated by {int(max_deviation)}ft. Within 100ft limits.")
                    else:
                        st.error(f"**❌ BUST:** Deviated by +{int(alt_gain)}ft / -{int(alt_loss)}ft.")
                        st.write("*Aero Fix: Anticipate vertical lift loss. Add slight back pressure/power passing 30 deg bank.*")
                        
        if graded_maneuvers == 0:
            st.write("No steep turns or course reversals > 150° detected. Pattern work filtered out.")
            
        st.markdown("---")
        
        # Interactive Telemetry Maps
        st.markdown("### 🗺️ Visual Telemetry")
        tab1, tab2 = st.tabs(["Ground Track (Speed & VSI)", "Altitude Smoothing Profile"])
        
        with tab1:
            fig_map = px.scatter_mapbox(
                df, lat="Latitude", lon="Longitude", hover_name="Timestamp", 
                hover_data=["Smoothed_Altitude", "Groundspeed (Kts)", "VSI (FPM)"],
                color="Groundspeed (Kts)", color_continuous_scale=px.colors.sequential.Inferno,
                zoom=10, height=600
            )
            fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
        with tab2:
            fig_alt = go.Figure()
            fig_alt.add_trace(go.Scatter(x=df['Timestamp'], y=df['Raw_Altitude'], mode='lines', name='Raw GPS (Jitter)', line=dict(color='rgba(255, 0, 0, 0.3)', width=1)))
            fig_alt.add_trace(go.Scatter(x=df['Timestamp'], y=df['Smoothed_Altitude'], mode='lines', name='Smoothed Actual', line=dict(color='blue', width=3)))
            fig_alt.update_layout(xaxis_title="Time", yaxis_title="Altitude (Feet MSL)", hovermode="x unified")
            st.plotly_chart(fig_alt, use_container_width=True)
