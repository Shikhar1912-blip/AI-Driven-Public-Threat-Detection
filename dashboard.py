import streamlit as st
import pandas as pd
import time
import os

st.title("AI-Powered Threat Detection Dashboard")
st.markdown("Real-time monitoring of suspicious objects in public spaces.")

# Initialize session state for last checked time
if 'last_checked' not in st.session_state:
    st.session_state.last_checked = 0

# Placeholder for video feed (Streamlit doesn't support live video, so we use detections)
st.subheader("Live Detection Feed")
video_placeholder = st.empty()

# Alert section
st.subheader("Suspicious Activity Alerts")
alert_placeholder = st.empty()

# Detection log
st.subheader("Detection Log")
log_placeholder = st.empty()

def load_detections():
    if os.path.exists('detections.csv'):
        return pd.read_csv('detections.csv')
    return pd.DataFrame(columns=['timestamp', 'object', 'confidence', 'x', 'y', 'w', 'h'])

def check_alerts(df):
    suspicious_objects = ['backpack', 'suitcase', 'handbag']
    recent_detections = df[df['timestamp'] > time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() - 10))]
    alerts = []
    for obj in suspicious_objects:
        obj_detections = recent_detections[recent_detections['object'] == obj]
        if not obj_detections.empty:
            alerts.append(f"Suspicious {obj} detected at {obj_detections['timestamp'].iloc[-1]}")
    return alerts

# Update dashboard every second
while True:
    df = load_detections()
    
    # Update detection log
    if not df.empty:
        log_placeholder.dataframe(df.sort_values(by='timestamp', ascending=False).head(10))
    
    # Check for alerts
    alerts = check_alerts(df)
    if alerts:
        alert_placeholder.warning("\n".join(alerts))
    else:
        alert_placeholder.success("No suspicious activity detected.")
    
    # Simulate video feed update (replace with actual video frame if possible)
    video_placeholder.image('sample_frame.jpg' if os.path.exists('sample_frame.jpg') else [], caption="Live Feed (Placeholder)")
    
    time.sleep(1)
    st.experimental_rerun()