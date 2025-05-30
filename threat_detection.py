import requests  
import cv2
import numpy as np
import pandas as pd
import time
import os
import pygame
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from ultralytics import YOLO
import streamlit as st
import threading
import webbrowser
import socketserver

# Twilio configuration
TWILIO_ACCOUNT_SID = "AC0b22c74ce9bd63afb38f01c9fa3f3ba3"
TWILIO_AUTH_TOKEN = "1dae08960fbb654a235d4d21186c3eb6"
TWILIO_PHONE_NUMBER = "+17853475341"
DESTINATION_PHONE_NUMBER = "+919319900972"

# Validate Twilio configuration
def validate_twilio_config():
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, DESTINATION_PHONE_NUMBER]):
        print("Error: Missing Twilio credentials.")
        return False
    if not TWILIO_PHONE_NUMBER.startswith('+') or not DESTINATION_PHONE_NUMBER.startswith('+'):
        print("Error: Phone numbers must include country code.")
        return False
    return True

# Initialize Twilio client
twilio_client = None
if validate_twilio_config():
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("DEBUG: Twilio client initialized")
    except Exception as e:
        print(f"Error initializing Twilio: {e}")

# Initialize pygame mixer for siren
try:
    pygame.mixer.init()
    siren_sound = pygame.mixer.Sound('Siren.mp3')
    siren_sound.set_volume(0.8)
    print("DEBUG: Siren sound loaded successfully")
except FileNotFoundError:
    print("Error: siren.wav not found. Please place a siren sound file in the project directory.")
    siren_sound = None
except Exception as e:
    print(f"Error initializing pygame mixer: {e}")
    siren_sound = None

# 📍 Get location for SMS
def get_location_link():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        latlong = res['loc']
        return f"https://maps.google.com/?q={latlong}"
    except Exception as e:
        print(f"Error getting location: {e}")
        return "Location unavailable"

# 🚨 Send SMS with location
def send_sms_alert(object_name, timestamp):
    if twilio_client is None:
        print("Twilio client not initialized")
        return
    location_link = get_location_link()
    message_body = (
        f"🚨 ALERT: Suspicious {object_name} detected at {timestamp}!\n"
        f"📍 Location: {location_link}"
    )
    try:
        message = twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=DESTINATION_PHONE_NUMBER
        )
        print(f"DEBUG: SMS sent: {message.sid}")
    except TwilioRestException as e:
        print(f"Twilio error {e.code}: {e}")
        if e.code == 21614:
            print("Hint: Verify DESTINATION_PHONE_NUMBER in Twilio Console (https://www.twilio.com/console/phone-numbers/verified).")
        elif e.code == 30007:
            print("Hint: SMS may be filtered. Text 'START' to your Twilio number from the recipient phone.")
        elif e.code == 21610:
            print("Hint: Recipient has blocked SMS. Text 'START' to unblock.")
    except Exception as e:
        print(f"Unexpected error sending SMS: {e}")

# Load YOLOv11 model
try:
    model = YOLO("yolo11n.pt")
    print("DEBUG: YOLOv11 model loaded")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit(1)

# Detection log
log_file = 'detections.csv'
if not os.path.exists(log_file):
    pd.DataFrame(columns=['timestamp', 'object', 'confidence', 'x', 'y', 'w', 'h']).to_csv(log_file, index=False)

# Tracking variables
tracked_objects = {}
alerted_objects = {}
paused = False
last_frame = None

# 🌐 Streamlit dashboard
def run_streamlit():
    st.title("AI-Powered Threat Detection Dashboard")
    st.markdown("Real-time monitoring of suspicious objects.")

    st.subheader("Live Feed")
    video_placeholder = st.empty()

    st.subheader("Alerts")
    alert_placeholder = st.empty()

    st.subheader("Detection Log")
    log_placeholder = st.empty()

    def load_detections():
        if os.path.exists('detections.csv'):
            return pd.read_csv('detections.csv')
        return pd.DataFrame(columns=['timestamp', 'object', 'confidence', 'x', 'y', 'w', 'h'])

    def check_alerts(df):
        suspicious_objects = ['backpack', 'suitcase', 'handbag', 'knife']
        recent = df[df['timestamp'] > time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() - 10))]
        alerts = []
        for obj in suspicious_objects:
            found = recent[recent['object'] == obj]
            if not found.empty:
                alerts.append(f"{obj} detected at {found['timestamp'].iloc[-1]}")
        return alerts

    while True:
        df = load_detections()
        if not df.empty:
            log_placeholder.dataframe(df.sort_values(by='timestamp', ascending=False).head(10))
        alerts = check_alerts(df)
        if alerts:
            alert_placeholder.warning("\n".join(alerts))
        else:
            alert_placeholder.success("No suspicious activity")
        video_placeholder.image('sample_frame.jpg' if os.path.exists('sample_frame.jpg') else [], caption="Live Feed")
        time.sleep(1)

# Start Streamlit in thread
def start_streamlit():
    port = 8501
    try:
        with socketserver.TCPServer(("", port), None) as s:
            pass
    except OSError:
        port = 8502
    print(f"Starting Streamlit dashboard at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    from streamlit.web import bootstrap
    bootstrap.run("threat_detection.py", "", [], flag_options={"server.port": port, "server.headless": True})

# Detection utilities
def is_suspicious(name, duration):
    return name in ['backpack', 'suitcase', 'handbag'] and duration > 10

def is_same_object(x1, y1, x2, y2, threshold=20):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold

# Start Streamlit
streamlit_thread = threading.Thread(target=start_streamlit, daemon=True)
streamlit_thread.start()

# 🚨 Main detection loop
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame")
            break
        last_frame = frame.copy()
        cv2.imwrite('sample_frame.jpg', frame)
    else:
        frame = last_frame.copy()

    if not paused:
        try:
            results = model(frame, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()
        except Exception as e:
            print(f"Detection error: {e}")
            continue

        current_time = time.time()
        new_tracked = {}

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            obj_name = model.names[int(cls)]
            obj_id = f"{obj_name}_{int(x1)}_{int(y1)}"
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            log_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'object': obj_name,
                'confidence': conf,
                'x': x1,
                'y': y1,
                'w': x2 - x1,
                'h': y2 - y1
            }
            try:
                pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error writing to CSV: {e}")

            if obj_name in ['backpack', 'suitcase', 'handbag']:
                matched = None
                for oid, (start, ox, oy) in tracked_objects.items():
                    if is_same_object(cx, cy, ox, oy):
                        matched = oid
                        break
                if matched:
                    new_tracked[matched] = [tracked_objects[matched][0], cx, cy]
                else:
                    new_tracked[obj_id] = [current_time, cx, cy]
                    print(f"Tracking {obj_name} at ({cx:.1f}, {cy:.1f})")

        tracked_objects = new_tracked

    for obj_id, (start_time, x, y) in tracked_objects.items():
        duration = time.time() - start_time
        name = obj_id.split('_')[0]
        if is_suspicious(name, duration):
            if obj_id not in alerted_objects or (time.time() - alerted_objects[obj_id] > 30):
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"🚨 ALERT: Suspicious {name} detected at {timestamp}")
                if siren_sound:
                    try:
                        siren_sound.play()
                        print("DEBUG: Siren played successfully")
                    except Exception as e:
                        print(f"Error playing siren: {e}")
                send_sms_alert(name, timestamp)
                alerted_objects[obj_id] = time.time()
                paused = True
                print("Paused. Press 'r' to resume or 'q' to quit")

    if paused:
        cv2.putText(frame, f"SUSPICIOUS {name.upper()} DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'r' to resume", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    results[0].plot()
    cv2.imshow("YOLOv11 Detection", frame if paused else results[0].plot())

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and paused:
        paused = False
        print("Camera resumed")

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("DEBUG: Program terminated")
