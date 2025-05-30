# AI-Driven-Public-Threat-Detection
A real time threat detection which we have put out for the purpose of public safety by means of the identification of suspicious items (for instance unattended suitcases) in video streams via YOLOv11. The system sets off audio sirens, sends out SMS’s which include location, has the camera stop and also puts up a web interface which has live feed and detection logs. We put this together for a hackathon which also ends up being a display of advanced computer vision.
## Problem statement
Public spaces like airports and train stations face security risks from unattended objects, such as suitcases or backpacks, which may pose threats. Manual monitoring is labor-intensive and prone to errors. The challenge is to develop an automated system that:
- Detects suspicious objects in reoal-time video feeds.
- Alerts security personnel through multiple channels (audio, SMS, visual).
- Provides a user-friendly interface for monitoring and response.
- Ensures rapid, reliable detection with minimal false positives.
## Approach and Solution
### Approach
- Object Detection: Use YOLOv11, a state-of-the-art (SOTA) model, to detect objects like suitcases, backpacks, and handbags in webcam feeds.
- Suspicious Behavior: Flag objects held steady for >10 seconds as potential threats, indicating possible abandonment.
- Multi-Modal Alerts: Trigger audible sirens, pause the camera for human review, send SMS with location, and update a web dashboard.
- Frontend: Develop a Streamlit-based web interface to display live feed, alerts, and detection logs, with a resume button for paused states.
- Integration: Combine computer vision, IoT (SMS, audio), and web technologies for a seamless system.
### Solution
The system processes webcam video using YOLOv11 to identify suspicious objects. When an object is detected for >10 seconds:
- A siren (siren.wav) plays to alert nearby personnel.
- The camera pauses, displaying “SUSPICIOUS OBJECT DETECTED!”.
- An SMS with a Google Maps location link is sent via Twilio.
- A Streamlit frontend shows the live feed, real-time alerts, and a log table.
- Users can resume detection via a button or ‘r’ key. Detections are logged to detections.csv for analysis. The solution is lightweight, scalable, and hackathon-ready.
## Features
- Real-Time Detection: Identifies suspicious objects (backpacks, suitcases, handbags) using YOLOv11.
- Web Interface: Streamlit frontend with live video feed, bounding boxes, alerts, and a detection log table.
- Audible Alerts: Plays siren.wav when threats are detected.
- SMS Notifications: Sends alerts with location links via Twilio.
- Camera Pause: Halts video feed on detection, resumable via button or 'r' key or quit it by 'q' key.
- Logging: Saves detection details (timestamp, object, confidence, coordinates) to detections.csv.
## Tech Stack
- Backend: Python, Streamlit, OpenCV, YOLOv11 (Ultralytics).
- Frontend: HTML, Bootstrap, Tailwind CSS, JavaScript (AJAX).
- Notifications: Twilio (SMS), ipinfo.io (location).
- Audio: Pygame (siren playback).
- Data: Pandas (CSV logging).
## Scrrenshots
![img alt](https://github.com/Shikhar1912-blip/AI-Driven-Public-Threat-Detection/blob/ba65e42f7190aac69a54516efbdc5271dce3c9b8/IMG%201.png)
![img alt](https://github.com/Shikhar1912-blip/AI-Driven-Public-Threat-Detection/blob/ba65e42f7190aac69a54516efbdc5271dce3c9b8/IMG%202.jpg)
## Run instructions
- Firstly start by creating a virtual enviornment by running "python -m venv venv" and then activate it by "venv\Scripts\activate"
- Then install the external python libraries by using the pip command by running "pip install torch opencv-python streamlit numpy pandas twilio pygame ultralytics threading"
- To open the Streamlit interface, run in the command prompt "streamlit run dashboard.py"
- To start the project run "python threat_detection.pya"
- Hold any suspicous object like 'suitcase', 'backpack or 'handbag' in front of the webcam for the threat detection.
- Additionally for getting messages via sms, create a Twilio account and replace the SID, Token, Twilio number and you own phone number in the code itself.
