Garbage Detection System 
An AI-powered web-based application for detecting garbage and waste materials from images or live camera feeds.  
This project uses computer vision and geolocation to identify pollution zones and visualize them on interactive maps.

Features
- Real-time garbage detection using **YOLOv8** (Ultralytics).
- Automatic **GPS extraction** from images to locate garbage spots.
- **Weather data integration** using the OpenWeather API.
- Centralized data storage using **MongoDB Atlas**.
- **Interactive pollution maps** created with Folium and Geopy.
- Dashboard for visualizing pollution zones and analysis results.

Technologies Used
**Backend:** Python (Flask)  
**AI Model:** YOLOv8 (Ultralytics)  
**Libraries:** OpenCV, cvzone, Geopy, Folium, Matplotlib  
**Database:** MongoDB Atlas  
**Frontend:** HTML, CSS, JavaScript  


Garbage-Detection-System/
│
├── GarbageDetector/         # Main backend and detection scripts
├── templates/               # HTML templates for Flask
├── static/                  # CSS, JS, and static files
├── uploads/                 # Uploaded image storage
├── results/                 # Detection outputs and results
├── maps/                    # Generated pollution maps
├── requirements.txt         # Required dependencies
└── app1.py                  # Flask main entry point
