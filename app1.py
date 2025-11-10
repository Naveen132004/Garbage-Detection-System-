from flask import Flask, render_template, request, jsonify, send_file, make_response
import os
import uuid
import math
import base64
from datetime import datetime, timedelta

import cv2
import cvzone
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from PIL import Image, ExifTags
import folium
from folium import plugins
from folium.plugins import HeatMap
from branca.element import Template, MacroElement
import requests
from geopy.geocoders import Nominatim

from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ----------------------------
# Flask Configuration
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('maps', exist_ok=True)


# ----------------------------
# MongoDB Atlas Connection
# ----------------------------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://navinkrsingh7856:viper7856@cluster0.41d6i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
DB_NAME = os.getenv("MONGO_DB", "garbage_db")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
detections_collection = db["detections"]
zones_collection = db["pollution_zones"]


# ----------------------------
# Load YOLO model
# ----------------------------
try:
    yolo_model = YOLO("Weights/best.pt")
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    print("Please ensure 'Weights/best.pt' exists.")
    yolo_model = None


# ----------------------------
# Labels & Weights
# ----------------------------
class_labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']
pollution_weights = {
    '0': 1,
    'c': 1,
    'garbage': 3,
    'garbage_bag': 5,
    'sampah-detection': 2,
    'trash': 2
}


# ----------------------------
# Geocoder & Weather
# ----------------------------
try:
    geolocator = Nominatim(user_agent="garbage_detector_app_v1.0 (contact: example@example.com)")
    print("Geocoder initialized successfully")
except Exception as e:
    print(f"Warning: Geocoder initialization failed: {e}")
    geolocator = None


def get_location_name(lat, lng):
    if not geolocator:
        return "Unknown Location"
    try:
        location = geolocator.reverse(f"{lat}, {lng}", timeout=5)
        return location.address if location else "Unknown Location"
    except Exception as e:
        print(f"Geocoding error: {e}")
        return f"Location ({lat:.4f}, {lng:.4f})"


def get_weather_data(lat, lng):
    """Get weather data for location (placeholder - add your API key)"""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY", "YOUR_WEATHER_API_KEY")
        if api_key == "YOUR_WEATHER_API_KEY":
            return {
                "main": {"temp": 25, "humidity": 60},
                "weather": [{"description": "clear sky"}],
                "wind": {"speed": 5}
            }
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


# ----------------------------
# EXIF GPS Helpers
# ----------------------------
def extract_gps_info(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name == 'GPSInfo':
                        gps_info = value
                        if 2 in gps_info and 4 in gps_info:
                            lat = convert_to_degrees(gps_info[2])
                            if gps_info.get(1) == 'S':
                                lat = -lat
                            lng = convert_to_degrees(gps_info[4])
                            if gps_info.get(3) == 'W':
                                lng = -lng
                            return lat, lng
    except Exception as e:
        print(f"GPS extraction error: {e}")
    return None, None


def convert_to_degrees(value):
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            d, m, s = value[:3]
            # some EXIF give rationals; cast to float safely
            d = float(d[0]) / float(d[1]) if isinstance(d, tuple) else float(d)
            m = float(m[0]) / float(m[1]) if isinstance(m, tuple) else float(m)
            s = float(s[0]) / float(s[1]) if isinstance(s, tuple) else float(s)
            return d + m/60.0 + s/3600.0
        return float(value)
    except Exception:
        return 0.0


# ----------------------------
# Scoring
# ----------------------------
def calculate_pollution_score(detections):
    if not detections:
        return 0
    total_score = 0
    for detection in detections:
        class_name = detection.get('class', 'garbage')
        confidence = detection.get('confidence', 0)
        weight = pollution_weights.get(class_name, 2)
        total_score += weight * confidence
    return min(100, round(total_score * 10, 2))


# ----------------------------
# Detection Pipeline
# ----------------------------
def process_image_with_location(image_path, output_path, lat=None, lng=None):
    if yolo_model is None:
        return False, "YOLO model not loaded. Please check if 'Weights/best.pt' exists."
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read image file"

        if lat is None or lng is None:
            img_lat, img_lng = extract_gps_info(image_path)
            lat = lat or img_lat
            lng = lng or img_lng

        results = yolo_model(img)

        detection_count = 0
        detections = []

        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is not None:
                for box in boxes:
                    # ensure Python ints/floats (tensors -> .item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                    conf = math.ceil(conf * 100) / 100
                    cls = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])

                    if conf > 0.3 and 0 <= cls < len(class_labels):
                        detection_count += 1
                        detections.append({
                            'class': class_labels[cls],
                            'confidence': conf,
                            'bbox': [x1, y1, w, h]
                        })
                        color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
                        cvzone.cornerRect(img, (x1, y1, w, h), t=2, colorR=color)
                        cvzone.putTextRect(
                            img,
                            f"{class_labels[cls]} {conf:.2f}",
                            (x1, y1 - 10),
                            scale=0.8,
                            thickness=1,
                            colorR=(255, 0, 0),
                            colorT=(255, 255, 255)
                        )

        if lat is not None and lng is not None:
            cv2.putText(img, f"Lat: {float(lat):.6f}, Lng: {float(lng):.6f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        pollution_score = calculate_pollution_score(detections)
        score_color = (0, 255, 0) if pollution_score < 30 else (0, 255, 255) if pollution_score < 70 else (0, 0, 255)
        cv2.putText(img, f"Pollution Score: {pollution_score}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

        cv2.imwrite(output_path, img)

        if lat is not None and lng is not None:
            store_detection_data(lat, lng, output_path, detection_count, pollution_score, detections)

        return True, {
            'detections': detections,
            'count': detection_count,
            'pollution_score': pollution_score,
            'coordinates': {'lat': float(lat), 'lng': float(lng)} if lat is not None and lng is not None else None,
            'location_name': get_location_name(float(lat), float(lng)) if lat is not None and lng is not None else None
        }
    except Exception as e:
        print(f"Image processing error: {e}")
        return False, f"Error processing image: {str(e)}"


# ----------------------------
# MongoDB Persistence
# ----------------------------
def store_detection_data(lat, lng, image_path, detection_count, pollution_score, detections):
    try:
        location_name = get_location_name(float(lat), float(lng))
        weather_data = get_weather_data(float(lat), float(lng))
        user_id = request.remote_addr if request else "system"

        detections_collection.insert_one({
            "timestamp": datetime.utcnow(),  # store in UTC
            "latitude": float(lat),
            "longitude": float(lng),
            "location_name": location_name,
            "image_path": image_path,
            "detection_count": int(detection_count),
            "pollution_score": float(pollution_score),
            "detections": detections,  # list of dicts
            "weather_data": weather_data,
            "user_id": user_id
        })

        update_pollution_zones(float(lat), float(lng), float(pollution_score))
        print(f"Detection data stored: {detection_count} objects, score: {pollution_score}")
    except PyMongoError as e:
        print(f"MongoDB insert error: {e}")


def update_pollution_zones(lat, lng, pollution_score):
    try:
        # Find zone within ~1km box (approx for small deltas)
        zone = zones_collection.find_one({
            "center_lat": {"$gte": lat - 0.01, "$lte": lat + 0.01},
            "center_lng": {"$gte": lng - 0.01, "$lte": lng + 0.01}
        })
        if zone:
            new_score = (float(zone.get("pollution_level", 0)) + float(pollution_score)) / 2
            zones_collection.update_one(
                {"_id": zone["_id"]},
                {"$set": {"pollution_level": new_score, "last_updated": datetime.utcnow()}}
            )
        else:
            zones_collection.insert_one({
                "zone_name": f"Zone_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "center_lat": float(lat),
                "center_lng": float(lng),
                "radius": 0.005,
                "pollution_level": float(pollution_score),
                "last_updated": datetime.utcnow()
            })
    except PyMongoError as e:
        print(f"Pollution zone update error: {e}")


# ----------------------------
# Map Generation (Flask only)
# ----------------------------
def build_pollution_map_html(save_path: str):
    # Center map (India) with proper bounds to prevent horizontal wrapping
    f = folium.Figure(width=1000, height=5000)
    m = folium.Map(
        location=[20.5937, 78.9629], 
        zoom_start=5, 
        control_scale=True, 
        max_bounds=True, 
        world_copy_jump=False,
        tiles=None
    ).add_to(f)

    google_api_key = os.getenv("AIzaSyD-WYOIr4Ya4D7kE6NuVWMGfu7zXfjqsng")

    # Add Google Maps layers with no_wrap=True to prevent repetition
    folium.TileLayer(
        tiles=f"https://maps.googleapis.com/maps/vt?lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={google_api_key}",
        attr="Google Maps", 
        name="Google Maps", 
        overlay=False, 
        control=True,
        no_wrap=True  # This is the key fix
    ).add_to(m)
    
    folium.TileLayer(
        tiles=f"https://maps.googleapis.com/maps/vt?lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={google_api_key}",
        attr="Google Satellite", 
        name="Satellite", 
        overlay=False, 
        control=True,
        no_wrap=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles=f"https://maps.googleapis.com/maps/vt?lyrs=y&x={{x}}&y={{y}}&z={{z}}&key={google_api_key}",
        attr="Google Hybrid", 
        name="Hybrid", 
        overlay=False, 
        control=True,
        no_wrap=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles=f"https://maps.googleapis.com/maps/vt?lyrs=p&x={{x}}&y={{y}}&z={{z}}&key={google_api_key}",
        attr="Google Terrain", 
        name="Terrain", 
        overlay=False, 
        control=True,
        no_wrap=True
    ).add_to(m)

    # Alternative: Use OpenStreetMap as default (naturally doesn't repeat)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(m)

    # Pull recent detections from DB (fallback to sample if none)
    try:
        cursor = detections_collection.find(
            {"latitude": {"$ne": None}, "longitude": {"$ne": None}},
            {"latitude": 1, "longitude": 1, "pollution_score": 1}
        ).sort("timestamp", -1).limit(500)
        detections = [{"latitude": d["latitude"], "longitude": d["longitude"],
                       "pollution_score": float(d.get("pollution_score", 0))}
                      for d in cursor]
    except Exception as e:
        print(f"DB fetch for map failed: {e}")
        detections = []

    if not detections:
        # Fallback example points
        detections = [
            {"latitude": 28.61, "longitude": 77.20, "pollution_score": 30},  # Delhi Low
            {"latitude": 19.07, "longitude": 72.87, "pollution_score": 55},  # Mumbai Medium
            {"latitude": 13.08, "longitude": 80.27, "pollution_score": 85},  # Chennai High
        ]

    # Markers
    for d in detections:
        score = d["pollution_score"]
        color = "green" if score <= 40 else "orange" if score <= 70 else "red"
        folium.CircleMarker(
            location=[d["latitude"], d["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Pollution Score: {score}"
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Additional JavaScript to enforce bounds and prevent wrapping
    bounds_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        for (let key in window) {
            if (window[key] && window[key]._container && window[key].setMaxBounds) {
                const map = window[key];
                const bounds = [[-85, -180], [85, 180]];
                map.setMaxBounds(bounds);
                map.options.maxBoundsViscosity = 1.0;
                map.options.worldCopyJump = false;
                break;
            }
        }
    });
    </script>
    """
    
    # Save the map
    m.save(save_path)
    
    # Read the saved HTML and inject the bounds script
    with open(save_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Insert the script before the closing body tag
    html_content = html_content.replace('</body>', bounds_script + '</body>')
    
    # Write back the modified HTML
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return save_path

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_with_location', methods=['POST'])
def upload_with_location():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        lat = request.form.get('latitude', type=float)
        lng = request.form.get('longitude', type=float)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            output_filename = f"processed_{unique_filename}"
            output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

            success, result = process_image_with_location(file_path, output_path, lat, lng)

            try:
                os.remove(file_path)
            except Exception:
                pass

            if success:
                return jsonify({'success': True, 'result_file': output_filename, 'analysis': result})
            else:
                return jsonify({'error': result})

        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.'})
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'})


@app.route('/capture_image', methods=['POST'])
def capture_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'})

        image_data = data.get('image')
        lat = data.get('latitude')
        lng = data.get('longitude')

        # coerce lat/lng if they exist
        try:
            lat = float(lat) if lat is not None else None
            lng = float(lng) if lng is not None else None
        except (TypeError, ValueError):
            lat = None
            lng = None

        if not image_data:
            return jsonify({'error': 'No image data received'})

        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'})

        unique_filename = f"capture_{uuid.uuid4()}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        with open(file_path, 'wb') as f:
            f.write(image_bytes)

        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

        success, result = process_image_with_location(file_path, output_path, lat, lng)

        try:
            os.remove(file_path)
        except Exception:
            pass

        if success:
            return jsonify({'success': True, 'result_file': output_filename, 'analysis': result})
        else:
            return jsonify({'error': result})
    except Exception as e:
        print(f"Capture error: {e}")
        return jsonify({'error': f'Error processing capture: {str(e)}'})


@app.route('/generate_pollution_map')
def generate_pollution_map():
    """
    Builds a fresh map HTML file and serves it.
    """
    try:
        map_path = os.path.abspath(os.path.join('maps', 'pollution_map.html'))
        build_pollution_map_html(map_path)
        # Serve with explicit MIME
        return send_file(map_path, mimetype='text/html')
    except Exception as e:
        print(f"Map generation/serving error: {e}")
        return jsonify({'error': f'Error generating map: {str(e)}'}), 500


# (Optional) back-compat alias: serve the same map file if it exists or build it first
@app.route('/get_map')
def get_map():
    try:
        map_path = os.path.abspath(os.path.join('maps', 'pollution_map.html'))
        if not os.path.exists(map_path):
            build_pollution_map_html(map_path)
        return send_file(map_path, mimetype='text/html')
    except Exception as e:
        print(f"Map serving error: {e}")
        return jsonify({'error': f'Error generating map: {str(e)}'}), 500


@app.route('/get_pollution_data')
def get_pollution_data():
    try:
        now = datetime.utcnow()
        # Recent stats (last 24h)
        recent_pipeline = [
            {"$match": {"timestamp": {"$gte": now - timedelta(hours=24)}}},
            {"$group": {
                "_id": None,
                "total_detections": {"$sum": 1},
                "avg_pollution_score": {"$avg": "$pollution_score"},
                "max_pollution_score": {"$max": "$pollution_score"}
            }}
        ]
        rs = list(detections_collection.aggregate(recent_pipeline))
        if rs:
            rs0 = rs[0]
            recent_stats = {
                'total_detections': int(rs0.get('total_detections', 0)),
                'avg_pollution_score': round(float(rs0.get('avg_pollution_score', 0) or 0), 2),
                'max_pollution_score': float(rs0.get('max_pollution_score', 0) or 0)
            }
        else:
            recent_stats = {'total_detections': 0, 'avg_pollution_score': 0, 'max_pollution_score': 0}

        # Hotspots (top 5 by avg score)
        hotspots_pipeline = [
            {"$match": {"latitude": {"$ne": None}, "location_name": {"$ne": None}}},
            {"$group": {
                "_id": "$location_name",
                "avg_score": {"$avg": "$pollution_score"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"avg_score": -1}},
            {"$limit": 5}
        ]
        hs = list(detections_collection.aggregate(hotspots_pipeline))
        hotspots = [{"name": h["_id"] or "Unknown", "score": round(float(h.get("avg_score", 0)), 2),
                     "count": int(h.get("count", 0))} for h in hs]

        # Trend data (last 7 days, average per day)
        trend_pipeline = [
            {"$match": {"timestamp": {"$gte": now - timedelta(days=7)}}},
            {"$addFields": {"date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}}},
            {"$group": {"_id": "$date", "avg_score": {"$avg": "$pollution_score"}}},
            {"$sort": {"_id": 1}}
        ]
        td = list(detections_collection.aggregate(trend_pipeline))
        trend_data = [{"date": t["_id"], "score": round(float(t.get("avg_score", 0)), 2)} for t in td]

        return jsonify({'recent_stats': recent_stats, 'hotspots': hotspots, 'trend_data': trend_data})

    except Exception as e:
        print(f"Pollution data error: {e}")
        return jsonify({'recent_stats': {'total_detections': 0, 'avg_pollution_score': 0, 'max_pollution_score': 0},
                        'hotspots': [], 'trend_data': []})


@app.route('/result/<filename>')
def get_result(filename):
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(file_path):
            # Guess MIME by extension
            ext = os.path.splitext(filename)[1].lower()
            mimetype = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png' if ext == '.png' else 'application/octet-stream'
            return send_file(file_path, mimetype=mimetype)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"File serving error: {e}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500


@app.route('/health')
def health_check():
    try:
        client.admin.command('ping')
        db_ok = True
    except Exception:
        db_ok = False
    return jsonify({
        'status': 'healthy',
        'yolo_model_loaded': yolo_model is not None,
        'database_accessible': db_ok,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })


# ----------------------------
# Error Handlers (Flask)
# ----------------------------
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    print("Initializing Garbage Detection System (MongoDB Atlas)...")
    print("Server starting on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
