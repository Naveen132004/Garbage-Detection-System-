from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import html
import json
import uuid
import math
import base64
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import cv2
import cvzone
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from PIL import Image, ExifTags
import folium
from branca.element import Template, MacroElement
import requests

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from geopy.geocoders import Nominatim
except ImportError:
    Nominatim = None

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    MongoClient = None
    PyMongoError = Exception


# ----------------------------
# Configuration (all secrets come from environment variables, see .env.example)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(BASE_DIR, 'results')
app.config['DATA_FOLDER'] = os.path.join(BASE_DIR, 'data')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

for folder in (app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER'],
               app.config['DATA_FOLDER'], os.path.join(BASE_DIR, 'maps')):
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "Weights", "best.pt"))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
APP_TIMEZONE = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))


# ----------------------------
# Labels, weights & score levels
# ----------------------------
class_labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# Friendly names shown to users instead of the raw model labels
display_names = {
    '0': 'Waste',
    'c': 'Waste',
    'garbage': 'Garbage',
    'garbage_bag': 'Garbage bag',
    'sampah-detection': 'Litter',
    'trash': 'Trash',
    'battery': 'Battery',
    'biological': 'Food / organic waste',
    'brown-glass': 'Brown glass',
    'cardboard': 'Cardboard',
    'clothes': 'Clothes',
    'green-glass': 'Green glass',
    'metal': 'Metal',
    'paper': 'Paper',
    'plastic': 'Plastic',
    'shoes': 'Shoes',
    'white-glass': 'White glass'
}

pollution_weights = {
    '0': 1,
    'c': 1,
    'garbage': 3,
    'garbage_bag': 5,
    'sampah-detection': 2,
    'trash': 2,
    'battery': 5,
    'biological': 3,
    'brown-glass': 3,
    'green-glass': 3,
    'white-glass': 3,
    'cardboard': 2,
    'clothes': 2,
    'metal': 3,
    'paper': 2,
    'plastic': 3,
    'shoes': 2
}

# One set of thresholds used everywhere (image overlay, map, API, page)
SCORE_LEVELS = [
    (30, 'low', 'Low', '#2e9e5b', (91, 158, 46)),
    (70, 'medium', 'Medium', '#e0a100', (0, 161, 224)),
    (101, 'high', 'High', '#d64545', (69, 69, 214)),
]


def score_level(score):
    for limit, key, label, hex_color, bgr in SCORE_LEVELS:
        if score < limit:
            return {'key': key, 'label': label, 'color': hex_color, 'bgr': bgr}
    limit, key, label, hex_color, bgr = SCORE_LEVELS[-1]
    return {'key': key, 'label': label, 'color': hex_color, 'bgr': bgr}


def calculate_pollution_score(detections):
    if not detections:
        return 0
    total_score = 0
    for detection in detections:
        weight = pollution_weights.get(detection.get('class', 'garbage'), 2)
        total_score += weight * detection.get('confidence', 0)
    return min(100, round(total_score * 10, 2))


# ----------------------------
# YOLO model
# ----------------------------
yolo_model = None
if YOLO is None:
    print("Warning: ultralytics is not installed; detection is disabled.")
elif not os.path.exists(MODEL_PATH):
    print(f"Warning: model weights not found at {MODEL_PATH}; detection is disabled.")
else:
    try:
        yolo_model = YOLO(MODEL_PATH)
        # Use the class names stored in the weights, so any YOLO model works
        names = getattr(yolo_model, 'names', None)
        if names:
            class_labels = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        print(f"YOLO model loaded successfully ({len(class_labels)} classes)")
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")


# ----------------------------
# Storage: MongoDB when MONGO_URI is set, otherwise a local JSON file
# ----------------------------
class LocalStore:
    """Keeps detections in data/detections.json so the app works without a database."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def _read(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def insert(self, doc):
        with self.lock:
            docs = self._read()
            stored = dict(doc, timestamp=doc['timestamp'].isoformat())
            docs.append(stored)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, indent=2)

    def find(self, since=None, limit=None):
        docs = []
        for d in self._read():
            try:
                d['timestamp'] = datetime.fromisoformat(d['timestamp'])
            except (KeyError, TypeError, ValueError):
                continue
            if d['timestamp'].tzinfo is None:
                d['timestamp'] = d['timestamp'].replace(tzinfo=timezone.utc)
            if since is None or d['timestamp'] >= since:
                docs.append(d)
        docs.sort(key=lambda d: d['timestamp'], reverse=True)
        return docs[:limit] if limit else docs

    def ping(self):
        return True

    name = 'local'


class MongoStore:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        db = self.client[db_name]
        self.detections = db["detections"]
        self.zones = db["pollution_zones"]

    def insert(self, doc):
        self.detections.insert_one(dict(doc))
        self._update_zone(doc['latitude'], doc['longitude'], doc['pollution_score'])

    def _update_zone(self, lat, lng, pollution_score):
        # Find a zone within ~1 km (approximation for small deltas)
        zone = self.zones.find_one({
            "center_lat": {"$gte": lat - 0.01, "$lte": lat + 0.01},
            "center_lng": {"$gte": lng - 0.01, "$lte": lng + 0.01}
        })
        now = datetime.now(timezone.utc)
        if zone:
            # Running average over every report in the zone, not just the last two
            count = int(zone.get("report_count", 1))
            new_score = (float(zone.get("pollution_level", 0)) * count + pollution_score) / (count + 1)
            self.zones.update_one(
                {"_id": zone["_id"]},
                {"$set": {"pollution_level": new_score, "report_count": count + 1, "last_updated": now}}
            )
        else:
            self.zones.insert_one({
                "zone_name": f"Zone_{now.strftime('%Y%m%d_%H%M%S')}",
                "center_lat": lat,
                "center_lng": lng,
                "radius": 0.005,
                "pollution_level": pollution_score,
                "report_count": 1,
                "last_updated": now
            })

    def find(self, since=None, limit=None):
        query = {"timestamp": {"$gte": since}} if since else {}
        cursor = self.detections.find(query, {"weather_data": 0}).sort("timestamp", -1)
        if limit:
            cursor = cursor.limit(limit)
        docs = list(cursor)
        for d in docs:
            if d['timestamp'].tzinfo is None:
                d['timestamp'] = d['timestamp'].replace(tzinfo=timezone.utc)
        return docs

    def ping(self):
        self.client.admin.command('ping')
        return True

    name = 'mongodb'


MONGO_URI = os.getenv("MONGO_URI")
store = LocalStore(os.path.join(app.config['DATA_FOLDER'], 'detections.json'))
if MONGO_URI and MongoClient is not None:
    try:
        store = MongoStore(MONGO_URI, os.getenv("MONGO_DB", "garbage_db"))
        print("Using MongoDB storage")
    except Exception as e:
        print(f"Warning: MongoDB unavailable ({e}); using local storage.")
else:
    print("MONGO_URI not set; saving detections to data/detections.json")


# ----------------------------
# Geocoder & Weather
# ----------------------------
geolocator = None
if Nominatim is not None:
    geolocator = Nominatim(user_agent=os.getenv("GEOCODER_USER_AGENT", "garbage_detector_app/2.0"))


def get_location_name(lat, lng):
    if not geolocator:
        return f"Location ({lat:.4f}, {lng:.4f})"
    try:
        location = geolocator.reverse(f"{lat}, {lng}", timeout=5)
        return location.address if location else f"Location ({lat:.4f}, {lng:.4f})"
    except Exception as e:
        print(f"Geocoding error: {e}")
        return f"Location ({lat:.4f}, {lng:.4f})"


def get_weather_data(lat, lng):
    """Real weather from OpenWeather when OPENWEATHER_API_KEY is set, otherwise nothing."""
    if not OPENWEATHER_API_KEY:
        return None
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lng, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=5
        )
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
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            if exif_data is not None:
                for tag, value in exif_data.items():
                    if ExifTags.TAGS.get(tag, tag) == 'GPSInfo':
                        gps_info = value
                        if 2 in gps_info and 4 in gps_info:
                            lat = convert_to_degrees(gps_info[2])
                            if gps_info.get(1) == 'S':
                                lat = -lat
                            lng = convert_to_degrees(gps_info[4])
                            if gps_info.get(3) == 'W':
                                lng = -lng
                            if lat is not None and lng is not None:
                                return lat, lng
    except Exception as e:
        print(f"GPS extraction error: {e}")
    return None, None


def convert_to_degrees(value):
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            d, m, s = value[:3]
            d = float(d[0]) / float(d[1]) if isinstance(d, tuple) else float(d)
            m = float(m[0]) / float(m[1]) if isinstance(m, tuple) else float(m)
            s = float(s[0]) / float(s[1]) if isinstance(s, tuple) else float(s)
            return d + m / 60.0 + s / 3600.0
        return float(value)
    except Exception:
        return None


def valid_coordinates(lat, lng):
    return (lat is not None and lng is not None
            and -90 <= lat <= 90 and -180 <= lng <= 180)


def parse_float(value):
    try:
        return float(value) if value not in (None, '') else None
    except (TypeError, ValueError):
        return None


# ----------------------------
# Detection Pipeline
# ----------------------------
class DetectionError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def process_image_with_location(image_path, output_path, lat=None, lng=None):
    if yolo_model is None:
        raise DetectionError("The detection model isn't loaded. Put the weights file at Weights/best.pt and restart the server.", 503)

    img = cv2.imread(image_path)
    if img is None:
        raise DetectionError("That file couldn't be read as an image. Try a JPG or PNG.")

    location_source = 'device' if valid_coordinates(lat, lng) else None
    if not location_source:
        img_lat, img_lng = extract_gps_info(image_path)
        if valid_coordinates(img_lat, img_lng):
            lat, lng, location_source = img_lat, img_lng, 'photo'
        else:
            lat = lng = None

    results = yolo_model(img, verbose=False)

    detections = []
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(float(box.conf[0]) * 100) / 100
            cls = int(box.cls[0])

            if conf > CONFIDENCE_THRESHOLD and 0 <= cls < len(class_labels):
                label = class_labels[cls]
                detections.append({
                    'class': label,
                    'label': display_names.get(label, label.replace('_', ' ').replace('-', ' ').capitalize()),
                    'confidence': conf,
                    'bbox': [x1, y1, w, h]
                })
                color = (0, 200, 0) if conf > 0.7 else (0, 200, 255) if conf > 0.5 else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), t=2, colorR=color)
                cvzone.putTextRect(
                    img, f"{display_names.get(label, label)} {conf:.2f}",
                    (max(0, x1), max(20, y1 - 10)),
                    scale=0.8, thickness=1, colorR=(40, 40, 40), colorT=(255, 255, 255)
                )

    pollution_score = calculate_pollution_score(detections)
    level = score_level(pollution_score)
    local_now = datetime.now(APP_TIMEZONE)

    if lat is not None:
        cv2.putText(img, f"Lat: {lat:.6f}, Lng: {lng:.6f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, local_now.strftime("%Y-%m-%d %H:%M:%S"), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Pollution Score: {pollution_score} ({level['label']})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, level['bgr'], 2)

    cv2.imwrite(output_path, img)

    # Count per friendly name for the result card
    breakdown = {}
    for d in detections:
        breakdown[d['label']] = breakdown.get(d['label'], 0) + 1

    location_name = get_location_name(lat, lng) if lat is not None else None
    saved, save_message = False, "Not saved to the map because the photo has no location. Allow location access or enter it manually."
    if lat is not None:
        saved, save_message = store_detection_data(lat, lng, output_path, location_name,
                                                   pollution_score, detections)

    return {
        'detections': detections,
        'count': len(detections),
        'breakdown': breakdown,
        'pollution_score': pollution_score,
        'level': {k: level[k] for k in ('key', 'label', 'color')},
        'coordinates': {'lat': lat, 'lng': lng} if lat is not None else None,
        'location_source': location_source,
        'location_name': location_name,
        'saved': saved,
        'save_message': save_message,
        'timestamp': local_now.isoformat()
    }


def store_detection_data(lat, lng, image_path, location_name, pollution_score, detections):
    try:
        store.insert({
            "timestamp": datetime.now(timezone.utc),
            "latitude": lat,
            "longitude": lng,
            "location_name": location_name,
            "image_path": os.path.basename(image_path),
            "detection_count": len(detections),
            "pollution_score": float(pollution_score),
            "detections": detections,
            "weather_data": get_weather_data(lat, lng),
            "user_id": request.remote_addr if request else "system"
        })
        return True, "Saved to the pollution map."
    except (PyMongoError, OSError) as e:
        print(f"Storage error: {e}")
        return False, "The result couldn't be saved to the database, so it won't appear on the map."


def run_detection(file_path, lat, lng):
    output_filename = f"processed_{os.path.splitext(os.path.basename(file_path))[0]}.jpg"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
    try:
        analysis = process_image_with_location(file_path, output_path, lat, lng)
        return jsonify({'success': True, 'result_file': output_filename, 'analysis': analysis})
    except DetectionError as e:
        return jsonify({'error': str(e)}), e.status
    except Exception as e:
        print(f"Image processing error: {e}")
        return jsonify({'error': "Something went wrong while analysing the image. Please try again."}), 500
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


# ----------------------------
# Map Generation
# ----------------------------
LEGEND_TEMPLATE = """
{% macro html(this, kwargs) %}
<div style="position: fixed; bottom: 24px; left: 12px; z-index: 9999; background: rgba(255,255,255,.95);
            padding: 10px 12px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.2);
            font: 13px/1.5 system-ui, sans-serif; color: #1d2a22;">
  <strong>Pollution score</strong><br>
  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2e9e5b;"></span> Low (under 30)<br>
  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#e0a100;"></span> Medium (30 to 69)<br>
  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#d64545;"></span> High (70 and above)
</div>
{% endmacro %}
"""


def build_pollution_map_html(save_path):
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        control_scale=True,
        max_bounds=True,
        world_copy_jump=False,
        tiles=None,
        min_zoom=2
    )

    folium.TileLayer("OpenStreetMap", name="Street map", no_wrap=True).add_to(m)
    if GOOGLE_MAPS_API_KEY:
        for code, name in (('s', 'Satellite'), ('y', 'Hybrid'), ('p', 'Terrain')):
            folium.TileLayer(
                tiles=f"https://maps.googleapis.com/maps/vt?lyrs={code}&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}",
                attr="Google Maps", name=name, overlay=False, control=True, no_wrap=True
            ).add_to(m)

    try:
        docs = store.find(limit=500)
    except Exception as e:
        print(f"Map data fetch failed: {e}")
        docs = []

    points = []
    for d in docs:
        lat, lng = d.get("latitude"), d.get("longitude")
        if not valid_coordinates(lat, lng):
            continue
        score = float(d.get("pollution_score", 0))
        level = score_level(score)
        when = d['timestamp'].astimezone(APP_TIMEZONE).strftime("%d %b %Y, %H:%M")
        popup = folium.Popup(
            f"<b>{level['label']} pollution</b> (score {score:g})<br>"
            f"{d.get('detection_count', 0)} item(s) detected<br>"
            f"{html.escape(str(d.get('location_name') or ''))}<br>"
            f"<small>{when}</small>",
            max_width=260
        )
        folium.CircleMarker(
            location=[lat, lng], radius=8, color=level['color'], weight=2,
            fill=True, fill_color=level['color'], fill_opacity=0.7, popup=popup
        ).add_to(m)
        points.append([lat, lng])

    if points:
        m.fit_bounds(points, max_zoom=14, padding=(30, 30))

    legend = MacroElement()
    legend._template = Template(LEGEND_TEMPLATE)
    m.get_root().add_child(legend)
    if GOOGLE_MAPS_API_KEY:
        folium.LayerControl().add_to(m)

    m.save(save_path)
    return save_path


# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html', max_upload_mb=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024))


@app.route('/upload_with_location', methods=['POST'])
def upload_with_location():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'Please choose an image to upload.'}), 400

    filename = secure_filename(file.filename) or 'upload.jpg'
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        return jsonify({'error': 'Unsupported file type. Please upload a JPG, PNG, BMP or WEBP image.'}), 400

    lat = parse_float(request.form.get('latitude'))
    lng = parse_float(request.form.get('longitude'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    file.save(file_path)
    return run_detection(file_path, lat, lng)


@app.route('/capture_image', methods=['POST'])
def capture_image():
    data = request.get_json(silent=True)
    if not data or not data.get('image'):
        return jsonify({'error': 'No photo was received from the camera.'}), 400

    image_data = data['image']
    lat = parse_float(data.get('latitude'))
    lng = parse_float(data.get('longitude'))

    try:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)
    except Exception:
        return jsonify({'error': 'The camera photo was damaged in transit. Please try again.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"capture_{uuid.uuid4().hex}.jpg")
    with open(file_path, 'wb') as f:
        f.write(image_bytes)
    return run_detection(file_path, lat, lng)


@app.route('/generate_pollution_map')
def generate_pollution_map():
    """Builds a fresh map HTML file and serves it."""
    try:
        map_path = os.path.join(BASE_DIR, 'maps', 'pollution_map.html')
        build_pollution_map_html(map_path)
        return send_from_directory(os.path.dirname(map_path), os.path.basename(map_path), mimetype='text/html')
    except Exception as e:
        print(f"Map generation error: {e}")
        return jsonify({'error': 'The map could not be generated.'}), 500


@app.route('/get_map')
def get_map():
    return generate_pollution_map()


@app.route('/get_pollution_data')
def get_pollution_data():
    now = datetime.now(timezone.utc)
    try:
        week = store.find(since=now - timedelta(days=7))
        recent_docs = store.find(limit=200)
    except Exception as e:
        print(f"Pollution data error: {e}")
        return jsonify({'error': 'Statistics are unavailable right now.'}), 503

    last_day = [d for d in week if d['timestamp'] >= now - timedelta(hours=24)]
    scores = [float(d.get('pollution_score', 0)) for d in last_day]
    recent_stats = {
        'total_detections': len(last_day),
        'items_found': sum(int(d.get('detection_count', 0)) for d in last_day),
        'avg_pollution_score': round(sum(scores) / len(scores), 2) if scores else 0,
        'max_pollution_score': max(scores) if scores else 0
    }

    # Hotspots: top 5 places by average score
    groups = {}
    for d in recent_docs:
        name = d.get('location_name') or 'Unknown location'
        g = groups.setdefault(name, {'name': name, 'scores': [], 'lat': d.get('latitude'), 'lng': d.get('longitude')})
        g['scores'].append(float(d.get('pollution_score', 0)))
    hotspots = sorted(
        ({'name': g['name'], 'score': round(sum(g['scores']) / len(g['scores']), 2),
          'count': len(g['scores']), 'lat': g['lat'], 'lng': g['lng'],
          'level': score_level(sum(g['scores']) / len(g['scores']))['key']}
         for g in groups.values()),
        key=lambda h: h['score'], reverse=True
    )[:5]

    # Trend: daily average for the last 7 days in the app's timezone, including empty days
    today = now.astimezone(APP_TIMEZONE).date()
    days = {(today - timedelta(days=i)).isoformat(): [] for i in range(6, -1, -1)}
    for d in week:
        key = d['timestamp'].astimezone(APP_TIMEZONE).date().isoformat()
        if key in days:
            days[key].append(float(d.get('pollution_score', 0)))
    trend_data = [{'date': k, 'score': round(sum(v) / len(v), 2) if v else None, 'count': len(v)}
                  for k, v in days.items()]

    recent = [{
        'time': d['timestamp'].astimezone(APP_TIMEZONE).isoformat(),
        'location_name': d.get('location_name'),
        'score': float(d.get('pollution_score', 0)),
        'count': int(d.get('detection_count', 0)),
        'level': score_level(float(d.get('pollution_score', 0)))['key']
    } for d in recent_docs[:5]]

    return jsonify({'recent_stats': recent_stats, 'hotspots': hotspots,
                    'trend_data': trend_data, 'recent': recent})


@app.route('/result/<path:filename>')
def get_result(filename):
    # send_from_directory rejects paths that escape the results folder
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/health')
def health_check():
    try:
        db_ok = store.ping()
    except Exception:
        db_ok = False
    return jsonify({
        'status': 'healthy',
        'yolo_model_loaded': yolo_model is not None,
        'database_accessible': db_ok,
        'storage': store.name,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


# ----------------------------
# Error Handlers
# ----------------------------
@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({'error': f'That image is too large. Please use a file under {mb} MB.'}), 413


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
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(f"Garbage Detection System running on http://{host}:{port} (storage: {store.name})")
    app.run(debug=debug, host=host, port=port)
