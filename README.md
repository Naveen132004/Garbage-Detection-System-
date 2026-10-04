---
title: Garbage Detection System
emoji: ♻️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ♻️ Garbage Detection System

An AI-based garbage detection system. Upload a photo or use your camera, and a custom YOLO model finds garbage in the picture, gives the spot a pollution score from 0 to 100, and adds it to a shared pollution map.

## Features

- Detect garbage in uploaded photos or live camera shots (YOLO, OpenCV)
- Pollution score with Low / Medium / High levels (under 30, 30 to 69, 70 and above)
- Location from the browser, from the photo's GPS data, or picked on a map
- Dashboard: reports in the last 24 hours, 7-day trend, top hotspots, latest reports
- Pollution map with a colour legend (OpenStreetMap, plus Google layers when a key is set)
- Works on phones, supports dark mode
- Stores results in MongoDB Atlas, or in a local JSON file when no database is configured

## Tech stack

Python, Flask, Ultralytics YOLO, OpenCV, cvzone, Folium / Leaflet, MongoDB (pymongo), geopy.

## Getting started

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the values you need
export $(grep -v '^#' .env | xargs)   # or set the variables another way
python app1.py
```

Open http://127.0.0.1:5000.

The model weights must be at `Weights/best.pt` (or set `MODEL_PATH`). Without them the page loads but detection is disabled, and the status pill at the top says "Model not loaded".

### Configuration

All secrets come from environment variables; see `.env.example`.

| Variable | Purpose |
| --- | --- |
| `MONGO_URI` | MongoDB connection string. Empty means results go to `data/detections.json`. |
| `GOOGLE_MAPS_API_KEY` | Optional, adds satellite/hybrid/terrain map layers. |
| `OPENWEATHER_API_KEY` | Optional, saves real weather with each report. |
| `APP_TIMEZONE` | Timezone for the dashboard's days (default `Asia/Kolkata`). |
| `HOST`, `PORT`, `FLASK_DEBUG` | Server settings. Keep `FLASK_DEBUG=0` on any shared network. |

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space with the **Docker** SDK and the **Blank** template. Free CPU hardware is enough.
2. In the Space's **Settings → Variables and secrets**, add a secret `MONGO_URI` (your MongoDB Atlas connection string, with a NEW password). Without it, results are stored inside the container and lost on every restart. Optionally add `GOOGLE_MAPS_API_KEY` and `OPENWEATHER_API_KEY`.
3. Push this repo to the Space, using a Hugging Face write token as the password (https://huggingface.co/settings/tokens):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. The Space builds the `Dockerfile`. The model is downloaded at build time from `MODEL_URL` (a public waste model by default). To use your own model, set a `MODEL_URL` build variable, or add `Weights/best.pt` with Git LFS.

The app is then live at `https://<your-username>-<space-name>.hf.space`.

Other hosts: the `Procfile` starts the same production server (`gunicorn app1:app`). The host needs at least 1–2 GB of RAM for PyTorch.

## Project structure

```
app1.py                 Flask web app (API + page)
templates/index.html    Web page
static/css/style.css    Styles
static/js/app.js        Page logic (upload, camera, location, dashboard, map)
GarbageDetector.py      Detect garbage in one image:  python GarbageDetector.py path/to/image.jpg
GarbageDetectorLive.py  Detect garbage in a video or camera:  python GarbageDetectorLive.py 0
graph.py                Charts for every image in Media/
Weights/best.pt         YOLO weights (not in git; downloaded at Docker build)
Dockerfile              Production image (Hugging Face Spaces)
Procfile                Production start command for other hosts
```

## API

| Route | Method | Description |
| --- | --- | --- |
| `/upload_with_location` | POST | Form upload: `file`, optional `latitude`, `longitude` |
| `/capture_image` | POST | JSON: `image` (data URL), optional `latitude`, `longitude` |
| `/get_pollution_data` | GET | Dashboard statistics |
| `/generate_pollution_map` | GET | Pollution map page |
| `/result/<file>` | GET | Processed image |
| `/health` | GET | Model and database status |

Errors return a JSON `{"error": "..."}` with a matching HTTP status code.

## Contributors

Naveen Kumar, B.Tech Computer Science and Business Systems, SRM Institute of Science and Technology. GitHub: https://github.com/Naveen132004

## Future improvements

- Integration with smart city waste systems
- Mobile app
- Better deep learning model
