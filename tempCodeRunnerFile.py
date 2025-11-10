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