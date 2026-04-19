"""
CrowdShield - Configuration File
=================================
All settings for the CrowdShield AI Public Safety System.
Change these values to customize detection behavior, thresholds,
video sources, and UI settings.

Author: CrowdShield Team
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
# BASE_DIR = the folder where this config.py file lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")       # Where AI model weights are stored
DATA_DIR = os.path.join(BASE_DIR, "data")            # Sample videos for testing
REPORTS_DIR = os.path.join(BASE_DIR, "reports")      # Generated report files
DB_PATH = os.path.join(BASE_DIR, "crowdshield.db")   # SQLite database for alerts/logs

# Create directories automatically if they don't exist
for _dir in [MODELS_DIR, DATA_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ============================================================
# MODEL SETTINGS
# ============================================================
# YOLOv8 model sizes:
#   'n' = nano   (fastest, least accurate, ~6MB)
#   's' = small  (fast, good accuracy, ~22MB)
#   'm' = medium (balanced speed/accuracy, ~50MB) <-- RECOMMENDED for RTX 4050
#   'l' = large  (slower, most accurate, ~84MB)
#   'x' = xlarge (slowest, highest accuracy, ~131MB)
YOLO_MODEL_SIZE = "s"

# Model file names (auto-downloaded on first run by ultralytics)
YOLO_DETECT_MODEL = f"yolov8{YOLO_MODEL_SIZE}.pt"    # Person + object detection
YOLO_POSE_MODEL = f"yolov8{YOLO_MODEL_SIZE}-pose.pt"  # Pose estimation for fights

# ============================================================
# DETECTION CONFIDENCE THRESHOLDS
# ============================================================
# Range: 0.0 to 1.0
# Lower value = more detections but more false positives
# Higher value = fewer detections but more reliable
PERSON_CONFIDENCE = 0.35      # Confidence for detecting people
WEAPON_CONFIDENCE = 0.30      # Confidence for detecting weapons (lower = catch more)
POSE_CONFIDENCE = 0.40        # Confidence for pose keypoints

# COCO Dataset Class IDs we care about:
# Full list: https://docs.ultralytics.com/datasets/detect/coco/
PERSON_CLASS_ID = 0           # 'person' in COCO
WEAPON_CLASS_IDS = [43, 76]   # 43='knife', 76='scissors' in COCO
# Note: COCO doesn't have 'gun' - for guns, a custom-trained model is needed
# We provide a placeholder to swap in a weapon-specific model later

# IOU (Intersection over Union) threshold for Non-Max Suppression
# Removes overlapping duplicate boxes
NMS_IOU_THRESHOLD = 0.5


# ============================================================
# TRACKING SETTINGS (ByteTrack)
# ============================================================
TRACKER_THRESH = 0.25         # Minimum confidence to start tracking
TRACKER_BUFFER = 30           # Frames to keep a lost track alive before removing
TRACKER_MATCH_THRESH = 0.8    # IOU threshold for matching detections to tracks
TRACKER_FRAME_RATE = 30       # Expected frame rate of the video


# ============================================================
# STAMPEDE DETECTION SETTINGS
# ============================================================
# Stampede is detected using optical flow (motion patterns):
# - High, uniform motion in one direction = potential stampede
# - Sudden density drop = people falling down
STAMPEDE_FLOW_THRESHOLD = 4.0        # Min average optical flow magnitude to flag
STAMPEDE_UNIFORMITY_THRESH = 0.55    # How aligned the flow must be (0=random, 1=perfect)
STAMPEDE_DENSITY_DROP = 0.3          # 30% density drop triggers concern
STAMPEDE_MIN_PEOPLE = 5              # Need at least this many people to check stampede
STAMPEDE_CONSEC_FRAMES = 8           # Alert only after this many consecutive alert frames
STAMPEDE_COOLDOWN_SEC = 15           # Seconds between repeated stampede alerts


# ============================================================
# FIGHT DETECTION SETTINGS
# ============================================================
# Fight detection uses pose keypoints to find aggressive movements:
# - Fast arm/wrist motion (punching/hitting)
# - Two people very close together
# - Sustained aggressive posture over several frames
FIGHT_PROXIMITY_PX = 120             # Max pixel distance between two people to check
FIGHT_VELOCITY_THRESH = 20.0         # Wrist/elbow velocity threshold (pixels/frame)
FIGHT_CONSEC_FRAMES = 6              # Frames of fighting before alert
FIGHT_COOLDOWN_SEC = 15              # Seconds between repeated fight alerts
FIGHT_MIN_PEOPLE = 2                 # Need at least 2 people close together


# ============================================================
# CHILD DETECTION SETTINGS
# ============================================================
# Children are detected by:
#   1. Bounding box height compared to average adult height in frame
#   2. Isolation check (no adult nearby)
CHILD_HEIGHT_RATIO = 0.42            # If person height is < 42% of avg adult = child (strict)
CHILD_MIN_HEIGHT_PX = 80             # Minimum pixel height to be considered (filter far-away people)
CHILD_ISOLATION_DIST_PX = 200        # If no adult within this distance = "alone"
CHILD_ALONE_FRAMES = 60              # Frames alone before alert (more conservative)
CHILD_COOLDOWN_SEC = 20              # Seconds between repeated child alerts
CHILD_TOP_FRAME_IGNORE = 0.30        # Ignore people in top 30% of frame (far away = small = false child)


# ============================================================
# CROWD ANALYZER SETTINGS
# ============================================================
CROWD_HISTORY_LENGTH = 600           # Frames of people-count history to keep (~20 sec at 30fps)
CROWD_SURGE_THRESHOLD = 0.5          # 50% sudden increase in count = crowd surge
CROWD_HIGH_THRESHOLD = 50            # More than this many people = high density warning


# ============================================================
# HEATMAP SETTINGS
# ============================================================
HEATMAP_DECAY = 0.985                # How fast old positions fade (0.99=slow, 0.95=fast)
HEATMAP_INTENSITY = 3                # Size multiplier for each person's heat blob
HEATMAP_RADIUS = 25                  # Pixel radius of each heat blob
HEATMAP_COLORMAP = 11                # OpenCV colormap: 11=JET (rainbow), 2=HOT


# ============================================================
# NIGHT VISION SETTINGS
# ============================================================
# Automatic low-light enhancement using CLAHE
# (Contrast Limited Adaptive Histogram Equalization)
NIGHT_VISION_AUTO = True             # Auto-detect low light and enhance
NIGHT_VISION_BRIGHTNESS_THRESH = 60  # Below this average brightness = "dark" (0-255)
NIGHT_VISION_CLIP_LIMIT = 3.0       # CLAHE clip limit (higher = more contrast)
NIGHT_VISION_TILE_SIZE = (8, 8)     # CLAHE tile grid size


# ============================================================
# TARGET TRACKING & RE-IDENTIFICATION SETTINGS
# ============================================================
# When you select a person to track:
#   1. Their appearance features (color histogram) are saved
#   2. If they leave the frame, the system tries to re-identify them
TARGET_REID_MATCH_THRESH = 0.55      # Similarity score to confirm re-identification
TARGET_LOST_TIMEOUT = 300            # Give up searching after this many frames
TARGET_FEATURE_BINS = 16             # Color histogram bins per channel


# ============================================================
# ALERT SETTINGS
# ============================================================
ALERT_COOLDOWN_DEFAULT = 10          # Default seconds between same-type alerts
ALERT_SOUND_ENABLED = True           # Play a beep sound on alert (Windows only)
ALERT_MAX_LOG = 1000                 # Max alerts to keep in memory

# Severity levels for each alert type
ALERT_SEVERITY = {
    "stampede":    "CRITICAL",  # Red
    "weapon":     "CRITICAL",  # Red
    "fight":      "HIGH",      # Orange
    "lost_child": "HIGH",      # Orange
    "crowd_surge": "MEDIUM",   # Yellow
    "crowd_high": "LOW",       # Blue
}


# ============================================================
# VIDEO SOURCE SETTINGS
# ============================================================
DEFAULT_SOURCE = 0                   # 0 = default webcam, 1 = second camera, etc.
IP_CAM_URL = ""                      # Phone IP camera URL (e.g., "http://192.168.1.5:8080/video")
MAX_FRAME_WIDTH = 1280               # Resize frames wider than this
MAX_FRAME_HEIGHT = 720               # Resize frames taller than this
TARGET_FPS = 30                      # Target display frame rate


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================
# When FPS drops below these thresholds, the system auto-adjusts:
FPS_LOW_THRESHOLD = 10               # Below this = switch to smaller model or skip frames
FRAME_SKIP_RATIO = 2                 # Process every Nth frame when FPS is low
ADAPTIVE_SKIP = True                 # Auto-increase frame skip when FPS is low
POSE_EVERY_N_FRAMES = 4              # Only run pose estimation every Nth frame (saves ~40ms)
USE_HALF_PRECISION = True            # Use FP16 on GPU for faster inference (RTX 4050 supports this)


# ============================================================
# UI / APP SETTINGS
# ============================================================
APP_NAME = "CrowdShield"
APP_ICON = "🛡️"
APP_TAGLINE = "AI-Powered Public Safety & Stampede Detection System"

# Color scheme for the UI
COLOR_CRITICAL = "#FF0000"
COLOR_HIGH = "#FF6600"
COLOR_MEDIUM = "#FFD700"
COLOR_LOW = "#4488FF"
COLOR_SAFE = "#00CC00"
