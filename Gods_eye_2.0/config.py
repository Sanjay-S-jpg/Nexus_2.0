# config.py - Gods Eye 2.0 Configuration
# All settings in one place

import os
import torch

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")

for d in [MODELS_DIR, VIDEOS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# DEVICE - GPU only (RTX 4050), fallback to CPU if unavailable
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL PATHS
# ============================================================
YOLO_MODEL = os.path.join(MODELS_DIR, "yolo11n.pt")       # Detection + tracking
YOLO_POSE_MODEL = os.path.join(MODELS_DIR, "yolo11n-pose.pt")  # Fight detection

# ============================================================
# DETECTION THRESHOLDS
# ============================================================
PERSON_CONFIDENCE = 0.25       # Person detection confidence
WEAPON_CONFIDENCE = 0.35       # Weapon detection confidence
POSE_CONFIDENCE = 0.30         # Pose estimation confidence

# ============================================================
# TRACKING
# ============================================================
TRACKER_TYPE = "botsort"       # botsort > bytetrack for re-ID
TRACK_BUFFER = 60              # Frames to keep lost tracks (2s at 30fps)
MATCH_THRESHOLD = 0.7          # IoU match threshold

# ============================================================
# CROWD DETECTION
# ============================================================
GRID_SIZE = (4, 4)             # Grid for density analysis
DENSITY_CRITICAL = 8           # People per cell = critical
DENSITY_HIGH = 5
DENSITY_MEDIUM = 3

# ============================================================
# STAMPEDE DETECTION (reworked - proper physics-based)
# ============================================================
STAMPEDE_THRESHOLDS = {
    "min_people": 8,           # Need at least this many to consider stampede
    "velocity": 12.0,          # Pixels/frame - running speed at 960x540
    "acceleration": 4.5,       # Sudden speed increase (px/frame²)
    "coherence": 0.6,          # Same direction (0=random, 1=all same)
    "density_change": 0.2,     # Rapid density increase ratio
    "edge_pressure": 0.4,      # Fraction of people near edges
    "sustained_frames": 36,    # Must sustain for ~1.2s (at 30fps)
}

ALERT_LEVELS = {
    0: {"name": "SAFE", "color": (0, 255, 0), "threshold": 0},
    1: {"name": "CAUTION", "color": (0, 255, 255), "threshold": 30},
    2: {"name": "WARNING", "color": (0, 165, 255), "threshold": 55},
    3: {"name": "CRITICAL", "color": (0, 0, 255), "threshold": 80},
}

# ============================================================
# WEAPON DETECTION (guns + knives only)
# ============================================================
WEAPON_CLASSES = {
    43: {"name": "Knife", "danger": "HIGH", "color": (0, 0, 255)},
}
# Note: COCO has no "gun" class. Class 43=knife is the only real weapon.
# We keep it focused to avoid false positives (bottles, bats are NOT weapons)

# ============================================================
# FIGHT DETECTION (pose-based)
# ============================================================
FIGHT_THRESHOLDS = {
    "proximity": 100,          # Max pixel distance between 2 people to check
    "arm_speed": 20.0,         # Wrist movement speed threshold (px/frame)
    "arm_above_shoulder": True, # Require arms raised above shoulder line
    "sustained_frames": 30,    # Must sustain for 1s before alert
}

# ============================================================
# CHILD DETECTION
# ============================================================
CHILD_THRESHOLDS = {
    "height_ratio": 0.55,     # Child bbox height < 55% of avg adult height
    "isolation_radius": 120,  # Pixels - no adult within this = "alone"
    "alert_after_seconds": 10, # Alert if alone for this long
}

# ============================================================
# RE-IDENTIFICATION (for target lock)
# ============================================================
REID_CONFIG = {
    "histogram_bins": 64,      # Color histogram resolution
    "match_threshold": 0.82,   # Min combined score for re-identify
    "min_similarity": 0.80,    # Strict appearance similarity gate
    "min_margin": 0.08,        # Best-vs-second-best score gap
    "grace_seconds": 1.2,      # Wait before switching to a new ID
    "search_radius": 120,      # Pixels around predicted position
    "max_lost_time": 10.0,     # Seconds before giving up search
}

# ============================================================
# DISPLAY SETTINGS
# ============================================================
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
DASHBOARD_WIDTH = 420
FPS_TARGET = 30

# ============================================================
# ALERT SETTINGS
# ============================================================
# Load from .env file if it exists, otherwise from environment variables
def _load_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

TELEGRAM_BOT_TOKEN = os.environ.get("GODS_EYE_TG_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("GODS_EYE_TG_CHAT", "")
ALERT_COOLDOWN = 30  # Seconds between same alerts

# ============================================================
# INPUT SOURCE
# ============================================================
# 0 = webcam, "Videos/file.mp4" = file, "http://..." = stream
VIDEO_SOURCE = "Videos/public.mp4"

# ============================================================
# DEMO MODE
# ============================================================
DEMO_ENABLED = False           # Toggle with 'D' key at runtime
DEMO_SCENARIO = 1              # 1=Normal, 2=Gathering, 3=Stampede