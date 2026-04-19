# 🛡️ CrowdShield - AI-Powered Public Safety & Stampede Detection System

> Final Year Project | Real-time crowd monitoring with stampede detection, weapon detection, fight detection, child safety, and more.

---

## 🚀 Quick Setup Guide (Step-by-Step for Beginners)

### Step 1: Install Python
1. Download Python 3.10+ from https://www.python.org/downloads/
2. During installation, **CHECK** "Add Python to PATH" ✅
3. Verify: Open Command Prompt and type `python --version`

### Step 2: Install CUDA (for GPU acceleration with your RTX 4050)
1. Download CUDA Toolkit 11.8+ from https://developer.nvidia.com/cuda-toolkit
2. Install it (default settings are fine)
3. Verify: Open Command Prompt and type `nvidia-smi`

### Step 3: Create Project Environment
Open Command Prompt/PowerShell in the project folder and run:

```bash
# Create a virtual environment (keeps project packages separate)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# Install PyTorch with CUDA support FIRST (for your RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

### Step 4: Run CrowdShield!
```bash
# Make sure virtual environment is activated, then:
streamlit run app.py
```

This will open a browser window at `http://localhost:8501` with the CrowdShield dashboard.

---

## 📱 Connecting Your Phone Camera

### Android (IP Webcam):
1. Install "IP Webcam" from Google Play Store
2. Open the app → Start server
3. Note the URL shown (e.g., `http://192.168.1.5:8080`)
4. In CrowdShield, select "📱 IP Camera" and enter: `http://192.168.1.5:8080/video`

### iPhone (EpocCam / IP Camera):
1. Install "IP Camera" from App Store
2. Start the camera
3. Enter the URL shown in CrowdShield

> **Important:** Your phone and laptop must be on the **same WiFi network**!

---

## 📂 Project Structure

```
CrowdShield/
├── app.py                  # Main entry point (Streamlit app)
├── config.py               # All configuration settings
├── requirements.txt        # Python dependencies
├── crowdshield.db          # Alert database (auto-created)
│
├── core/                   # AI detection modules
│   ├── detector.py         # YOLOv8 object detection
│   ├── tracker.py          # Multi-object tracking
│   ├── stampede.py         # Stampede detection (optical flow)
│   ├── weapon_detector.py  # Weapon detection & filtering
│   ├── fight_detector.py   # Fight detection (pose-based)
│   ├── crowd_analyzer.py   # Crowd counting & density
│   ├── heatmap.py          # Heatmap generation
│   ├── night_vision.py     # Low-light enhancement
│   ├── child_detector.py   # Lost child detection
│   ├── target_tracker.py   # Target tracking & re-ID
│   └── pipeline.py         # Main processing pipeline
│
├── pages/                  # Streamlit UI pages
│   ├── live_monitor.py     # Live video monitoring
│   ├── reports.py          # Alert reports & analytics
│   ├── heatmap_view.py     # Heatmap visualization
│   └── settings.py         # Configuration UI
│
├── utils/                  # Utility modules
│   ├── video_source.py     # Video input handling
│   ├── alert_system.py     # Alerts & database
│   └── drawing.py          # Visual annotation helpers
│
├── models/                 # AI model files (auto-downloaded)
├── data/                   # Sample test videos
└── reports/                # Generated reports & snapshots
```

---

## 🎯 Features

| Feature | Method | Status |
|---------|--------|--------|
| 👥 People Detection & Counting | YOLOv8 | ✅ |
| 🏃 Stampede Detection | Optical Flow + Density Analysis | ✅ |
| 🔪 Weapon Detection | YOLOv8 (knife/scissors) + Context Filtering | ✅ |
| 🥊 Fight Detection | YOLOv8-Pose + Motion Analysis | ✅ |
| 👶 Lost Child Detection | Height Ratio + Isolation Check | ✅ |
| 🗺️ Heatmap | Gaussian Accumulation | ✅ |
| 🌙 Night Vision | CLAHE Auto-Enhancement | ✅ |
| 🎯 Target Tracking | Color Histogram Re-ID | ✅ |
| 📊 Reports & Analytics | SQLite + Charts | ✅ |
| 📹 Multi-Source Video | Webcam + IP Cam + Files | ✅ |

---

## 🔧 Tuning for Your Demo

### If getting too many false positives:
- Increase confidence thresholds in Settings page
- Increase consecutive frames required before alerts
- Increase cooldown periods

### If missing detections:
- Lower confidence thresholds
- Use a larger YOLO model (change `YOLO_MODEL_SIZE` in config.py to "l")
- Ensure good lighting or enable Night Vision

### If FPS is too low:
- Use a smaller YOLO model ("s" or "n")
- Reduce frame resolution in Settings
- Ensure FP16 is enabled (requires RTX GPU)
- Disable features you don't need for the demo

---

## 🎓 For Your Project Report

### Technologies Used:
- **YOLOv8** (Ultralytics) - Object detection & pose estimation
- **OpenCV** - Video processing, optical flow, image enhancement
- **Streamlit** - Web dashboard
- **SQLite** - Alert database
- **Python** - Programming language

### Key Algorithms:
1. **Stampede Detection**: Farneback Dense Optical Flow + Direction Uniformity Analysis
2. **Fight Detection**: Pose Keypoint Velocity Analysis (wrist/elbow speed)
3. **Child Detection**: Height Ratio Comparison + Spatial Isolation Check
4. **Night Vision**: Contrast Limited Adaptive Histogram Equalization (CLAHE)
5. **Target Re-ID**: HSV Color Histogram Correlation Matching
6. **Tracking**: IoU-based Greedy Matching (simplified ByteTrack)

---

## 📝 Sample Test Videos

Download crowd/stampede videos for testing from:
- https://www.pexels.com/search/videos/crowd/
- https://www.youtube.com/results?search_query=crowd+stampede+cctv
- Place them in the `data/` folder and select "File Path" in the app

---

*Built with ❤️ for public safety*
