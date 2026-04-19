"""
CrowdShield - Settings Page
===============================
Configuration interface for adjusting:
  - Detection thresholds
  - Alert settings
  - Video settings
  - Night vision settings
  - Performance settings
"""

import streamlit as st
import config


def render_settings():
    """Render the Settings page."""
    
    st.markdown('<p class="main-title">⚙️ Settings</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Configure CrowdShield detection parameters</p>', unsafe_allow_html=True)
    
    st.info(
        "💡 **Tip:** Changes here take effect immediately on the running pipeline. "
        "To make changes permanent, update the values in `config.py`."
    )
    
    # Get pipeline reference
    pipeline = st.session_state.get("pipeline")
    
    # ============================================================
    # DETECTION THRESHOLDS
    # ============================================================
    st.markdown("### 🎯 Detection Confidence Thresholds")
    st.markdown(
        "Lower values = more detections but more false positives. "
        "Higher values = fewer detections but more reliable."
    )
    
    thresh_col1, thresh_col2, thresh_col3 = st.columns(3)
    
    with thresh_col1:
        person_conf = st.slider(
            "Person Confidence",
            0.1, 0.9, config.PERSON_CONFIDENCE,
            step=0.05,
            help="Minimum confidence to detect a person"
        )
    
    with thresh_col2:
        weapon_conf = st.slider(
            "Weapon Confidence",
            0.1, 0.9, config.WEAPON_CONFIDENCE,
            step=0.05,
            help="Minimum confidence to detect a weapon"
        )
    
    with thresh_col3:
        pose_conf = st.slider(
            "Pose Confidence",
            0.1, 0.9, config.POSE_CONFIDENCE,
            step=0.05,
            help="Minimum confidence for pose keypoints"
        )
    
    # Apply to config (runtime only)
    config.PERSON_CONFIDENCE = person_conf
    config.WEAPON_CONFIDENCE = weapon_conf
    config.POSE_CONFIDENCE = pose_conf
    
    # ============================================================
    # STAMPEDE SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🏃 Stampede Detection Settings")
    
    stmp_col1, stmp_col2 = st.columns(2)
    
    with stmp_col1:
        config.STAMPEDE_FLOW_THRESHOLD = st.slider(
            "Flow Threshold",
            1.0, 10.0, config.STAMPEDE_FLOW_THRESHOLD,
            step=0.5,
            help="Minimum optical flow magnitude to flag movement"
        )
        
        config.STAMPEDE_UNIFORMITY_THRESH = st.slider(
            "Uniformity Threshold",
            0.2, 0.9, config.STAMPEDE_UNIFORMITY_THRESH,
            step=0.05,
            help="How aligned movement must be (0=random, 1=perfect unison)"
        )
    
    with stmp_col2:
        config.STAMPEDE_CONSEC_FRAMES = st.slider(
            "Consecutive Frames",
            3, 20, config.STAMPEDE_CONSEC_FRAMES,
            help="Frames of stampede-like motion before alerting"
        )
        
        config.STAMPEDE_MIN_PEOPLE = st.slider(
            "Min People for Stampede Check",
            2, 20, config.STAMPEDE_MIN_PEOPLE,
            help="Need at least this many people to check for stampede"
        )
    
    # ============================================================
    # FIGHT DETECTION SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🥊 Fight Detection Settings")
    
    fight_col1, fight_col2 = st.columns(2)
    
    with fight_col1:
        config.FIGHT_PROXIMITY_PX = st.slider(
            "Proximity (pixels)",
            50, 300, config.FIGHT_PROXIMITY_PX,
            step=10,
            help="Max distance between two people to check for fighting"
        )
        
        config.FIGHT_VELOCITY_THRESH = st.slider(
            "Velocity Threshold",
            5.0, 50.0, config.FIGHT_VELOCITY_THRESH,
            step=1.0,
            help="Wrist/elbow speed threshold for aggressive motion"
        )
    
    with fight_col2:
        config.FIGHT_CONSEC_FRAMES = st.slider(
            "Consecutive Frames (Fight)",
            3, 15, config.FIGHT_CONSEC_FRAMES,
            help="Frames of fighting before alert"
        )
        
        config.FIGHT_COOLDOWN_SEC = st.slider(
            "Cooldown (seconds)",
            5, 60, config.FIGHT_COOLDOWN_SEC,
            help="Seconds between repeat fight alerts"
        )
    
    # ============================================================
    # CHILD DETECTION SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### 👶 Child Detection Settings")
    
    child_col1, child_col2 = st.columns(2)
    
    with child_col1:
        config.CHILD_HEIGHT_RATIO = st.slider(
            "Height Ratio (Child vs Adult)",
            0.3, 0.7, config.CHILD_HEIGHT_RATIO,
            step=0.05,
            help="If person height < this ratio of average adult = child. "
                 "Lower = only very short people detected as children"
        )
        
        config.CHILD_ISOLATION_DIST_PX = st.slider(
            "Isolation Distance (pixels)",
            50, 500, config.CHILD_ISOLATION_DIST_PX,
            step=10,
            help="If no adult within this distance = child is alone"
        )
    
    with child_col2:
        config.CHILD_ALONE_FRAMES = st.slider(
            "Alone Duration (frames)",
            10, 90, config.CHILD_ALONE_FRAMES,
            help="Frames alone before alerting (~30 frames = 1 second)"
        )
        
        config.CHILD_COOLDOWN_SEC = st.slider(
            "Cooldown (seconds)",
            5, 60, config.CHILD_COOLDOWN_SEC,
            help="Seconds between repeat child alerts"
        )
    
    # ============================================================
    # NIGHT VISION SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🌙 Night Vision Settings")
    
    nv_col1, nv_col2 = st.columns(2)
    
    with nv_col1:
        config.NIGHT_VISION_AUTO = st.checkbox(
            "Auto Night Vision",
            value=config.NIGHT_VISION_AUTO,
            help="Automatically detect low light and enhance"
        )
        
        config.NIGHT_VISION_BRIGHTNESS_THRESH = st.slider(
            "Brightness Threshold",
            10, 150, config.NIGHT_VISION_BRIGHTNESS_THRESH,
            help="Below this average brightness = dark (0-255)"
        )
    
    with nv_col2:
        config.NIGHT_VISION_CLIP_LIMIT = st.slider(
            "Enhancement Strength",
            1.0, 10.0, config.NIGHT_VISION_CLIP_LIMIT,
            step=0.5,
            help="CLAHE clip limit (higher = more contrast enhancement)"
        )
        
        force_nv = st.checkbox(
            "Force Night Vision ON",
            value=False,
            help="Enable night vision regardless of brightness"
        )
        if pipeline and hasattr(pipeline, 'night_vision'):
            pipeline.night_vision.set_force_enabled(force_nv)
    
    # ============================================================
    # HEATMAP SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🗺️ Heatmap Settings")
    
    hm_col1, hm_col2 = st.columns(2)
    
    with hm_col1:
        config.HEATMAP_DECAY = st.slider(
            "Decay Rate",
            0.95, 0.999, config.HEATMAP_DECAY,
            format="%.3f",
            step=0.001,
            help="How fast old positions fade (higher = slower fade)"
        )
    
    with hm_col2:
        config.HEATMAP_RADIUS = st.slider(
            "Heat Blob Radius",
            10, 60, config.HEATMAP_RADIUS,
            help="Size of each person's heat contribution"
        )
    
    if pipeline and pipeline.heatmap_gen:
        pipeline.heatmap_gen.decay = config.HEATMAP_DECAY
        pipeline.heatmap_gen.radius = config.HEATMAP_RADIUS
    
    # ============================================================
    # PERFORMANCE SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### ⚡ Performance Settings")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        config.USE_HALF_PRECISION = st.checkbox(
            "FP16 Half Precision (GPU)",
            value=config.USE_HALF_PRECISION,
            help="Uses less GPU memory and is faster. Requires RTX GPU."
        )
        
        config.MAX_FRAME_WIDTH = st.slider(
            "Max Frame Width",
            640, 1920, config.MAX_FRAME_WIDTH,
            step=64,
            help="Frames wider than this are resized down"
        )
    
    with perf_col2:
        config.MAX_FRAME_HEIGHT = st.slider(
            "Max Frame Height",
            360, 1080, config.MAX_FRAME_HEIGHT,
            step=64,
            help="Frames taller than this are resized down"
        )
        
        model_size = st.selectbox(
            "YOLO Model Size",
            ["n (Nano - Fastest)", "s (Small)", "m (Medium - Recommended)", "l (Large)", "x (XLarge - Most Accurate)"],
            index=2,
            help="Larger models are more accurate but slower. Restart required."
        )
        
        st.warning(
            "⚠️ Changing model size requires restarting the application. "
            "Set the YOLO_MODEL_SIZE in config.py and restart."
        )
    
    # ============================================================
    # ALERT SETTINGS
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔔 Alert Settings")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        config.ALERT_SOUND_ENABLED = st.checkbox(
            "Sound Alerts",
            value=config.ALERT_SOUND_ENABLED,
            help="Play a beep sound when critical alerts occur (Windows)"
        )
        
        config.ALERT_COOLDOWN_DEFAULT = st.slider(
            "Default Alert Cooldown (seconds)",
            5, 60, config.ALERT_COOLDOWN_DEFAULT,
            help="Minimum seconds between same-type alerts"
        )
    
    with alert_col2:
        st.markdown("**Alert Severity Mapping:**")
        for alert_type, severity in config.ALERT_SEVERITY.items():
            st.text(f"  {alert_type}: {severity}")
    
    # ============================================================
    # SYSTEM INFO
    # ============================================================
    st.markdown("---")
    st.markdown("### 💻 System Information")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if gpu_available else "N/A"
    except:
        gpu_available = False
        gpu_name = "Unable to detect"
        gpu_memory = "N/A"
    
    sys_col1, sys_col2 = st.columns(2)
    
    with sys_col1:
        st.markdown(f"**GPU Available:** {'✅ Yes' if gpu_available else '❌ No (using CPU)'}")
        st.markdown(f"**GPU Name:** {gpu_name}")
        st.markdown(f"**GPU Memory:** {gpu_memory}")
    
    with sys_col2:
        st.markdown(f"**Model Size:** {config.YOLO_MODEL_SIZE}")
        st.markdown(f"**Detection Model:** {config.YOLO_DETECT_MODEL}")
        st.markdown(f"**Pose Model:** {config.YOLO_POSE_MODEL}")
        st.markdown(f"**Database:** {config.DB_PATH}")
