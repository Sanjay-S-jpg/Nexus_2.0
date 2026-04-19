"""
CrowdShield - Live Monitor Page
==================================
The main monitoring page with:
  - Real-time video feed with detection overlays
  - Video source selection (webcam, IP cam, file upload, file path)
  - Feature toggle controls
  - Real-time metrics (people count, FPS, alerts)
  - Alert feed
  - Target selection for tracking
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os

import config


def get_pipeline():
    """Get or create the processing pipeline (cached in session state)."""
    if "pipeline" not in st.session_state:
        with st.spinner("🔄 Loading CrowdShield AI models... (first time takes ~30 seconds)"):
            from core.pipeline import Pipeline
            st.session_state.pipeline = Pipeline()
    return st.session_state.pipeline


def get_video_source():
    """Get or create the video source."""
    if "video_source" not in st.session_state:
        from utils.video_source import VideoSource
        st.session_state.video_source = VideoSource()
    return st.session_state.video_source


def render_live_monitor():
    """Render the Live Monitor page."""
    
    st.markdown('<p class="main-title">🖥️ Live Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time AI-powered crowd safety monitoring</p>', unsafe_allow_html=True)
    
    # ============================================================
    # VIDEO SOURCE CONTROLS
    # ============================================================
    st.markdown("### 📹 Video Source")
    
    source_col1, source_col2 = st.columns([2, 1])
    
    with source_col1:
        source_type = st.radio(
            "Select Input Source",
            ["🎥 Webcam", "📱 IP Camera (Phone)", "📁 Upload Video", "📂 File Path"],
            horizontal=True,
            key="source_type"
        )
    
    video_source = get_video_source()
    source_value = None
    
    if source_type == "🎥 Webcam":
        cam_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0,
                                     help="0 = default webcam, 1 = second camera")
        source_value = int(cam_index)
    
    elif source_type == "📱 IP Camera (Phone)":
        st.markdown("""
        **How to connect your phone camera:**
        1. Install **IP Webcam** (Android) or **EpocCam** (iOS)
        2. Start the app on your phone
        3. Enter the URL shown in the app below
        """)
        ip_url = st.text_input(
            "IP Camera URL",
            value=config.IP_CAM_URL or "http://192.168.1.5:8080/video",
            placeholder="http://192.168.1.5:8080/video"
        )
        source_value = ip_url
    
    elif source_type == "📁 Upload Video":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="video_upload"
        )
        if uploaded_file is not None:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            source_value = tfile.name
            st.session_state.temp_video_path = tfile.name
    
    elif source_type == "📂 File Path":
        file_path = st.text_input(
            "Video File Path",
            value="",
            placeholder=r"C:\path\to\video.mp4 or relative path like data\sample.mp4"
        )
        if file_path:
            # Support relative paths
            if not os.path.isabs(file_path):
                file_path = os.path.join(config.BASE_DIR, file_path)
            source_value = file_path
    
    # ============================================================
    # START / STOP CONTROLS
    # ============================================================
    col_start, col_stop, col_reset = st.columns(3)
    
    with col_start:
        start_btn = st.button("▶️ Start Monitoring", type="primary", use_container_width=True)
    
    with col_stop:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
    
    with col_reset:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    if reset_btn:
        if "pipeline" in st.session_state:
            st.session_state.pipeline.reset()
        video_source.release()
        st.session_state.pipeline_running = False
        st.rerun()
    
    if stop_btn:
        video_source.release()
        st.session_state.pipeline_running = False
        st.rerun()
    
    # ============================================================
    # FEATURE TOGGLES  
    # ============================================================
    with st.expander("🎛️ Feature Controls", expanded=False):
        feat_cols = st.columns(4)
        
        features = {
            "stampede_detection": ("🏃 Stampede Detection", True),
            "weapon_detection": ("🔪 Weapon Detection", True),
            "fight_detection": ("🥊 Fight Detection", True),
            "child_detection": ("👶 Child Detection", False),
            "crowd_analysis": ("👥 Crowd Analysis", True),
            "heatmap": ("🗺️ Heatmap", True),
            "night_vision": ("🌙 Night Vision (Auto)", True),
            "tracking": ("🎯 Person Tracking", True),
        }
        
        for i, (key, (label, default)) in enumerate(features.items()):
            # Use session state to preserve checkbox values across reruns
            state_key = f"feat_{key}"
            if state_key not in st.session_state:
                st.session_state[state_key] = default
            with feat_cols[i % 4]:
                enabled = st.checkbox(label, key=state_key)
                if "pipeline" in st.session_state:
                    st.session_state.pipeline.set_feature(key, enabled)
    
    # ============================================================
    # TARGET TRACKING — select a person by track ID
    # ============================================================
    with st.expander("🎯 Target Tracking", expanded=False):
        st.markdown("**Track a specific person**: Enter the `#ID` shown on a person's dot to follow them.")
        tt_cols = st.columns([2, 1, 1])
        with tt_cols[0]:
            target_id_input = st.number_input(
                "Person Track ID", min_value=0, max_value=9999, value=0,
                step=1, key="target_track_id_input",
                help="Enter the # number shown above a person's dot"
            )
        with tt_cols[1]:
            track_btn = st.button("🎯 Track", use_container_width=True)
        with tt_cols[2]:
            clear_track_btn = st.button("❌ Stop Tracking", use_container_width=True)
        
        if track_btn and target_id_input > 0 and "pipeline" in st.session_state:
            # Find the track with this ID and set it as target
            pipeline_ref = st.session_state.pipeline
            tracks_list = getattr(pipeline_ref, '_last_tracks', [])
            target_set = False
            for t in tracks_list:
                if t.track_id == target_id_input and t.lost_frames == 0:
                    frame_for_target = getattr(pipeline_ref, '_last_frame', None)
                    if frame_for_target is not None:
                        pipeline_ref.select_target(frame_for_target, t.bbox, t.track_id)
                        st.success(f"🎯 Now tracking person #{target_id_input}")
                        target_set = True
                    break
            if not target_set:
                st.warning(f"⚠️ Person #{target_id_input} not found. Make sure the ID is visible on screen.")
        
        if clear_track_btn and "pipeline" in st.session_state:
            st.session_state.pipeline.clear_target()
            st.info("Target tracking stopped.")
    
    # ============================================================
    # DISPLAY OPTIONS
    # ============================================================
    with st.expander("🎨 Display Options", expanded=False):
        disp_cols = st.columns(4)
        
        draw_opts = {
            "show_boxes": ("📦 Bounding Boxes", True),
            "show_ids": ("🏷️ Track IDs", True),
            "show_trails": ("〰️ Movement Trails", False),
            "show_heatmap": ("🗺️ Heatmap Overlay", False),
            "show_info_panel": ("📊 Info Panel", True),
            "show_optical_flow": ("🌊 Optical Flow", False),
        }
        
        for i, (key, (label, default)) in enumerate(draw_opts.items()):
            state_key = f"draw_{key}"
            if state_key not in st.session_state:
                st.session_state[state_key] = default
            with disp_cols[i % 4]:
                enabled = st.checkbox(label, key=state_key)
                if "pipeline" in st.session_state:
                    st.session_state.pipeline.set_draw_option(key, enabled)
    
    # ============================================================
    # MAIN VIDEO FEED + METRICS
    # ============================================================
    if start_btn and source_value is not None:
        st.session_state.pipeline_running = True
        st.session_state.source_value = source_value
    
    if st.session_state.get("pipeline_running", False):
        pipeline = get_pipeline()
        sv = st.session_state.get("source_value")
        
        if sv is not None and not video_source.is_open:
            success = video_source.open(sv)
            if not success:
                st.error(f"❌ Failed to open video source: {sv}")
                st.session_state.pipeline_running = False
                return
        
        # Layout: Video on left (2/3), Metrics on right (1/3)
        video_col, metrics_col = st.columns([2, 1])
        
        with video_col:
            st.markdown("### 📺 Video Feed")
            frame_placeholder = st.empty()
        
        with metrics_col:
            st.markdown("### 📊 Live Metrics")
            metrics_placeholder = st.empty()
            
            st.markdown("### 🚨 Alert Feed")
            alert_placeholder = st.empty()
        
        # Video processing loop
        while st.session_state.get("pipeline_running", False):
            frame = video_source.read()
            if frame is None:
                st.warning("⚠️ No frame received. Check your video source.")
                time.sleep(0.5)
                continue
            
            # Process through pipeline
            results = pipeline.process_frame(frame)
            
            # Update session state for sidebar
            st.session_state.current_fps = results["fps"]
            st.session_state.people_count = results["people_count"]
            
            # Display annotated frame
            annotated = results["annotated_frame"]
            if annotated is not None:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Display metrics
            with metrics_placeholder.container():
                m1, m2 = st.columns(2)
                m1.metric("👥 People", results["people_count"])
                m2.metric("⚡ FPS", f"{results['fps']:.1f}")
                
                m3, m4 = st.columns(2)
                crowd = results.get("crowd_results", {})
                m3.metric("📈 Density", crowd.get("density", "LOW"))
                m4.metric("📊 Trend", crowd.get("trend", "stable"))
                
                # Threat status
                stampede_sev = results.get("stampede_results", {}).get("severity", "NONE")
                weapon_threat = results.get("weapon_results", {}).get("threat_level", "NONE") 
                fight_sev = results.get("fight_results", {}).get("severity", "NONE")
                
                if stampede_sev == "CRITICAL" or weapon_threat == "HIGH":
                    st.markdown('<p class="status-danger">🔴 CRITICAL THREAT</p>', unsafe_allow_html=True)
                elif stampede_sev == "WARNING" or fight_sev != "NONE":
                    st.markdown('<p class="status-warning">🟡 WARNING</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="status-safe">🟢 ALL CLEAR</p>', unsafe_allow_html=True)
                
                # Night mode indicator
                if results.get("is_night_mode"):
                    st.info(f"🌙 Night Vision Active (Brightness: {results['brightness']:.0f})")
                
                # Child detection
                child_results = results.get("child_results", {})
                if child_results.get("total_children", 0) > 0:
                    st.warning(f"👶 Children detected: {child_results['total_children']} "
                              f"(alone: {len(child_results.get('alone_children', []))})")
            
            # Display recent alerts
            with alert_placeholder.container():
                recent_alerts = pipeline.alert_manager.get_recent(5)
                if recent_alerts:
                    for alert in recent_alerts:
                        severity = alert.get("severity", "LOW").lower()
                        st.markdown(
                            f'<div class="alert-{severity}">'
                            f'<strong>[{alert.get("severity")}]</strong> '
                            f'{alert.get("alert_type", "").upper()}: '
                            f'{alert.get("message", "")}<br>'
                            f'<small>{alert.get("datetime", "")}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("*No alerts yet*")
            
            # Small delay to prevent overloading Streamlit
            time.sleep(0.03)
    
    else:
        # Not running - show instructions
        st.markdown("---")
        st.info(
            "👆 **Select a video source above and click 'Start Monitoring'** to begin.\n\n"
            "**Quick Start:**\n"
            "1. Select '🎥 Webcam' for your laptop camera\n"
            "2. Select '📁 Upload Video' to test with a video file\n"
            "3. Select '📱 IP Camera' to stream from your phone\n"
            "4. Click ▶️ Start Monitoring!"
        )
        
        # Quick stats from previous session
        if "pipeline" in st.session_state:
            stats = st.session_state.pipeline.alert_manager.get_statistics()
            if stats["total"] > 0:
                st.markdown("### 📊 Previous Session Stats")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Alerts", stats["total"])
                col2.metric("Last 24h", stats["last_24h"])
                
                by_type = stats.get("by_type", {})
                col3.metric("Stampede Alerts", by_type.get("stampede", 0))
                col4.metric("Weapon Alerts", by_type.get("weapon", 0))
