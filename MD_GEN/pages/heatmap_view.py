"""
CrowdShield - Heatmap View Page
==================================
Standalone heatmap analysis page with:
  - Full-screen heatmap display
  - Hotspot identification
  - Historical heatmap playback
  - Export options
"""

import streamlit as st
import cv2
import numpy as np

import config


def render_heatmap_view():
    """Render the Heatmap View page."""
    
    st.markdown('<p class="main-title">🗺️ Heatmap View</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Crowd density visualization and hotspot analysis</p>', unsafe_allow_html=True)
    
    # Check if pipeline has heatmap data
    pipeline = st.session_state.get("pipeline")
    
    if pipeline is None or pipeline.heatmap_gen is None:
        st.warning(
            "No heatmap data available yet. Start monitoring from the **Live Monitor** page first, "
            "and the heatmap will build up over time."
        )
        
        # Show a demo/placeholder heatmap
        st.markdown("### 📖 How the Heatmap Works")
        st.markdown("""
        The heatmap shows **where people spend the most time**:
        
        - 🔵 **Blue** = Few or no people (cool area)  
        - 🟢 **Green** = Some foot traffic
        - 🟡 **Yellow** = Moderate activity
        - 🔴 **Red** = Heavy activity / hotspot
        
        **Use cases for security:**
        - Identify **bottleneck areas** where stampedes are likely
        - Find **popular gathering spots** that need extra attention
        - Track **crowd flow patterns** over time
        - Plan **emergency exit routes** based on traffic
        """)
        
        # Generate a demo heatmap
        st.markdown("### 🎨 Demo Heatmap")
        demo_heatmap = _generate_demo_heatmap()
        st.image(demo_heatmap, caption="Demo heatmap visualization", use_container_width=True)
        return
    
    # ============================================================
    # HEATMAP CONTROLS
    # ============================================================
    st.markdown("### 🎛️ Heatmap Controls")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    
    with ctrl_col1:
        heatmap_opacity = st.slider(
            "Overlay Opacity",
            0.0, 1.0, 0.5,
            help="Transparency of the heatmap overlay on the video"
        )
    
    with ctrl_col2:
        decay_rate = st.slider(
            "Decay Rate",
            0.95, 0.999, config.HEATMAP_DECAY,
            format="%.3f",
            help="How fast old positions fade. Higher = slower decay"
        )
        if pipeline.heatmap_gen:
            pipeline.heatmap_gen.decay = decay_rate
    
    with ctrl_col3:
        colormap_options = {
            "JET (Rainbow)": 11,
            "HOT (White-Red)": 2,
            "INFERNO": 1,
            "BONE": 3,
            "OCEAN": 5,
        }
        colormap_name = st.selectbox("Color Scheme", list(colormap_options.keys()))
        if pipeline.heatmap_gen:
            pipeline.heatmap_gen.colormap = colormap_options[colormap_name]
    
    # ============================================================
    # HEATMAP DISPLAY
    # ============================================================
    st.markdown("---")
    
    heatmap_col, info_col = st.columns([2, 1])
    
    with heatmap_col:
        st.markdown("### 🗺️ Current Heatmap")
        
        if pipeline.heatmap_gen and pipeline.heatmap_gen.frame_count > 0:
            heatmap_img = pipeline.heatmap_gen.get_heatmap_image()
            
            if heatmap_img is not None:
                # Convert BGR to RGB
                heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, caption="Crowd Heatmap", use_container_width=True)
                
                # Save button
                if st.button("💾 Save Heatmap Image"):
                    save_path = os.path.join(config.REPORTS_DIR, "heatmap_export.png")
                    cv2.imwrite(save_path, heatmap_img)
                    st.success(f"Saved to: {save_path}")
            else:
                st.info("Heatmap is still building up. Keep monitoring for a few seconds.")
        else:
            st.info("No heatmap data yet. Run the monitor to build the heatmap.")
    
    with info_col:
        st.markdown("### 🔥 Hotspots")
        
        if pipeline.heatmap_gen and pipeline.heatmap_gen.frame_count > 0:
            hotspots = pipeline.heatmap_gen.get_hotspots(top_n=5)
            
            if hotspots:
                for i, (x, y, intensity) in enumerate(hotspots):
                    norm_intensity = intensity / max(h[2] for h in hotspots) if hotspots else 0
                    
                    if norm_intensity > 0.7:
                        emoji = "🔴"
                    elif norm_intensity > 0.4:
                        emoji = "🟡"
                    else:
                        emoji = "🟢"
                    
                    st.markdown(
                        f"{emoji} **Hotspot #{i+1}**: Position ({x}, {y}) | "
                        f"Intensity: {norm_intensity:.0%}"
                    )
            else:
                st.info("No significant hotspots detected yet.")
        else:
            st.info("Start monitoring to see hotspots.")
        
        st.markdown("---")
        st.markdown("### 📊 Heatmap Stats")
        
        if pipeline.heatmap_gen:
            st.metric("Frames Processed", pipeline.heatmap_gen.frame_count)
            
            max_heat = pipeline.heatmap_gen.heat_map.max()
            st.metric("Max Heat Value", f"{max_heat:.1f}")
            
            non_zero = np.count_nonzero(pipeline.heatmap_gen.heat_map > 0.1)
            total_pixels = pipeline.heatmap_gen.heat_map.size
            coverage = non_zero / total_pixels * 100
            st.metric("Coverage", f"{coverage:.1f}%",
                      help="Percentage of the frame that has any heat")
    
    # ============================================================
    # REGION DENSITY
    # ============================================================
    if pipeline and hasattr(pipeline, 'crowd_analyzer'):
        st.markdown("---")
        st.markdown("### 📊 Region Density Grid")
        st.markdown("People count in each quadrant of the frame:")
        
        region_counts = pipeline.crowd_analyzer.region_counts
        
        if region_counts is not None and region_counts.sum() > 0:
            rows, cols = region_counts.shape
            
            for r in range(rows):
                grid_cols = st.columns(cols)
                for c in range(cols):
                    count = int(region_counts[r, c])
                    if count > 5:
                        color = "🔴"
                    elif count > 2:
                        color = "🟡"
                    elif count > 0:
                        color = "🟢"
                    else:
                        color = "⬜"
                    
                    with grid_cols[c]:
                        st.markdown(f"{color} **{count}** people")
        else:
            st.info("Region data will appear once monitoring starts.")


def _generate_demo_heatmap():
    """Generate a demo heatmap for the info page."""
    # Create a sample heatmap
    h, w = 400, 600
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Add some demo hot spots
    spots = [
        (300, 200, 50, 1.0),  # Center
        (150, 150, 40, 0.7),  # Top-left-ish
        (450, 300, 35, 0.5),  # Bottom-right-ish
        (100, 350, 30, 0.4),  # Bottom-left
    ]
    
    for sx, sy, radius, intensity in spots:
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist_sq = (x_coords - sx) ** 2 + (y_coords - sy) ** 2
        sigma = radius / 2.0
        gaussian = intensity * np.exp(-dist_sq / (2 * sigma ** 2))
        heatmap += gaussian
    
    # Normalize and colorize
    normalized = (heatmap / heatmap.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored_rgb


# Need os for save path
import os
