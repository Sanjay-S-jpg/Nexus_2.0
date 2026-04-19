"""
CrowdShield - Main Streamlit Application
==========================================
This is the entry point for the CrowdShield dashboard.
Run with: streamlit run app.py

Multi-page app with:
  - Live Monitor: Real-time video feed with all detections
  - Reports: Alert history and analytics
  - Heatmap View: Standalone heatmap analysis
  - Settings: Configure detection parameters
"""

import streamlit as st

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="CrowdShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS for better styling
# ============================================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #888;
        margin-top: -10px;
    }
    
    /* Alert cards */
    .alert-critical {
        background-color: rgba(255, 0, 0, 0.15);
        border-left: 4px solid #FF0000;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    .alert-high {
        background-color: rgba(255, 102, 0, 0.15);
        border-left: 4px solid #FF6600;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    .alert-medium {
        background-color: rgba(255, 215, 0, 0.15);
        border-left: 4px solid #FFD700;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    .alert-low {
        background-color: rgba(68, 136, 255, 0.15);
        border-left: 4px solid #4488FF;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23 0%, #2d3139 100%);
        border-radius: 10px;
        padding: 15px 20px;
        border: 1px solid #333;
    }
    
    /* Status indicators */
    .status-safe { color: #00CC00; font-weight: bold; }
    .status-warning { color: #FFD700; font-weight: bold; }
    .status-danger { color: #FF0000; font-weight: bold; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    
    /* Video frame styling */
    .stImage img {
        border-radius: 8px;
        border: 2px solid #333;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("# 🛡️ CrowdShield")
    st.markdown("*AI Public Safety System*")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🖥️ Live Monitor", "📊 Reports", "🗺️ Heatmap View", "⚙️ Settings"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    
    # Show system status in sidebar
    if "pipeline_running" in st.session_state and st.session_state.pipeline_running:
        st.markdown('🟢 **System Active**')
    else:
        st.markdown('🔴 **System Idle**')
    
    if "current_fps" in st.session_state:
        st.metric("FPS", f"{st.session_state.get('current_fps', 0):.1f}")
    
    if "people_count" in st.session_state:
        st.metric("People", st.session_state.get("people_count", 0))


# ============================================================
# PAGE ROUTING
# ============================================================
if page == "🖥️ Live Monitor":
    from pages.live_monitor import render_live_monitor
    render_live_monitor()

elif page == "📊 Reports":
    from pages.reports import render_reports
    render_reports()

elif page == "🗺️ Heatmap View":
    from pages.heatmap_view import render_heatmap_view
    render_heatmap_view()

elif page == "⚙️ Settings":
    from pages.settings import render_settings
    render_settings()
