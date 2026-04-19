"""
CrowdShield - Reports Page
=============================
Security report page with:
  - Alert history table (filterable by type, severity, time)
  - Alert statistics charts
  - Alert details with saved frame snapshots
  - Export options
"""

import streamlit as st
import pandas as pd
import datetime
import os
import config
from utils.alert_system import AlertDatabase


def render_reports():
    """Render the Reports page."""
    
    st.markdown('<p class="main-title">📊 Reports & Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Alert history, statistics, and security analysis</p>', unsafe_allow_html=True)
    
    # Initialize database
    db = AlertDatabase()
    
    # ============================================================
    # STATISTICS OVERVIEW
    # ============================================================
    st.markdown("### 📈 Statistics Overview")
    
    stats = db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Alerts", stats["total"],
                help="Total alerts recorded since first use")
    col2.metric("Last 24 Hours", stats["last_24h"],
                help="Alerts in the last 24 hours")
    
    critical_count = stats.get("by_severity", {}).get("CRITICAL", 0)
    high_count = stats.get("by_severity", {}).get("HIGH", 0)
    col3.metric("Critical Alerts", critical_count)
    col4.metric("High Priority", high_count)
    
    # ============================================================
    # CHARTS
    # ============================================================
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Alerts by Type")
        by_type = stats.get("by_type", {})
        if by_type:
            type_df = pd.DataFrame({
                "Alert Type": list(by_type.keys()),
                "Count": list(by_type.values())
            })
            st.bar_chart(type_df.set_index("Alert Type"))
        else:
            st.info("No alert data yet. Start monitoring to generate alerts.")
    
    with chart_col2:
        st.markdown("#### Alerts by Severity")
        by_severity = stats.get("by_severity", {})
        if by_severity:
            sev_df = pd.DataFrame({
                "Severity": list(by_severity.keys()),
                "Count": list(by_severity.values())
            })
            st.bar_chart(sev_df.set_index("Severity"))
        else:
            st.info("No alert data yet.")
    
    # ============================================================
    # ALERT HISTORY TABLE
    # ============================================================
    st.markdown("---")
    st.markdown("### 📋 Alert History")
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        type_filter = st.multiselect(
            "Filter by Type",
            ["stampede", "weapon", "fight", "lost_child", "crowd_surge", "crowd_high"],
            default=[],
            key="type_filter"
        )
    
    with filter_col2:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=[],
            key="severity_filter"
        )
    
    with filter_col3:
        limit = st.slider("Max Results", 10, 500, 100, key="result_limit")
    
    # Fetch alerts
    alerts = db.get_recent_alerts(limit=limit)
    
    # Apply filters
    if type_filter:
        alerts = [a for a in alerts if a.get("alert_type") in type_filter]
    if severity_filter:
        alerts = [a for a in alerts if a.get("severity") in severity_filter]
    
    if alerts:
        # Convert to DataFrame for nice display
        df = pd.DataFrame(alerts)
        
        # Format columns
        display_cols = ["id", "datetime_str", "alert_type", "severity", "message"]
        available_cols = [c for c in display_cols if c in df.columns]
        
        if available_cols:
            display_df = df[available_cols].copy()
            display_df.columns = ["ID", "Time", "Type", "Severity", "Message"][:len(available_cols)]
            
            # Color-code severity
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # ============================================================
        # ALERT DETAIL VIEW
        # ============================================================
        st.markdown("### 🔍 Alert Details")
        
        selected_id = st.selectbox(
            "Select an alert to view details",
            [a["id"] for a in alerts],
            format_func=lambda x: f"Alert #{x} - {next((a['alert_type'] for a in alerts if a['id'] == x), '?')} ({next((a['datetime_str'] for a in alerts if a['id'] == x), '')})"
        )
        
        if selected_id:
            selected = next((a for a in alerts if a["id"] == selected_id), None)
            if selected:
                detail_col1, detail_col2 = st.columns([1, 1])
                
                with detail_col1:
                    st.markdown(f"**Type:** {selected.get('alert_type', 'N/A')}")
                    st.markdown(f"**Severity:** {selected.get('severity', 'N/A')}")
                    st.markdown(f"**Time:** {selected.get('datetime_str', 'N/A')}")
                    st.markdown(f"**Message:** {selected.get('message', 'N/A')}")
                    
                    if selected.get("data"):
                        with st.expander("Raw Data"):
                            st.code(selected["data"])
                
                with detail_col2:
                    # Show snapshot if available
                    frame_path = selected.get("frame_path")
                    if frame_path and os.path.exists(frame_path):
                        st.image(frame_path, caption="Frame at time of alert")
                    else:
                        st.info("No snapshot available for this alert.")
    else:
        st.info("No alerts matching your filters. Start monitoring to generate alerts!")
    
    # ============================================================
    # EXPORT & MANAGEMENT
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔧 Management")
    
    mgmt_col1, mgmt_col2 = st.columns(2)
    
    with mgmt_col1:
        if alerts:
            # Export to CSV
            df_export = pd.DataFrame(alerts)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📥 Export Alerts to CSV",
                csv,
                "crowdshield_alerts.csv",
                "text/csv",
                use_container_width=True
            )
    
    with mgmt_col2:
        if st.button("🗑️ Clear All Alerts", type="secondary", use_container_width=True):
            db.clear_all()
            st.success("All alerts cleared!")
            st.rerun()
