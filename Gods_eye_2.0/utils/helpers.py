# utils/helpers.py - Utility Functions

import cv2
import numpy as np
import time
import os

def apply_night_vision(frame):
    """Apply night vision effect to frame"""
    # Increase gamma
    gamma = 1.6
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    brightened = cv2.LUT(frame, table)
    
    # CLAHE for contrast
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return cv2.medianBlur(enhanced, 3)

def generate_report(stats, heatmap_image=None, save_path="Reports"):
    """Generate HTML report"""
    os.makedirs(save_path, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"
    filepath = os.path.join(save_path, filename)
    
    # Calculate risk score
    risk_score = min(100, 
        stats.get('peak_count', 0) * 2 +
        stats.get('max_stampede_risk', 0) +
        stats.get('weapons_detected', 0) * 20
    )
    
    risk_level = "LOW"
    risk_color = "#00ff88"
    if risk_score > 30:
        risk_level = "ELEVATED"
        risk_color = "#ffcc00"
    if risk_score > 60:
        risk_level = "CRITICAL"
        risk_color = "#ff4444"
    
    # Encode heatmap if provided
    heatmap_html = ""
    if heatmap_image is not None:
        import base64
        _, buffer = cv2.imencode('.png', heatmap_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        heatmap_html = f'''
        <h2>Heatmap Analysis</h2>
        <div class="card">
            <img src="data:image/png;base64,{img_base64}" style="max-width:100%; border-radius:10px;">
            <p style="color:#888; font-size:12px; margin-top:10px;">Movement density visualization</p>
        </div>
        '''
    
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>CrowdIntel Report</title>
    <style>
        :root {{ --primary: #00f2ff; --bg: #0a0a0c; --card: #16161a; --accent: {risk_color}; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: var(--bg); color: #e0e0e0; margin: 0; padding: 40px; }}
        .container {{ max-width: 1000px; margin: auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        h1 {{ color: var(--primary); margin: 0; }}
        .badge {{ background: var(--accent); color: #000; padding: 8px 20px; border-radius: 20px; font-weight: bold; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: var(--card); border: 1px solid #2a2a2a; padding: 25px; border-radius: 12px; }}
        .stat-value {{ font-size: 42px; font-weight: bold; color: var(--primary); }}
        .stat-label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
        h2 {{ color: var(--primary); border-bottom: 1px solid #333; padding-bottom: 10px; }}
        .footer {{ text-align: center; margin-top: 40px; color: #555; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>CROWDINTEL REPORT</h1>
                <p style="color:#666;">Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            <div class="badge">{risk_level} RISK</div>
        </header>
        
        <div class="grid">
            <div class="card">
                <div class="stat-label">Security Score</div>
                <div class="stat-value">{risk_score}%</div>
            </div>
            <div class="card">
                <div class="stat-label">Peak Crowd</div>
                <div class="stat-value">{stats.get('peak_count', 0)}</div>
            </div>
            <div class="card">
                <div class="stat-label">Max Stampede Risk</div>
                <div class="stat-value">{stats.get('max_stampede_risk', 0)}%</div>
            </div>
            <div class="card">
                <div class="stat-label">Weapons Detected</div>
                <div class="stat-value">{stats.get('weapons_detected', 0)}</div>
            </div>
        </div>
        
        <h2>Session Statistics</h2>
        <div class="card">
            <p><strong>Duration:</strong> {stats.get('duration', 'N/A')}</p>
            <p><strong>Average FPS:</strong> {stats.get('avg_fps', 'N/A')}</p>
            <p><strong>Total Persons Tracked:</strong> {stats.get('total_tracked', 0)}</p>
        </div>
        
        {heatmap_html}
        
        <div class="footer">
            <p>CrowdIntel - AI Crowd Surveillance System</p>
            <p>© 2026</p>
        </div>
    </div>
</body>
</html>
'''
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    print(f"[Report] Saved to {filepath}")
    return filepath