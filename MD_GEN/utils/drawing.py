"""
CrowdShield - Drawing Utilities
==================================
Helper functions for drawing bounding boxes, labels, overlays, and
other visual elements on video frames.

These are used by the UI to render detection results on the video feed.
"""

import cv2
import numpy as np
import config


# ============================================================
# COLOR DEFINITIONS (BGR format for OpenCV)
# ============================================================
COLORS = {
    "person":     (255, 180, 0),    # Light blue
    "child":      (147, 20, 255),   # Pink/Magenta
    "weapon":     (0, 0, 255),      # Red
    "fight":      (0, 100, 255),    # Orange
    "stampede":   (0, 0, 200),      # Dark red
    "target":     (0, 255, 0),      # Green
    "searching":  (0, 255, 255),    # Yellow
    "trail":      (0, 200, 200),    # Dark yellow
    "safe":       (0, 200, 0),      # Green
    "warning":    (0, 200, 255),    # Orange
    "critical":   (0, 0, 255),      # Red
}

# Person ID colors (cycle through these for different tracked people)
ID_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (100, 200, 150), (150, 100, 200),
    (255, 200, 50),  (50, 200, 255),  (200, 50, 255),
]


def get_id_color(track_id):
    """Get a consistent color for a track ID."""
    if track_id is None:
        return COLORS["person"]
    return ID_COLORS[track_id % len(ID_COLORS)]


def draw_detection(frame, detection, label=None, color=None, thickness=2):
    """
    Draw a bounding box and label for a detection.
    
    Args:
        frame:     Image to draw on (modified in place)
        detection: Detection object with .bbox attribute
        label:     Label text (auto-generated if None)
        color:     BGR color tuple (auto from class if None)
        thickness: Box line thickness
    """
    x1, y1, x2, y2 = [int(v) for v in detection.bbox]
    
    # Choose color
    if color is None:
        if detection.class_name in COLORS:
            color = COLORS[detection.class_name]
        elif hasattr(detection, 'track_id') and detection.track_id is not None:
            color = get_id_color(detection.track_id)
        else:
            color = COLORS["person"]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Build label
    if label is None:
        parts = []
        if hasattr(detection, 'track_id') and detection.track_id is not None:
            parts.append(f"#{detection.track_id}")
        parts.append(f"{detection.confidence:.0%}")
        label = " ".join(parts)
    
    # Draw label background
    if label:
        draw_label(frame, label, (x1, y1 - 5), color)


def draw_label(frame, text, position, color=(255, 255, 255), font_scale=0.5, thickness=1):
    """
    Draw a text label with a filled background.
    
    Args:
        frame:     Image to draw on
        text:      Text string
        position:  (x, y) position for the text baseline
        color:     BGR color for the background
    """
    x, y = int(position[0]), int(position[1])
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Ensure text stays within frame
    h, w = frame.shape[:2]
    x = max(0, min(x, w - text_w))
    y = max(text_h + baseline, min(y, h))
    
    # Draw filled rectangle behind text
    cv2.rectangle(frame, (x, y - text_h - baseline), (x + text_w, y), color, -1)
    
    # Draw text (white or black depending on background brightness)
    brightness = sum(color) / 3
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    cv2.putText(frame, text, (x, y - baseline), font, font_scale, text_color, thickness)


def draw_tracks(frame, tracks, show_trail=True, show_id=True):
    """
    Draw all tracked people with dots at their feet and optional trails.
    Uses colored dots (like GOD'S EYE NEXUS style) instead of full rectangles
    for a cleaner look in crowded scenes.
    
    Args:
        frame:      Image to draw on
        tracks:     List of Track objects
        show_trail: Whether to draw movement trails
        show_id:    Whether to show track IDs
    """
    for track in tracks:
        if track.lost_frames > 0:
            continue  # Don't draw lost tracks
        
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        color = get_id_color(track.track_id)
        
        # Draw a DOT at the center of the person instead of a rectangle
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, color, -1)        # Filled dot
        cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1)    # Black outline
        
        # Draw track ID as small label above the dot
        if show_id:
            label = f"#{track.track_id}"
            draw_label(frame, label, (cx - 10, cy - 18), color)
        
        # Draw trail
        if show_trail and len(track.history) > 1:
            points = list(track.history)
            for i in range(1, len(points)):
                alpha = i / len(points)
                trail_color = tuple(int(c * alpha) for c in color)
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(frame, pt1, pt2, trail_color, max(1, int(2 * alpha)))


def draw_weapon_alert(frame, weapon_info):
    """
    Draw weapon detection with prominent alert styling.
    
    Args:
        frame:       Image to draw on
        weapon_info: Dict from WeaponDetector with bbox, type, etc.
    """
    x1, y1, x2, y2 = [int(v) for v in weapon_info["bbox"]]
    color = COLORS["weapon"]
    
    # Draw thick red box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw warning triangles around the weapon
    label = f"WEAPON: {weapon_info['type'].upper()}"
    draw_label(frame, label, (x1, y1 - 5), color, font_scale=0.6, thickness=2)
    
    # Draw pulsing border effect (using frame count as time)
    _draw_alert_border(frame, (x1-5, y1-5, x2+5, y2+5), color)


def draw_fight_alert(frame, fight_pair):
    """
    Draw fight detection between two people.
    
    Args:
        frame:      Image to draw on
        fight_pair: Dict with person1 and person2 info
    """
    color = COLORS["fight"]
    
    for person_key in ["person1", "person2"]:
        person = fight_pair[person_key]
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw line between the two people
    c1 = fight_pair["person1"]["center"]
    c2 = fight_pair["person2"]["center"]
    cv2.line(frame, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), color, 2)
    
    # Draw "FIGHT" label between them
    mid_x = int((c1[0] + c2[0]) / 2)
    mid_y = int((c1[1] + c2[1]) / 2)
    draw_label(frame, "FIGHT DETECTED", (mid_x - 50, mid_y), color, font_scale=0.7, thickness=2)


def draw_child_alert(frame, child_info, is_alone=False):
    """
    Draw a detected child with special styling.
    
    Args:
        frame:      Image to draw on  
        child_info: Dict with child detection info
        is_alone:   Whether the child is alone (triggers alert styling)
    """
    x1, y1, x2, y2 = [int(v) for v in child_info["bbox"]]
    
    if is_alone:
        color = COLORS["critical"]
        label = "LOST CHILD!"
        thickness = 3
    else:
        color = COLORS["child"]
        label = "Child"
        thickness = 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label(frame, label, (x1, y1 - 5), color, font_scale=0.6)


def draw_stampede_warning(frame, result):
    """
    Draw stampede warning overlay on the frame.
    
    Args:
        frame:  Image to draw on
        result: Dict from StampedeDetector
    """
    severity = result["severity"]
    
    if severity == "NONE":
        return
    
    if severity == "CRITICAL":
        color = COLORS["critical"]
        text = "STAMPEDE ALERT!"
    else:
        color = COLORS["warning"]
        text = "STAMPEDE WARNING"
    
    h, w = frame.shape[:2]
    
    # Draw semi-transparent red border
    overlay = frame.copy()
    border = 8
    cv2.rectangle(overlay, (0, 0), (w, border), color, -1)        # Top
    cv2.rectangle(overlay, (0, h-border), (w, h), color, -1)      # Bottom
    cv2.rectangle(overlay, (0, 0), (border, h), color, -1)        # Left
    cv2.rectangle(overlay, (w-border, 0), (w, h), color, -1)      # Right
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw centered text
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, 1.5, 3)
    x = (w - text_w) // 2
    y = text_h + 30
    
    # Text shadow
    cv2.putText(frame, text, (x+2, y+2), font, 1.5, (0, 0, 0), 4)
    cv2.putText(frame, text, (x, y), font, 1.5, color, 3)
    
    # Draw flow direction arrow
    direction = result.get("flow_direction", (0, 0))
    if abs(direction[0]) > 0.1 or abs(direction[1]) > 0.1:
        arrow_start = (w // 2, h // 2)
        arrow_end = (
            int(w // 2 + direction[0] * 100),
            int(h // 2 + direction[1] * 100)
        )
        cv2.arrowedLine(frame, arrow_start, arrow_end, (255, 255, 255), 3, tipLength=0.3)


def draw_info_panel(frame, info_dict, position="top-right"):
    """
    Draw an info panel with key-value pairs.
    
    Args:
        frame:     Image to draw on
        info_dict: Dict of {label: value} pairs to display
        position:  "top-right", "top-left", "bottom-right", "bottom-left"
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_height = 22
    padding = 10
    
    # Calculate panel size
    lines = [f"{k}: {v}" for k, v in info_dict.items()]
    max_text_width = max(
        cv2.getTextSize(line, font, font_scale, 1)[0][0] 
        for line in lines
    ) if lines else 100
    
    panel_w = max_text_width + padding * 2
    panel_h = len(lines) * line_height + padding * 2
    
    # Position
    if "right" in position:
        px = w - panel_w - 10
    else:
        px = 10
    
    if "top" in position:
        py = 10
    else:
        py = h - panel_h - 10
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (100, 100, 100), 1)
    
    # Draw text
    for i, line in enumerate(lines):
        text_y = py + padding + (i + 1) * line_height
        cv2.putText(frame, line, (px + padding, text_y), font, font_scale, (255, 255, 255), 1)


def _draw_alert_border(frame, rect, color, thickness=2):
    """Draw a dashed/animated alert border around a rectangle."""
    x1, y1, x2, y2 = [int(v) for v in rect]
    
    # Draw corner brackets instead of full rectangle
    corner_len = min(20, (x2-x1)//4, (y2-y1)//4)
    
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
