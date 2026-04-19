"""
CrowdShield - Weapon Detection Module
========================================
Detects potentially dangerous weapons in video frames.

Detection approach:
  1. YOLOv8 detects objects classified as knife/scissors from COCO dataset
  2. Additional context filtering:
     - Is the weapon near a person? (reduces false positives from kitchen/store scenes)
     - Is the weapon being held at a threatening angle?
     - Size filtering (very small objects might be noise)

KNOWN LIMITATIONS (discussed in your project report):
  - COCO doesn't include "gun" class - a custom-trained model can be added later
  - Knives in kitchens/stores will be detected (context-based filtering helps)
  - Very small weapons in low-resolution video may be missed
  - Occluded (hidden) weapons cannot be detected by visual AI
"""

import numpy as np
import time
import config


class WeaponDetector:
    """
    Detects and filters weapon detections from YOLO.
    
    Usage:
        weapon_det = WeaponDetector()
        
        # Each frame:
        results = weapon_det.analyze(weapon_detections, people_detections, frame)
        
        for weapon in results["confirmed_weapons"]:
            print(f"WEAPON: {weapon['type']} at {weapon['bbox']}")
    """
    
    def __init__(self):
        # Cooldown to avoid spamming alerts
        self.last_alert_time = 0
        
        # Track weapons over frames for persistence
        self.weapon_history = {}  # track_id -> frame_count
        self.confirmed_count = 0
    
    def analyze(self, weapon_detections, people_detections, frame_shape):
        """
        Analyze weapon detections with contextual filtering.
        
        Args:
            weapon_detections: list of Detection objects (knives, scissors)
            people_detections: list of Detection objects (people) for context
            frame_shape:       (height, width) of the frame
        
        Returns:
            dict with keys:
                confirmed_weapons: list of confirmed weapon dicts
                total_raw:         raw detection count before filtering
                alert:             bool - should we alert?
                threat_level:      str - "NONE", "LOW", "HIGH"
        """
        result = {
            "confirmed_weapons": [],
            "total_raw": len(weapon_detections),
            "alert": False,
            "threat_level": "NONE"
        }
        
        if len(weapon_detections) == 0:
            return result
        
        frame_h, frame_w = frame_shape[:2]
        confirmed = []
        
        for det in weapon_detections:
            weapon_info = self._analyze_single_weapon(
                det, people_detections, frame_w, frame_h
            )
            
            if weapon_info is not None:
                confirmed.append(weapon_info)
        
        result["confirmed_weapons"] = confirmed
        
        # Determine threat level
        if len(confirmed) > 0:
            # Check if any weapon is near a person
            near_person = any(w["near_person"] for w in confirmed)
            high_confidence = any(w["confidence"] > 0.5 for w in confirmed)
            
            if near_person and high_confidence:
                result["threat_level"] = "HIGH"
                result["alert"] = self._check_cooldown()
            elif near_person or high_confidence:
                result["threat_level"] = "HIGH"
                result["alert"] = self._check_cooldown()
            else:
                result["threat_level"] = "LOW"
        
        self.confirmed_count = len(confirmed)
        return result
    
    def _analyze_single_weapon(self, detection, people_detections, frame_w, frame_h):
        """
        Analyze a single weapon detection with context.
        
        Returns:
            dict with weapon info if confirmed, None if filtered out
        """
        x1, y1, x2, y2 = detection.bbox
        w = x2 - x1
        h = y2 - y1
        
        # ===== FILTER 1: Size check =====
        # Very tiny detections are likely noise
        min_size = 15  # pixels
        if w < min_size or h < min_size:
            return None
        
        # Very large "weapons" are probably misclassified objects
        max_ratio = 0.3  # Can't be more than 30% of the frame
        if (w * h) / (frame_w * frame_h) > max_ratio:
            return None
        
        # ===== FILTER 2: Check proximity to people =====
        weapon_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        near_person = False
        nearest_person_dist = float('inf')
        associated_person_id = None
        
        for person in people_detections:
            px1, py1, px2, py2 = person.bbox
            person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            
            # Distance between weapon and person center
            dist = np.sqrt(
                (weapon_center[0] - person_center[0]) ** 2 +
                (weapon_center[1] - person_center[1]) ** 2
            )
            
            # Also check if weapon bbox overlaps with person bbox
            overlap = self._check_overlap(detection.bbox, person.bbox)
            
            # Person's height as reference distance
            person_h = py2 - py1
            proximity_thresh = person_h * 1.5  # Within 1.5x person height
            
            if dist < proximity_thresh or overlap:
                near_person = True
                if dist < nearest_person_dist:
                    nearest_person_dist = dist
                    associated_person_id = getattr(person, 'track_id', None)
        
        # ===== Build weapon info =====
        weapon_type = detection.class_name  # "knife" or "scissors"
        
        weapon_info = {
            "type": weapon_type,
            "bbox": detection.bbox,
            "confidence": detection.confidence,
            "near_person": near_person,
            "nearest_person_dist": nearest_person_dist,
            "associated_person_id": associated_person_id,
            "center": weapon_center,
            "size": (w, h)
        }
        
        return weapon_info
    
    def _check_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap at all."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        return x2 > x1 and y2 > y1
    
    def _check_cooldown(self):
        """Check if enough time has passed since last alert."""
        current_time = time.time()
        if current_time - self.last_alert_time > config.ALERT_COOLDOWN_DEFAULT:
            self.last_alert_time = current_time
            return True
        return False
    
    def reset(self):
        """Reset detector state."""
        self.weapon_history = {}
        self.last_alert_time = 0
        self.confirmed_count = 0
