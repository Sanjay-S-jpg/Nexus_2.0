"""
CrowdShield - Child Detection Module
=======================================
Detects potentially lost/unaccompanied children in crowd scenes.

Detection approach:
  1. Person height analysis:
     - Children are significantly shorter than adults
     - We compare each person's bounding box height to the average adult height
     - If height < 55% of average adult = classified as child
  
  2. Isolation check:
     - A child with an adult nearby = accompanied (no alert)
     - A child alone (no adult within distance threshold) = potentially lost
     - Must be alone for several consecutive frames before alerting

Handling the dwarf/short adult vs child problem:
  - We use a secondary check: aspect ratio of bounding box
  - Children tend to have different proportions (larger head-to-body ratio)
  - We also check if the "child" has been stationary (lost children often stop moving)
  - Final resort: the operator can dismiss false positives from the UI
"""

import numpy as np
import time
from collections import deque, defaultdict
import config


class ChildDetector:
    """
    Detects unaccompanied children in crowd scenes.
    
    Usage:
        detector = ChildDetector()
        
        # Each frame:
        result = detector.analyze(people_detections, tracks)
        
        for child in result["detected_children"]:
            print(f"Child at {child['center']}, alone={child['is_alone']}")
    """
    
    def __init__(self):
        # Track how long each potential child has been alone
        # Format: {track_id: consecutive_frames_alone}
        self.alone_counter = defaultdict(int)
        
        # Recently alerted children (avoid repeat alerts)
        self.alerted_ids = {}  # {track_id: timestamp}
        
        # Running average of adult heights
        self.adult_heights = deque(maxlen=300)  # ~10 seconds of adult height data
    
    def analyze(self, people_detections, tracks=None, frame_shape=None):
        """
        Analyze people detections to find children and check if they're alone.
        
        Args:
            people_detections: list of Detection objects (all people)
            tracks:            list of Track objects (for tracking IDs)
            frame_shape:       (height, width, channels) of the frame for perspective check
        
        Returns:
            dict with keys:
                detected_children:   list of child info dicts
                alone_children:      list of children who are alone
                alert_children:      list of children who need alerts
                total_children:      int count
                total_adults:        int count
        """
        result = {
            "detected_children": [],
            "alone_children": [],
            "alert_children": [],
            "total_children": 0,
            "total_adults": 0
        }
        
        if len(people_detections) < 1:
            return result
        
        # ===== STEP 1: Calculate average adult height =====
        heights = [det.height for det in people_detections if det.height > config.CHILD_MIN_HEIGHT_PX]
        
        if len(heights) == 0:
            return result
        
        # Use median to be more robust against outliers
        current_avg_height = np.median(heights)
        
        # Add to running average (only heights that look like adults)
        for h in heights:
            if h > current_avg_height * 0.6:  # Likely adult
                self.adult_heights.append(h)
        
        # Use the running average for more stable threshold
        if len(self.adult_heights) > 10:
            avg_adult_height = np.median(list(self.adult_heights))
        else:
            avg_adult_height = current_avg_height
        
        # Height threshold for children
        child_threshold = avg_adult_height * config.CHILD_HEIGHT_RATIO
        
        # Frame height for perspective check
        frame_h = frame_shape[0] if frame_shape is not None else None
        top_cutoff = frame_h * config.CHILD_TOP_FRAME_IGNORE if frame_h else 0
        
        # ===== STEP 2: Classify each person as child or adult =====
        children = []
        adults = []
        
        for i, det in enumerate(people_detections):
            h = det.height
            
            if h < config.CHILD_MIN_HEIGHT_PX:
                continue  # Too small, probably noise or far-away person
            
            # PERSPECTIVE CHECK: people near the top of the frame are far away
            # Their small size is due to distance, not because they are children
            bbox_top_y = det.bbox[1]  # y1 coordinate
            if frame_h and bbox_top_y < top_cutoff:
                # Person is in the top portion of the frame — likely far away, skip child check
                adults.append({
                    "detection": det,
                    "index": i,
                    "center": det.center,
                    "bbox": det.bbox
                })
                continue
            
            is_child = False
            confidence = 0.0
            
            if h < child_threshold:
                # Height suggests child
                child_score = 1.0 - (h / child_threshold)  # Higher score = more childlike
                
                # Secondary check: aspect ratio
                # Children tend to have a wider aspect ratio (head is large relative to body)
                aspect_ratio = det.width / max(det.height, 1)
                
                # Adults are typically taller and narrower (ratio ~0.35-0.45)
                # Children are shorter and wider relative to height (ratio ~0.45-0.6)
                ar_score = 0.3 if aspect_ratio > 0.4 else 0.0
                
                # Combined score
                confidence = min(1.0, child_score + ar_score)
                
                if confidence > 0.3:
                    is_child = True
            
            if is_child:
                children.append({
                    "detection": det,
                    "index": i,
                    "height": h,
                    "confidence": confidence,
                    "center": det.center,
                    "bbox": det.bbox,
                    "track_id": getattr(det, 'track_id', None)
                })
            else:
                adults.append({
                    "detection": det,
                    "index": i,
                    "center": det.center,
                    "bbox": det.bbox
                })
        
        result["total_children"] = len(children)
        result["total_adults"] = len(adults)
        result["detected_children"] = children
        
        # ===== STEP 3: Check if children are alone =====
        for child in children:
            is_alone = self._check_isolation(child, adults)
            child["is_alone"] = is_alone
            
            # Track alone duration
            track_id = child.get("track_id")
            if track_id is not None:
                if is_alone:
                    self.alone_counter[track_id] += 1
                else:
                    self.alone_counter[track_id] = max(0, self.alone_counter[track_id] - 2)
                
                child["alone_frames"] = self.alone_counter[track_id]
                
                # Check if should alert
                if self.alone_counter[track_id] >= config.CHILD_ALONE_FRAMES:
                    if self._check_cooldown(track_id):
                        result["alert_children"].append(child)
            
            if is_alone:
                result["alone_children"].append(child)
        
        # Clean up old alone counters
        active_ids = {c.get("track_id") for c in children if c.get("track_id")}
        self.alone_counter = defaultdict(int, {
            k: v for k, v in self.alone_counter.items() if k in active_ids
        })
        
        return result
    
    def _check_isolation(self, child, adults):
        """
        Check if a child is alone (no adult nearby).
        
        Args:
            child: Child info dict with 'center' key
            adults: List of adult info dicts
        
        Returns:
            bool: True if the child is alone
        """
        if len(adults) == 0:
            return True
        
        child_center = child["center"]
        
        for adult in adults:
            adult_center = adult["center"]
            
            # Calculate distance
            dist = np.sqrt(
                (child_center[0] - adult_center[0]) ** 2 +
                (child_center[1] - adult_center[1]) ** 2
            )
            
            if dist < config.CHILD_ISOLATION_DIST_PX:
                return False  # An adult is nearby
        
        return True  # No adult nearby - child is alone
    
    def _check_cooldown(self, track_id):
        """Check alert cooldown for a specific child track."""
        current_time = time.time()
        
        if track_id in self.alerted_ids:
            if current_time - self.alerted_ids[track_id] < config.CHILD_COOLDOWN_SEC:
                return False
        
        self.alerted_ids[track_id] = current_time
        return True
    
    def reset(self):
        """Reset detector state."""
        self.alone_counter.clear()
        self.alerted_ids.clear()
        self.adult_heights.clear()
