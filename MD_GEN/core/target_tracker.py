"""
CrowdShield - Target Tracker & Re-Identification Module
=========================================================
Allows security to select a specific person and track them across frames.
If the person goes out of frame or is occluded, the system tries to
re-identify them when they reappear using appearance features.

How Re-Identification works:
  1. When a target is selected, we extract their appearance "signature":
     - Color histogram of their clothing/body
     - Bounding box proportions
  2. If the target is lost (leaves frame or is occluded), we compare
     all new detections against the saved signature
  3. The best match above our threshold is considered the re-identified target

LIMITATIONS:
  - This uses simple color histograms, not a deep learning Re-ID model
  - It works well when the target's clothing is distinctive
  - May struggle if many people wear similar colors
  - For a production system, you'd use OSNet or similar deep Re-ID networks
  - Works great for demo purposes!
"""

import cv2
import numpy as np
from collections import deque
import config


class TargetTracker:
    """
    Tracks and re-identifies a selected target person.
    
    Usage:
        tracker = TargetTracker()
        
        # User clicks on a person to select as target:
        tracker.set_target(frame, person_bbox, track_id=5)
        
        # Each frame:
        result = tracker.update(frame, people_detections, tracks)
        
        if result["target_found"]:
            print(f"Target at: {result['target_bbox']}")
        elif result["is_searching"]:
            print("Target lost - searching...")
    """
    
    def __init__(self):
        # Target info
        self.is_active = False          # Is there an active target?
        self.target_track_id = None     # Track ID of the target
        self.target_features = None     # Appearance features (color histogram)
        self.target_bbox = None         # Last known bounding box
        self.target_crop = None         # Cropped image of the target (for display)
        
        # Status
        self.is_found = False           # Currently tracking the target
        self.is_searching = False       # Target lost, searching for re-ID
        self.lost_frames = 0            # Frames since target was last seen
        
        # History of target positions (for drawing trail)
        self.position_history = deque(maxlen=120)  # ~4 seconds at 30fps
    
    def set_target(self, frame, bbox, track_id=None):
        """
        Select a person as the tracking target.
        
        Args:
            frame:    Current video frame (BGR)
            bbox:     [x1, y1, x2, y2] bounding box of the target person
            track_id: Optional track ID (from the multi-object tracker)
        """
        self.is_active = True
        self.target_track_id = track_id
        self.target_bbox = list(bbox)
        self.is_found = True
        self.is_searching = False
        self.lost_frames = 0
        self.position_history.clear()
        
        # Extract appearance features
        self.target_features = self._extract_features(frame, bbox)
        
        # Save a cropped image of the target (for showing in UI)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        self.target_crop = frame[y1:y2, x1:x2].copy()
        
        # Add initial position
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.position_history.append((cx, cy))
        
        print(f"[TargetTracker] Target set: track_id={track_id}")
    
    def clear_target(self):
        """Stop tracking the current target."""
        self.is_active = False
        self.target_track_id = None
        self.target_features = None
        self.target_bbox = None
        self.target_crop = None
        self.is_found = False
        self.is_searching = False
        self.lost_frames = 0
        self.position_history.clear()
        print("[TargetTracker] Target cleared.")
    
    def update(self, frame, people_detections, tracks=None):
        """
        Update target tracking for a new frame.
        
        Args:
            frame:              Current video frame (BGR)
            people_detections:  list of Detection objects
            tracks:             list of Track objects
        
        Returns:
            dict with keys:
                target_found:  bool - is the target currently visible
                target_bbox:   [x1,y1,x2,y2] or None
                is_searching:  bool - looking for lost target
                is_active:     bool - is tracking active
                lost_frames:   int - frames since last seen
                match_score:   float - re-ID match confidence
                trail:         list of (x,y) positions
        """
        result = {
            "target_found": False,
            "target_bbox": None,
            "is_searching": False,
            "is_active": self.is_active,
            "lost_frames": self.lost_frames,
            "match_score": 0.0,
            "trail": list(self.position_history)
        }
        
        if not self.is_active or self.target_features is None:
            return result
        
        # ===== Method 1: Try to find by track ID =====
        found_by_track = False
        if self.target_track_id is not None and tracks is not None:
            for track in tracks:
                if track.track_id == self.target_track_id and track.lost_frames == 0:
                    self._update_found(frame, track.bbox)
                    found_by_track = True
                    result["target_found"] = True
                    result["target_bbox"] = self.target_bbox
                    result["match_score"] = 1.0
                    break
        
        if found_by_track:
            result["trail"] = list(self.position_history)
            return result
        
        # ===== Method 2: Try to re-identify by appearance =====
        best_match = None
        best_score = 0.0
        
        for det in people_detections:
            features = self._extract_features(frame, det.bbox)
            if features is None:
                continue
            
            score = self._compare_features(self.target_features, features)
            
            if score > best_score:
                best_score = score
                best_match = det
        
        if best_match is not None and best_score >= config.TARGET_REID_MATCH_THRESH:
            # Re-identified!
            self._update_found(frame, best_match.bbox)
            
            # Update track ID if available
            if hasattr(best_match, 'track_id') and best_match.track_id is not None:
                self.target_track_id = best_match.track_id
            
            result["target_found"] = True
            result["target_bbox"] = self.target_bbox
            result["match_score"] = best_score
            
            if self.is_searching:
                print(f"[TargetTracker] Target re-identified! Score: {best_score:.2f}")
                self.is_searching = False
        else:
            # Target not found this frame
            self.lost_frames += 1
            self.is_found = False
            
            if self.lost_frames > 5:  # Give a few frames grace period
                self.is_searching = True
            
            result["is_searching"] = self.is_searching
            
            # Give up if lost too long
            if self.lost_frames > config.TARGET_LOST_TIMEOUT:
                print("[TargetTracker] Target lost for too long. Clearing.")
                self.is_searching = False
                # Don't clear - keep the target info but mark as lost
        
        result["lost_frames"] = self.lost_frames
        result["trail"] = list(self.position_history)
        return result
    
    def _update_found(self, frame, bbox):
        """Update state when target is found/matched."""
        self.target_bbox = list(bbox)
        self.is_found = True
        self.is_searching = False
        self.lost_frames = 0
        
        # Update position history
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.position_history.append((cx, cy))
        
        # Update appearance features periodically (appearance changes as they move)
        if len(self.position_history) % 15 == 0:  # Every 0.5 seconds
            new_features = self._extract_features(frame, bbox)
            if new_features is not None:
                # Blend new features with old (weighted average)
                # This adapts to gradual appearance changes
                self.target_features = (
                    0.7 * self.target_features + 0.3 * new_features
                )
    
    def _extract_features(self, frame, bbox):
        """
        Extract appearance features from a person's bounding box.
        
        Uses color histogram in HSV space:
          - Hue captures the color of clothing
          - Saturation captures color intensity
          - Value captures brightness
        
        HSV is better than RGB for appearance matching because
        it's more robust to lighting changes.
        
        Args:
            frame: BGR image
            bbox:  [x1, y1, x2, y2]
        
        Returns:
            numpy array of histogram features (normalized)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        
        # Clip to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        
        # Crop the person
        crop = frame[y1:y2, x1:x2]
        
        # Use only the middle portion (avoid background at edges)
        ch, cw = crop.shape[:2]
        margin_x = int(cw * 0.15)
        margin_y = int(ch * 0.1)
        if ch > margin_y * 3 and cw > margin_x * 3:
            crop = crop[margin_y:ch-margin_y, margin_x:cw-margin_x]
        
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        bins = config.TARGET_FEATURE_BINS
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Concatenate and normalize
        features = np.concatenate([
            h_hist.flatten(),
            s_hist.flatten(),
            v_hist.flatten()
        ])
        
        # Normalize to sum = 1
        total = features.sum()
        if total > 0:
            features = features / total
        
        return features
    
    def _compare_features(self, features1, features2):
        """
        Compare two feature vectors using histogram comparison.
        
        Uses Bhattacharyya distance (works well for histograms):
          - Score of 1.0 = identical
          - Score of 0.0 = completely different
        
        Args:
            features1, features2: Feature vectors from _extract_features
        
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Ensure same shape
        if features1.shape != features2.shape:
            return 0.0
        
        # Use correlation method (1 = identical, -1 = opposite)
        # This is more intuitive than Bhattacharyya for our use
        score = cv2.compareHist(
            features1.astype(np.float32),
            features2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Normalize to 0-1 range
        return max(0.0, (score + 1.0) / 2.0)
    
    def draw_target_overlay(self, frame):
        """
        Draw the target tracking visualization on a frame.
        
        Draws:
          - Thick colored bounding box around target
          - Trail of past positions
          - "TRACKING" / "SEARCHING" label
          - Target ID
        
        Args:
            frame: BGR image to draw on (modified in place)
        
        Returns:
            Modified frame
        """
        if not self.is_active:
            return frame
        
        # Draw trail (line connecting past positions)
        if len(self.position_history) > 1:
            points = list(self.position_history)
            for i in range(1, len(points)):
                # Fade trail: older = more transparent
                alpha = i / len(points)
                color = (0, int(255 * alpha), 255)  # Yellow to green
                thickness = max(1, int(2 * alpha))
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(frame, pt1, pt2, color, thickness)
        
        if self.is_found and self.target_bbox is not None:
            # Draw thick green box around target
            x1, y1, x2, y2 = [int(v) for v in self.target_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Label
            label = f"TARGET #{self.target_track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif self.is_searching and self.target_bbox is not None:
            # Draw dashed red box at last known position
            x1, y1, x2, y2 = [int(v) for v in self.target_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            label = "SEARCHING..."
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
