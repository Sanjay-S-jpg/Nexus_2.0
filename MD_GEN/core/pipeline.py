"""
CrowdShield - Processing Pipeline
====================================
The main processing pipeline that connects all detection modules together.
This is the "brain" of CrowdShield - it takes a video frame and runs it
through all detection modules, collecting results.

Processing flow for each frame:
  1. Night vision enhancement (if dark)
  2. YOLO detection (people + weapons in one pass)
  3. Pose estimation (for fight detection)
  4. Multi-object tracking (assign IDs to people)
  5. Stampede detection (optical flow + density)
  6. Weapon analysis (contextual filtering)
  7. Fight detection (pose-based)
  8. Child detection (height + isolation)
  9. Crowd analysis (counting, trends)
  10. Heatmap update
  11. Target tracking (if active)
  12. Alert generation

The pipeline supports enabling/disabling individual features
and automatically adjusts processing when FPS drops.
"""

import time
import cv2
import numpy as np

# Core modules
from core.detector import YOLODetector
from core.tracker import MultiObjectTracker
from core.stampede import StampedeDetector
from core.weapon_detector import WeaponDetector
from core.fight_detector import FightDetector
from core.crowd_analyzer import CrowdAnalyzer
from core.heatmap import HeatmapGenerator
from core.night_vision import NightVisionEnhancer
from core.child_detector import ChildDetector
from core.target_tracker import TargetTracker

# Utilities
from utils.alert_system import AlertManager
from utils.drawing import (
    draw_tracks, draw_weapon_alert, draw_fight_alert,
    draw_child_alert, draw_stampede_warning, draw_info_panel
)

import config


class Pipeline:
    """
    Main CrowdShield processing pipeline.
    
    Usage:
        pipeline = Pipeline()
        
        # Process a frame:
        results = pipeline.process_frame(frame)
        
        # Get annotated frame:
        annotated = results["annotated_frame"]
        
        # Check alerts:
        if results["alerts"]:
            for alert in results["alerts"]:
                print(f"ALERT: {alert}")
    """
    
    def __init__(self):
        """Initialize all modules."""
        print("=" * 50)
        print("  CrowdShield - Initializing Pipeline")
        print("=" * 50)
        
        # ===== Initialize core modules =====
        print("\n[Pipeline] Loading AI models (this may take a moment)...")
        self.detector = YOLODetector()
        
        print("[Pipeline] Initializing tracker...")
        self.tracker = MultiObjectTracker()
        
        print("[Pipeline] Initializing detection modules...")
        self.stampede_detector = StampedeDetector()
        self.weapon_detector = WeaponDetector()
        self.fight_detector = FightDetector()
        self.child_detector = ChildDetector()
        self.crowd_analyzer = CrowdAnalyzer()
        self.heatmap_gen = None   # Initialized on first frame (needs frame size)
        self.night_vision = NightVisionEnhancer()
        self.target_tracker = TargetTracker()
        
        print("[Pipeline] Initializing alert system...")
        self.alert_manager = AlertManager()
        
        # ===== Feature toggles =====
        # The UI can enable/disable these
        self.features_enabled = {
            "people_detection": True,   # Always on (base feature)
            "tracking": True,           # Multi-object tracking
            "stampede_detection": True,
            "weapon_detection": True,
            "fight_detection": True,
            "child_detection": True,
            "crowd_analysis": True,
            "heatmap": True,
            "night_vision": True,
            "target_tracking": False,   # Only when user selects a target
        }
        
        # ===== Drawing toggles =====
        self.draw_options = {
            "show_boxes": True,         # Bounding boxes
            "show_ids": True,           # Track IDs
            "show_trails": False,       # Movement trails
            "show_heatmap": False,      # Heatmap overlay
            "show_info_panel": True,    # Stats panel
            "show_optical_flow": False, # Flow visualization
        }
        
        # ===== Performance tracking =====
        self.fps = 0.0
        self._fps_timer = time.time()
        self._fps_count = 0
        self._frame_count = 0
        self._skip_counter = 0  # For frame skipping when FPS is low
        self._frame_skip = config.FRAME_SKIP_RATIO  # Process every Nth frame
        self._last_results = None        # Cache last results for skipped frames
        self._pose_counter = 0           # Run pose every N frames
        self._cached_poses = []          # Cached pose detections
        
        print("\n[Pipeline] Initialization complete!")
        print("=" * 50)
    
    def process_frame(self, frame):
        """
        Process one video frame through the entire pipeline.
        
        Args:
            frame: BGR image (numpy array) from video source
        
        Returns:
            dict with ALL results from every module:
            {
                "annotated_frame":  frame with all drawings,
                "original_frame":   unmodified frame,
                "people_count":     int,
                "people_detections": list,
                "weapon_results":   dict,
                "stampede_results":  dict,
                "fight_results":    dict,
                "child_results":    dict,
                "crowd_results":    dict,
                "target_results":   dict,
                "heatmap_image":    image or None,
                "alerts":           list of new alerts this frame,
                "fps":              float,
                "brightness":       float,
                "is_night_mode":    bool,
                "tracks":           list of Track objects,
            }
        """
        if frame is None:
            return self._empty_results()
        
        start_time = time.time()
        self._frame_count += 1
        
        # ===== FRAME SKIPPING (big FPS boost) =====
        # Only fully process every Nth frame; return cached results otherwise
        self._skip_counter += 1
        if self._skip_counter < self._frame_skip and self._last_results is not None:
            # Return cached results with the current frame drawn
            return self._last_results
        self._skip_counter = 0
        
        alerts_this_frame = []
        
        # Initialize heatmap on first frame
        if self.heatmap_gen is None:
            h, w = frame.shape[:2]
            self.heatmap_gen = HeatmapGenerator(w, h)
        
        # ===== 1. NIGHT VISION ENHANCEMENT =====
        original_frame = frame.copy()
        is_night = False
        
        if self.features_enabled["night_vision"]:
            frame, is_night = self.night_vision.process(frame)
        
        brightness = self.night_vision.current_brightness
        
        # ===== 2. YOLO DETECTION (People + Weapons) =====
        people_detections = []
        weapon_detections = []
        
        if self.features_enabled["people_detection"]:
            if self.features_enabled["weapon_detection"]:
                # Detect both in one pass (efficient)
                people_detections, weapon_detections = self.detector.detect_all_relevant(frame)
            else:
                people_detections = self.detector.detect_people(frame)
        
        # ===== 3. POSE ESTIMATION (for fight detection) =====
        # Only run pose every N frames to save processing time (~40ms per run)
        pose_detections = []
        self._pose_counter += 1
        if self.features_enabled["fight_detection"] and len(people_detections) >= 2:
            if self._pose_counter >= config.POSE_EVERY_N_FRAMES:
                pose_detections = self.detector.detect_poses(frame)
                self._cached_poses = pose_detections
                self._pose_counter = 0
            else:
                pose_detections = self._cached_poses  # Use cached poses
        
        # ===== 4. MULTI-OBJECT TRACKING =====
        tracks = []
        if self.features_enabled["tracking"]:
            tracks = self.tracker.update(people_detections)
        
        # ===== 5. STAMPEDE DETECTION =====
        stampede_results = {"is_stampede": False, "severity": "NONE"}
        if self.features_enabled["stampede_detection"]:
            stampede_results = self.stampede_detector.analyze(
                frame, len(people_detections), tracks
            )
            if stampede_results["is_stampede"]:
                alert = self.alert_manager.trigger_alert(
                    "stampede",
                    stampede_results["severity"],
                    f"Stampede detected! Flow: {stampede_results['flow_magnitude']:.1f}, "
                    f"Uniformity: {stampede_results['uniformity']:.1%}",
                    data=stampede_results,
                    frame=frame
                )
                if alert:
                    alerts_this_frame.append(alert.to_dict())
        
        # ===== 6. WEAPON DETECTION =====
        weapon_results = {"confirmed_weapons": [], "threat_level": "NONE"}
        if self.features_enabled["weapon_detection"] and len(weapon_detections) > 0:
            weapon_results = self.weapon_detector.analyze(
                weapon_detections, people_detections, frame.shape
            )
            if weapon_results["alert"]:
                for w in weapon_results["confirmed_weapons"]:
                    alert = self.alert_manager.trigger_alert(
                        "weapon",
                        "CRITICAL",
                        f"Weapon detected: {w['type']} (confidence: {w['confidence']:.0%})",
                        data=w,
                        frame=frame
                    )
                    if alert:
                        alerts_this_frame.append(alert.to_dict())
        
        # ===== 7. FIGHT DETECTION =====
        fight_results = {"is_fight": False, "severity": "NONE", "fight_pairs": []}
        if self.features_enabled["fight_detection"] and len(pose_detections) >= 2:
            fight_results = self.fight_detector.analyze(pose_detections, tracks)
            if fight_results["is_fight"]:
                alert = self.alert_manager.trigger_alert(
                    "fight",
                    fight_results["severity"],
                    f"Fight detected! {len(fight_results['fight_pairs'])} pair(s) involved",
                    data=fight_results,
                    frame=frame
                )
                if alert:
                    alerts_this_frame.append(alert.to_dict())
        
        # ===== 8. CHILD DETECTION =====
        child_results = {"detected_children": [], "alone_children": [], "alert_children": []}
        if self.features_enabled["child_detection"] and len(people_detections) >= 1:
            child_results = self.child_detector.analyze(people_detections, tracks, frame.shape)
            for child in child_results.get("alert_children", []):
                alert = self.alert_manager.trigger_alert(
                    "lost_child",
                    "HIGH",
                    f"Unaccompanied child detected! Alone for {child.get('alone_frames', 0)} frames",
                    data=child,
                    frame=frame
                )
                if alert:
                    alerts_this_frame.append(alert.to_dict())
        
        # ===== 9. CROWD ANALYSIS =====
        crowd_results = {"count": 0, "density": "LOW", "trend": "stable"}
        if self.features_enabled["crowd_analysis"]:
            crowd_results = self.crowd_analyzer.update(people_detections, frame.shape)
            if crowd_results.get("is_surge"):
                alert = self.alert_manager.trigger_alert(
                    "crowd_surge",
                    "MEDIUM",
                    f"Crowd surge detected! Count: {crowd_results['count']} "
                    f"(peak: {crowd_results['peak_count']})",
                    data=crowd_results,
                    frame=frame
                )
                if alert:
                    alerts_this_frame.append(alert.to_dict())
        
        # ===== 10. HEATMAP =====
        heatmap_image = None
        if self.features_enabled["heatmap"] and self.heatmap_gen is not None:
            self.heatmap_gen.update(people_detections)
            heatmap_image = self.heatmap_gen.get_heatmap_image()
        
        # ===== 11. TARGET TRACKING =====
        target_results = {"target_found": False, "is_active": False}
        if self.features_enabled["target_tracking"]:
            target_results = self.target_tracker.update(
                frame, people_detections, tracks
            )
        
        # ===== 12. DRAW ANNOTATIONS =====
        annotated = frame.copy()
        
        # Draw people boxes and tracks
        if self.draw_options["show_boxes"]:
            if tracks and self.features_enabled["tracking"]:
                draw_tracks(
                    annotated, tracks,
                    show_trail=self.draw_options["show_trails"],
                    show_id=self.draw_options["show_ids"]
                )
        
        # Draw weapon alerts
        for weapon in weapon_results.get("confirmed_weapons", []):
            draw_weapon_alert(annotated, weapon)
        
        # Draw fight alerts
        for fight_pair in fight_results.get("fight_pairs", []):
            draw_fight_alert(annotated, fight_pair)
        
        # Draw child detections
        for child in child_results.get("detected_children", []):
            is_alone = child in child_results.get("alone_children", [])
            draw_child_alert(annotated, child, is_alone)
        
        # Draw stampede warning overlay
        if stampede_results.get("severity") != "NONE":
            draw_stampede_warning(annotated, stampede_results)
        
        # Draw heatmap overlay
        if self.draw_options["show_heatmap"] and self.heatmap_gen is not None:
            annotated = self.heatmap_gen.get_overlay(annotated, alpha=0.4)
        
        # Draw optical flow overlay
        if self.draw_options["show_optical_flow"]:
            annotated = self.stampede_detector.get_flow_overlay(annotated, alpha=0.3)
        
        # Draw target tracking overlay
        if self.features_enabled["target_tracking"]:
            annotated = self.target_tracker.draw_target_overlay(annotated)
        
        # Draw info panel
        if self.draw_options["show_info_panel"]:
            info = {
                "People": crowd_results.get("count", 0),
                "FPS": f"{self.fps:.1f}",
                "Density": crowd_results.get("density", "LOW"),
                "Trend": crowd_results.get("trend", "stable"),
            }
            if is_night:
                info["Mode"] = "Night Vision"
            draw_info_panel(annotated, info)
        
        # ===== CALCULATE FPS =====
        self._update_fps()
        
        # ===== ADAPTIVE FRAME SKIP =====
        # If FPS is too low, increase frame skip; if good, decrease
        if config.ADAPTIVE_SKIP:
            if self.fps > 0 and self.fps < config.FPS_LOW_THRESHOLD:
                self._frame_skip = min(4, self._frame_skip + 1)
            elif self.fps > config.FPS_LOW_THRESHOLD * 1.5:
                self._frame_skip = max(1, self._frame_skip - 1)
        
        # ===== CACHE RESULTS for skipped frames =====
        result = {
            "annotated_frame": annotated,
            "original_frame": original_frame,
            "people_count": len(people_detections),
            "people_detections": people_detections,
            "weapon_results": weapon_results,
            "stampede_results": stampede_results,
            "fight_results": fight_results,
            "child_results": child_results,
            "crowd_results": crowd_results,
            "target_results": target_results,
            "heatmap_image": heatmap_image,
            "alerts": alerts_this_frame,
            "fps": self.fps,
            "brightness": brightness,
            "is_night_mode": is_night,
            "tracks": tracks,
            "frame_count": self._frame_count,
        }
        self._last_results = result
        # Save frame and tracks for target selection from UI
        self._last_frame = frame.copy()
        self._last_tracks = tracks
        return result
    
    def _update_fps(self):
        """Calculate processing FPS."""
        self._fps_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_timer = time.time()
    
    def _empty_results(self):
        """Return empty results dict."""
        return {
            "annotated_frame": None,
            "original_frame": None,
            "people_count": 0,
            "people_detections": [],
            "weapon_results": {"confirmed_weapons": [], "threat_level": "NONE"},
            "stampede_results": {"is_stampede": False, "severity": "NONE"},
            "fight_results": {"is_fight": False, "severity": "NONE", "fight_pairs": []},
            "child_results": {"detected_children": [], "alone_children": []},
            "crowd_results": {"count": 0, "density": "LOW", "trend": "stable"},
            "target_results": {"target_found": False, "is_active": False},
            "heatmap_image": None,
            "alerts": [],
            "fps": 0.0,
            "brightness": 0,
            "is_night_mode": False,
            "tracks": [],
            "frame_count": 0,
        }
    
    def set_feature(self, feature_name, enabled):
        """Enable or disable a feature."""
        if feature_name in self.features_enabled:
            self.features_enabled[feature_name] = enabled
            print(f"[Pipeline] {feature_name}: {'ON' if enabled else 'OFF'}")
    
    def set_draw_option(self, option_name, enabled):
        """Enable or disable a drawing option."""
        if option_name in self.draw_options:
            self.draw_options[option_name] = enabled
    
    def select_target(self, frame, bbox, track_id=None):
        """Select a person as the tracking target."""
        self.target_tracker.set_target(frame, bbox, track_id)
        self.features_enabled["target_tracking"] = True
    
    def clear_target(self):
        """Stop tracking the current target."""
        self.target_tracker.clear_target()
        self.features_enabled["target_tracking"] = False
    
    def reset(self):
        """Reset all modules to initial state."""
        self.tracker.reset()
        self.stampede_detector.reset()
        self.weapon_detector.reset()
        self.fight_detector.reset()
        self.child_detector.reset()
        self.crowd_analyzer.reset()
        if self.heatmap_gen:
            self.heatmap_gen.reset()
        self.target_tracker.clear_target()
        self._frame_count = 0
        print("[Pipeline] Reset complete.")
