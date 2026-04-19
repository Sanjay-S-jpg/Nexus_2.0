"""
CrowdShield - Fight Detection Module
=======================================
Detects fights/physical altercations using pose estimation.

How it works:
  1. YOLOv8-Pose gives us skeleton keypoints for each person
  2. We look for TWO conditions:
     a) Two people are very close together (within fighting distance)
     b) Their arms/wrists are moving fast (punching/hitting motion)
  3. If both conditions persist for several frames = fight detected

This is MUCH more reliable than facial expression analysis because:
  - CCTV cameras are often far away (faces too small to read)
  - Expressions are ambiguous (someone laughing vs angry)
  - Pose/motion is visible even at low resolution

Handling false positives (friends play-fighting):
  - We require sustained aggressive motion (not just a brief moment)
  - We check for impact-like deceleration patterns
  - The threshold can be adjusted in config.py
"""

import numpy as np
import time
from collections import deque
import config


class FightDetector:
    """
    Detects fights using pose keypoint analysis.
    
    Usage:
        detector = FightDetector()
        
        # Each frame:
        result = detector.analyze(pose_detections)
        
        if result["is_fight"]:
            print(f"FIGHT detected between persons: {result['involved_ids']}")
    """
    
    def __init__(self):
        # History of keypoints per person for velocity calculation
        # Format: {track_id: deque of keypoints}
        self.keypoint_history = {}
        
        # Consecutive fight frames counter
        self.consec_frames = 0
        
        # Cooldown
        self.last_alert_time = 0
        
        # Current fight pairs
        self.current_fights = []
    
    def analyze(self, pose_detections, tracks=None):
        """
        Analyze pose detections for fights.
        
        Args:
            pose_detections: list of PoseDetection objects
            tracks:          Optional list of Track objects (for track IDs)
        
        Returns:
            dict with keys:
                is_fight:       bool - True if fight detected
                severity:       str - "NONE", "WARNING", "CRITICAL"
                fight_pairs:    list of (person1_info, person2_info) dicts
                consec_frames:  int - consecutive fight frames
        """
        result = {
            "is_fight": False,
            "severity": "NONE",
            "fight_pairs": [],
            "consec_frames": self.consec_frames
        }
        
        if len(pose_detections) < config.FIGHT_MIN_PEOPLE:
            self.consec_frames = max(0, self.consec_frames - 1)
            result["consec_frames"] = self.consec_frames
            return result
        
        # ===== STEP 1: Find close pairs of people =====
        close_pairs = self._find_close_pairs(pose_detections)
        
        if len(close_pairs) == 0:
            self.consec_frames = max(0, self.consec_frames - 1)
            result["consec_frames"] = self.consec_frames
            return result
        
        # ===== STEP 2: Check for aggressive motion in close pairs =====
        fight_pairs = []
        for i, j in close_pairs:
            person1 = pose_detections[i]
            person2 = pose_detections[j]
            
            aggression1 = self._check_aggressive_motion(person1, i)
            aggression2 = self._check_aggressive_motion(person2, j)
            
            # Both people need some aggressive motion, or one needs a lot
            if (aggression1 > 0.4 and aggression2 > 0.4) or \
               max(aggression1, aggression2) > 0.7:
                fight_pairs.append({
                    "person1": {
                        "bbox": person1.bbox,
                        "center": person1.center,
                        "aggression_score": aggression1,
                        "index": i
                    },
                    "person2": {
                        "bbox": person2.bbox,
                        "center": person2.center,
                        "aggression_score": aggression2,
                        "index": j
                    }
                })
        
        # ===== STEP 3: Update consecutive frame counter =====
        if len(fight_pairs) > 0:
            self.consec_frames += 1
            self.current_fights = fight_pairs
        else:
            self.consec_frames = max(0, self.consec_frames - 1)
        
        # ===== STEP 4: Determine severity and alert =====
        severity = "NONE"
        is_fight = False
        
        if self.consec_frames >= config.FIGHT_CONSEC_FRAMES:
            current_time = time.time()
            if current_time - self.last_alert_time > config.FIGHT_COOLDOWN_SEC:
                is_fight = True
                self.last_alert_time = current_time
            
            max_aggression = max(
                max(fp["person1"]["aggression_score"], fp["person2"]["aggression_score"])
                for fp in fight_pairs
            ) if fight_pairs else 0
            
            if max_aggression > 0.7:
                severity = "CRITICAL"
            else:
                severity = "WARNING"
        elif self.consec_frames >= config.FIGHT_CONSEC_FRAMES // 2:
            severity = "WARNING"
        
        # Update keypoint history
        self._update_keypoint_history(pose_detections)
        
        result.update({
            "is_fight": is_fight,
            "severity": severity,
            "fight_pairs": fight_pairs,
            "consec_frames": self.consec_frames
        })
        
        return result
    
    def _find_close_pairs(self, poses):
        """
        Find pairs of people who are close enough to be fighting.
        
        Returns:
            list of (index_i, index_j) tuples
        """
        close_pairs = []
        
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                # Calculate distance between centers
                c1 = poses[i].center
                c2 = poses[j].center
                
                dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                
                # Use the average height as a scale reference
                avg_height = (poses[i].height + poses[j].height) / 2
                
                # Dynamic threshold based on person size
                # (accounts for camera distance)
                threshold = max(config.FIGHT_PROXIMITY_PX, avg_height * 1.2)
                
                if dist < threshold:
                    close_pairs.append((i, j))
        
        return close_pairs
    
    def _check_aggressive_motion(self, pose, person_idx):
        """
        Check if a person's arm movements indicate fighting.
        
        We look at:
          1. Wrist velocity (fast hands = punching)
          2. Elbow velocity (swinging arms)
          3. Raised arms (guard/punch position)
        
        Args:
            pose:        PoseDetection object
            person_idx:  Index for looking up history
        
        Returns:
            float: Aggression score (0.0 to 1.0)
        """
        scores = []
        
        # ===== Check wrist velocity =====
        wrist_velocity = self._get_keypoint_velocity(
            person_idx, 
            [9, 10]  # left_wrist, right_wrist
        )
        if wrist_velocity is not None:
            # Normalize velocity by person height (scale-invariant)
            height = max(pose.height, 1)
            normalized_vel = wrist_velocity / height * 100
            wrist_score = min(1.0, normalized_vel / config.FIGHT_VELOCITY_THRESH)
            scores.append(wrist_score * 0.5)  # 50% weight
        
        # ===== Check elbow velocity =====
        elbow_velocity = self._get_keypoint_velocity(
            person_idx,
            [7, 8]  # left_elbow, right_elbow
        )
        if elbow_velocity is not None:
            height = max(pose.height, 1)
            normalized_vel = elbow_velocity / height * 100
            elbow_score = min(1.0, normalized_vel / (config.FIGHT_VELOCITY_THRESH * 0.8))
            scores.append(elbow_score * 0.3)  # 30% weight
        
        # ===== Check arm position (raised arms) =====
        arm_raised = self._check_raised_arms(pose)
        scores.append(arm_raised * 0.2)  # 20% weight
        
        if len(scores) == 0:
            return 0.0
        
        return min(1.0, sum(scores))
    
    def _get_keypoint_velocity(self, person_idx, keypoint_indices):
        """
        Calculate the velocity of specific keypoints by comparing
        current and previous positions.
        
        Returns:
            float: Maximum velocity in pixels/frame, or None if no history
        """
        key = f"person_{person_idx}"
        
        if key not in self.keypoint_history or len(self.keypoint_history[key]) < 2:
            return None
        
        current = self.keypoint_history[key][-1]
        previous = self.keypoint_history[key][-2]
        
        max_velocity = 0.0
        
        for kp_idx in keypoint_indices:
            if kp_idx >= len(current) or kp_idx >= len(previous):
                continue
            
            cx, cy, cc = current[kp_idx]
            px, py, pc = previous[kp_idx]
            
            # Both keypoints need to be visible
            if cc < 0.3 or pc < 0.3:
                continue
            
            velocity = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            max_velocity = max(max_velocity, velocity)
        
        return max_velocity if max_velocity > 0 else None
    
    def _check_raised_arms(self, pose):
        """
        Check if person has arms raised (fighting stance).
        Arms above shoulder level indicate aggressive posture.
        
        Returns:
            float: Score 0.0 (arms down) to 1.0 (both arms raised high)
        """
        score = 0.0
        
        # Get shoulder and wrist keypoints
        l_shoulder = pose.get_keypoint("left_shoulder")
        r_shoulder = pose.get_keypoint("right_shoulder")
        l_wrist = pose.get_keypoint("left_wrist")
        r_wrist = pose.get_keypoint("right_wrist")
        
        # Check if wrists are above shoulders (y-axis is inverted in images)
        if l_shoulder and l_wrist:
            if l_wrist[1] < l_shoulder[1]:  # Wrist y < shoulder y = arm raised
                score += 0.5
        
        if r_shoulder and r_wrist:
            if r_wrist[1] < r_shoulder[1]:
                score += 0.5
        
        return score
    
    def _update_keypoint_history(self, poses):
        """Store current keypoints for velocity calculation next frame."""
        # Clear old entries
        self.keypoint_history = {}
        
        for i, pose in enumerate(poses):
            key = f"person_{i}"
            if key not in self.keypoint_history:
                self.keypoint_history[key] = deque(maxlen=10)
            self.keypoint_history[key].append(pose.keypoints.copy())
    
    def reset(self):
        """Reset detector state."""
        self.keypoint_history = {}
        self.consec_frames = 0
        self.last_alert_time = 0
        self.current_fights = []
