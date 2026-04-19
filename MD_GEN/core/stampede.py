"""
CrowdShield - Stampede Detection Module
=========================================
Detects potential stampede situations using TWO methods:

Method 1: Optical Flow Analysis
  - Calculates how pixels move between consecutive frames
  - A stampede has HIGH magnitude and HIGH uniformity (everyone moving same direction)
  - Normal crowds have random, varied movement directions

Method 2: Density Drop Detection
  - If many people suddenly "disappear" (fall down), their standing bounding boxes vanish
  - A rapid drop in detected people count can indicate trampling

Both methods must agree for a certain number of consecutive frames
before we trigger an alert (to avoid false positives from camera shake etc.)
"""

import cv2
import numpy as np
import time
from collections import deque
import config


class StampedeDetector:
    """
    Detects stampede conditions in a video feed.
    
    Usage:
        detector = StampedeDetector()
        
        # Each frame:
        result = detector.analyze(frame, people_count)
        
        if result["is_stampede"]:
            print("STAMPEDE DETECTED!")
            print(f"Severity: {result['severity']}")
            print(f"Flow direction: {result['flow_direction']}")
    """
    
    def __init__(self):
        # Store previous frame (grayscale) for optical flow
        self.prev_gray = None
        
        # Counter for consecutive stampede-like frames
        self.consec_frames = 0
        
        # Track people count over time for density drop detection
        self.count_history = deque(maxlen=90)  # ~3 seconds at 30fps
        
        # Cooldown timer to avoid spamming alerts
        self.last_alert_time = 0
        
        # Current analysis results
        self.current_flow_mag = 0.0       # Average flow magnitude
        self.current_uniformity = 0.0     # Flow direction uniformity
        self.current_density_drop = 0.0   # Density drop percentage
        self.flow_direction = (0, 0)      # Average flow direction vector
        
        # For visualization
        self.flow_visualization = None
    
    def analyze(self, frame, people_count, tracks=None):
        """
        Analyze a frame for stampede conditions.
        
        Args:
            frame:         BGR image (numpy array)
            people_count:  Number of people detected in this frame
            tracks:        Optional list of Track objects for velocity analysis
        
        Returns:
            dict with keys:
                is_stampede:    bool - True if stampede detected
                severity:       str - "NONE", "WARNING", "CRITICAL"
                flow_magnitude: float - average pixel movement
                uniformity:     float - how aligned the movement is (0-1)
                density_drop:   float - people count drop percentage (0-1)
                flow_direction: tuple - (dx, dy) average flow direction
                consec_frames:  int - consecutive stampede frames
        """
        # Convert current frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Default result
        result = {
            "is_stampede": False,
            "severity": "NONE",
            "flow_magnitude": 0.0,
            "uniformity": 0.0,
            "density_drop": 0.0,
            "flow_direction": (0, 0),
            "consec_frames": self.consec_frames
        }
        
        # Add current count to history
        self.count_history.append(people_count)
        
        # Need previous frame to compute flow
        if self.prev_gray is None:
            self.prev_gray = gray
            return result
        
        # Need minimum people to check stampede
        if people_count < config.STAMPEDE_MIN_PEOPLE:
            self.prev_gray = gray
            self.consec_frames = max(0, self.consec_frames - 1)
            result["consec_frames"] = self.consec_frames
            return result
        
        # ===== METHOD 1: OPTICAL FLOW =====
        flow_mag, uniformity, direction = self._compute_optical_flow(gray)
        self.current_flow_mag = flow_mag
        self.current_uniformity = uniformity
        self.flow_direction = direction
        
        # ===== METHOD 2: DENSITY DROP =====
        density_drop = self._check_density_drop()
        self.current_density_drop = density_drop
        
        # ===== OPTIONAL: TRACK VELOCITY ANALYSIS =====
        track_score = 0
        if tracks and len(tracks) >= config.STAMPEDE_MIN_PEOPLE:
            track_score = self._analyze_track_velocities(tracks)
        
        # ===== COMBINE SIGNALS =====
        flow_alert = (flow_mag > config.STAMPEDE_FLOW_THRESHOLD and 
                      uniformity > config.STAMPEDE_UNIFORMITY_THRESH)
        density_alert = density_drop > config.STAMPEDE_DENSITY_DROP
        track_alert = track_score > 0.6
        
        # Any two of three signals = stampede-like
        signals = sum([flow_alert, density_alert, track_alert])
        
        if signals >= 1:  # At least one strong signal
            self.consec_frames += 1
        else:
            # Slowly decrease counter (don't reset immediately)
            self.consec_frames = max(0, self.consec_frames - 1)
        
        # Determine severity
        severity = "NONE"
        is_stampede = False
        
        if self.consec_frames >= config.STAMPEDE_CONSEC_FRAMES:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_alert_time > config.STAMPEDE_COOLDOWN_SEC:
                is_stampede = True
                self.last_alert_time = current_time
            
            if signals >= 2:
                severity = "CRITICAL"
            else:
                severity = "WARNING"
        elif self.consec_frames >= config.STAMPEDE_CONSEC_FRAMES // 2:
            severity = "WARNING"
        
        # Update visualization
        self._create_flow_visualization(gray)
        
        # Save for next frame
        self.prev_gray = gray
        
        result.update({
            "is_stampede": is_stampede,
            "severity": severity,
            "flow_magnitude": flow_mag,
            "uniformity": uniformity,
            "density_drop": density_drop,
            "flow_direction": direction,
            "consec_frames": self.consec_frames
        })
        
        return result
    
    def _compute_optical_flow(self, gray):
        """
        Compute dense optical flow between previous and current frame.
        
        Optical flow = for each pixel, how far did it move?
        We use Farneback's method which gives a flow vector for every pixel.
        
        Returns:
            (magnitude, uniformity, direction):
                magnitude:  Average flow magnitude (pixels moved)
                uniformity: How aligned all flow vectors are (0=random, 1=same direction)
                direction:  (dx, dy) average flow direction
        """
        # Compute dense optical flow
        # This returns flow[y, x] = (dx, dy) for each pixel
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,                  # Previous flow (None = compute fresh)
            pyr_scale=0.5,         # Pyramid scale
            levels=3,              # Pyramid levels
            winsize=15,            # Window size
            iterations=3,          # Iterations per level
            poly_n=5,              # Polynomial expansion size
            poly_sigma=1.2,        # Gaussian std for polynomial
            flags=0
        )
        
        # Split into x and y components
        fx, fy = flow[..., 0], flow[..., 1]
        
        # Calculate magnitude and angle for each pixel
        magnitude = np.sqrt(fx ** 2 + fy ** 2)
        angle = np.arctan2(fy, fx)  # In radians (-pi to pi)
        
        # Average magnitude (how much movement overall)
        avg_magnitude = np.mean(magnitude)
        
        # Calculate uniformity / alignment of flow
        # If everyone moves the same direction, the average of unit vectors
        # will have a large magnitude (close to 1.0)
        # If random directions, they cancel out (close to 0.0)
        mask = magnitude > 1.0  # Only consider pixels that actually moved
        if np.sum(mask) < 100:  # Too few moving pixels
            return (avg_magnitude, 0.0, (0, 0))
        
        # Compute unit vectors (direction only, magnitude = 1)
        unit_x = np.where(mask, fx / (magnitude + 1e-6), 0)
        unit_y = np.where(mask, fy / (magnitude + 1e-6), 0)
        
        # Average direction
        avg_dx = np.mean(unit_x[mask])
        avg_dy = np.mean(unit_y[mask])
        
        # Uniformity = magnitude of the average direction vector
        uniformity = np.sqrt(avg_dx ** 2 + avg_dy ** 2)
        
        return (avg_magnitude, uniformity, (avg_dx, avg_dy))
    
    def _check_density_drop(self):
        """
        Check if there's a sudden drop in detected people count.
        A stampede often causes people to fall, making them temporarily
        undetectable by the standing person detector.
        
        Returns:
            float: Drop ratio (0.0 = no drop, 1.0 = everyone disappeared)
        """
        if len(self.count_history) < 15:  # Need at least 0.5 seconds of history
            return 0.0
        
        history = list(self.count_history)
        
        # Compare recent count to slightly older count
        recent = np.mean(history[-5:])     # Last 5 frames
        older = np.mean(history[-15:-5])   # 5-15 frames ago
        
        if older <= 0:
            return 0.0
        
        # Calculate drop percentage
        if recent < older:
            drop = (older - recent) / older
            return drop
        
        return 0.0
    
    def _analyze_track_velocities(self, tracks):
        """
        Analyze tracked people's velocities to detect stampede.
        In a stampede, many people move fast in the same direction.
        
        Returns:
            float: Score from 0.0 (no stampede) to 1.0 (clear stampede)
        """
        velocities = []
        for track in tracks:
            if track.lost_frames == 0:
                vx, vy = track.get_velocity()
                speed = np.sqrt(vx ** 2 + vy ** 2)
                if speed > 2.0:  # Only consider moving people
                    velocities.append((vx, vy, speed))
        
        if len(velocities) < config.STAMPEDE_MIN_PEOPLE:
            return 0.0
        
        # Check if most people are moving fast
        speeds = [v[2] for v in velocities]
        avg_speed = np.mean(speeds)
        
        # Check direction uniformity
        dirs = [(v[0] / v[2], v[1] / v[2]) for v in velocities]
        avg_dir_x = np.mean([d[0] for d in dirs])
        avg_dir_y = np.mean([d[1] for d in dirs])
        uniformity = np.sqrt(avg_dir_x ** 2 + avg_dir_y ** 2)
        
        # Combine speed and uniformity into a score
        speed_score = min(1.0, avg_speed / 15.0)  # Normalize
        score = speed_score * uniformity
        
        return score
    
    def _create_flow_visualization(self, gray):
        """
        Create an HSV visualization of the optical flow.
        Hue = direction, Saturation = always full, Value = magnitude.
        
        This is useful for the UI to show what the algorithm "sees".
        """
        if self.prev_gray is None:
            return
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV image
        hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction (0-180)
        hsv[..., 1] = 255                       # Full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
        
        # Convert to BGR for display
        self.flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def get_flow_overlay(self, frame, alpha=0.4):
        """
        Get the frame with optical flow visualization overlaid.
        
        Args:
            frame: Original BGR frame
            alpha: Transparency of the overlay (0=invisible, 1=opaque)
        
        Returns:
            BGR image with flow overlay, or original frame if no flow data
        """
        if self.flow_visualization is None:
            return frame
        
        # Resize flow to match frame if needed
        if self.flow_visualization.shape[:2] != frame.shape[:2]:
            flow_resized = cv2.resize(self.flow_visualization, 
                                       (frame.shape[1], frame.shape[0]))
        else:
            flow_resized = self.flow_visualization
        
        # Blend
        return cv2.addWeighted(frame, 1 - alpha, flow_resized, alpha, 0)
    
    def reset(self):
        """Reset the detector state."""
        self.prev_gray = None
        self.consec_frames = 0
        self.count_history.clear()
        self.last_alert_time = 0
        self.flow_visualization = None
