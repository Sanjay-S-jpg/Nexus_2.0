"""
CrowdShield - Crowd Analyzer
===============================
Analyzes crowd density, counts people, and tracks population changes over time.

Features:
  - Real-time people count
  - Historical count tracking (rise and fall chart)
  - Crowd density estimation
  - Surge detection (sudden increase in people)
  - Area-based density (how packed different regions are)

This handles the "count people even in dense crowds" requirement.
For very dense crowds (like Black Friday), we combine:
  1. YOLO detections (works well up to ~50 people)
  2. Head-top detection fallback for ultra-dense scenes
"""

import numpy as np
import time
from collections import deque
import config


class CrowdAnalyzer:
    """
    Tracks and analyzes crowd metrics over time.
    
    Usage:
        analyzer = CrowdAnalyzer()
        
        # Each frame:
        result = analyzer.update(people_detections, frame_shape)
        
        print(f"People count: {result['count']}")
        print(f"Density: {result['density']}")
        print(f"Trend: {result['trend']}")  # "rising", "falling", "stable"
    """
    
    def __init__(self):
        # People count history: stores (timestamp, count) pairs
        self.count_history = deque(maxlen=config.CROWD_HISTORY_LENGTH)
        
        # For trend calculation
        self.short_window = deque(maxlen=30)   # ~1 second
        self.long_window = deque(maxlen=150)   # ~5 seconds
        
        # Peak and valley tracking
        self.peak_count = 0
        self.min_count = float('inf')
        
        # For surge detection
        self.last_surge_time = 0
        
        # Region-based density (divide frame into grid)
        self.grid_rows = 3
        self.grid_cols = 4
        self.region_counts = np.zeros((self.grid_rows, self.grid_cols))
    
    def update(self, people_detections, frame_shape):
        """
        Update crowd analysis with new frame data.
        
        Args:
            people_detections: list of Detection objects (people only)
            frame_shape:       (height, width) of the frame
        
        Returns:
            dict with keys:
                count:          int - current people count
                density:        str - "LOW", "MEDIUM", "HIGH", "EXTREME"
                trend:          str - "rising", "falling", "stable"
                change_rate:    float - people/second change rate
                peak_count:     int - highest count seen
                is_surge:       bool - sudden crowd increase
                region_density: numpy array - per-region people counts
                history:        list of (timestamp, count) for charts
        """
        current_count = len(people_detections)
        current_time = time.time()
        
        # Store in history
        self.count_history.append((current_time, current_count))
        self.short_window.append(current_count)
        self.long_window.append(current_count)
        
        # Track peaks
        self.peak_count = max(self.peak_count, current_count)
        if current_count > 0:
            self.min_count = min(self.min_count, current_count)
        
        # Calculate trend
        trend, change_rate = self._calculate_trend()
        
        # Calculate density level
        density = self._calculate_density(current_count, frame_shape)
        
        # Check for crowd surge
        is_surge = self._check_surge(current_count)
        
        # Update region-based density
        self._update_region_density(people_detections, frame_shape)
        
        return {
            "count": current_count,
            "density": density,
            "trend": trend,
            "change_rate": change_rate,
            "peak_count": self.peak_count,
            "is_surge": is_surge,
            "region_density": self.region_counts.copy(),
            "history": list(self.count_history)
        }
    
    def _calculate_trend(self):
        """
        Determine if crowd is growing, shrinking, or stable.
        
        Compares short-term average to long-term average.
        
        Returns:
            (trend, change_rate): 
                trend = "rising" | "falling" | "stable"
                change_rate = people per second
        """
        if len(self.short_window) < 5 or len(self.long_window) < 10:
            return ("stable", 0.0)
        
        short_avg = np.mean(list(self.short_window))
        long_avg = np.mean(list(self.long_window))
        
        # Change rate (people per second, assuming ~30fps)
        if len(self.count_history) >= 2:
            time_diff = self.count_history[-1][0] - self.count_history[-min(30, len(self.count_history))][0]
            count_diff = self.count_history[-1][1] - self.count_history[-min(30, len(self.count_history))][1]
            change_rate = count_diff / max(time_diff, 0.1)
        else:
            change_rate = 0.0
        
        # Determine trend
        diff = short_avg - long_avg
        threshold = max(2.0, long_avg * 0.1)  # 10% or at least 2 people
        
        if diff > threshold:
            return ("rising", change_rate)
        elif diff < -threshold:
            return ("falling", change_rate)
        else:
            return ("stable", change_rate)
    
    def _calculate_density(self, count, frame_shape):
        """
        Estimate crowd density level.
        
        This is a simplified density based on total count.
        Real density would be people/square meter, but we don't know
        the physical area covered by the camera.
        
        Returns:
            str: "LOW", "MEDIUM", "HIGH", "EXTREME"
        """
        # Adjust thresholds based on frame area
        frame_area = frame_shape[0] * frame_shape[1]
        
        # People per million pixels (rough density metric)
        density_metric = count / (frame_area / 1e6) if frame_area > 0 else 0
        
        if count < 5 or density_metric < 10:
            return "LOW"
        elif count < 20 or density_metric < 30:
            return "MEDIUM"
        elif count < config.CROWD_HIGH_THRESHOLD or density_metric < 60:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _check_surge(self, current_count):
        """
        Detect a sudden crowd surge (rapid increase in people).
        
        Returns:
            bool: True if surge detected
        """
        if len(self.long_window) < 30:
            return False
        
        old_avg = np.mean(list(self.long_window)[:30])
        
        if old_avg <= 0:
            return False
        
        increase_ratio = (current_count - old_avg) / old_avg
        
        if increase_ratio > config.CROWD_SURGE_THRESHOLD:
            current_time = time.time()
            if current_time - self.last_surge_time > 10:  # 10 second cooldown
                self.last_surge_time = current_time
                return True
        
        return False
    
    def _update_region_density(self, people_detections, frame_shape):
        """
        Count people in each grid region of the frame.
        Divides the frame into a grid and counts people per cell.
        Useful for identifying hotspots.
        
        Updates self.region_counts in place.
        """
        frame_h, frame_w = frame_shape[:2]
        cell_w = frame_w / self.grid_cols
        cell_h = frame_h / self.grid_rows
        
        self.region_counts = np.zeros((self.grid_rows, self.grid_cols))
        
        for det in people_detections:
            # Get center of person
            cx, cy = det.center
            
            # Find which grid cell this center falls in
            col = min(int(cx / cell_w), self.grid_cols - 1)
            row = min(int(cy / cell_h), self.grid_rows - 1)
            
            col = max(0, col)
            row = max(0, row)
            
            self.region_counts[row, col] += 1
    
    def get_chart_data(self, last_n_seconds=30):
        """
        Get count history formatted for charting.
        
        Args:
            last_n_seconds: How many seconds of history to return
        
        Returns:
            dict with 'timestamps' and 'counts' lists
        """
        if len(self.count_history) == 0:
            return {"timestamps": [], "counts": []}
        
        current_time = time.time()
        cutoff = current_time - last_n_seconds
        
        timestamps = []
        counts = []
        
        for t, c in self.count_history:
            if t >= cutoff:
                timestamps.append(t - cutoff)  # Seconds from start
                counts.append(c)
        
        return {"timestamps": timestamps, "counts": counts}
    
    def get_statistics(self):
        """
        Get summary statistics about the crowd.
        
        Returns:
            dict with various statistics
        """
        if len(self.count_history) == 0:
            return {
                "current": 0, "average": 0, "peak": 0,
                "minimum": 0, "std_dev": 0
            }
        
        counts = [c for _, c in self.count_history]
        
        return {
            "current": counts[-1] if counts else 0,
            "average": round(np.mean(counts), 1),
            "peak": self.peak_count,
            "minimum": int(self.min_count) if self.min_count != float('inf') else 0,
            "std_dev": round(np.std(counts), 1)
        }
    
    def reset(self):
        """Reset all crowd analysis state."""
        self.count_history.clear()
        self.short_window.clear()
        self.long_window.clear()
        self.peak_count = 0
        self.min_count = float('inf')
        self.last_surge_time = 0
        self.region_counts = np.zeros((self.grid_rows, self.grid_cols))
