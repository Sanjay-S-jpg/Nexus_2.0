# core/heatmap.py - Individual Person Heatmap Trails

import cv2
import numpy as np
from collections import deque


class Heatmap:
    """
    Individual vertical heatmap trails per person.
    Each person gets their own colored trail showing movement history.
    """
    
    def __init__(self, width, height, trail_length=80):
        self.width = width
        self.height = height
        self.trail_length = trail_length
        
        # Individual trails per person: track_id -> deque of (x, y) positions
        self.trails = {}
        
        # Colors for different people (cycling)
        self.colors = [
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Magenta
            (255, 255, 0),   # Cyan
            (0, 165, 255),   # Orange
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (128, 0, 255),   # Purple
            (255, 128, 0),   # Light Blue
            (0, 255, 128),   # Spring Green
            (255, 0, 128),   # Pink
        ]
    
    def update(self, positions):
        """
        Update trails with current positions.
        
        Args:
            positions: dict of track_id -> (cx, cy)
        """
        current_ids = set(positions.keys())
        
        # Update existing trails or create new ones
        for track_id, (cx, cy) in positions.items():
            if track_id not in self.trails:
                self.trails[track_id] = deque(maxlen=self.trail_length)
            
            self.trails[track_id].append((int(cx), int(cy)))
        
        # Remove old trails for people no longer visible
        old_ids = set(self.trails.keys()) - current_ids
        for old_id in old_ids:
            # Keep trail for a bit, then fade out
            if len(self.trails[old_id]) > 0:
                # Gradually reduce trail
                self.trails[old_id].popleft()
            if len(self.trails[old_id]) == 0:
                del self.trails[old_id]
    
    def get_color(self, track_id):
        """Get consistent color for a track ID"""
        return self.colors[track_id % len(self.colors)]
    
    def draw_trails(self, frame):
        """Draw individual heatmap trails for each person"""
        for track_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            
            color = self.get_color(track_id)
            points = list(trail)
            
            # Draw gradient trail (fades from old to new)
            for i in range(1, len(points)):
                # Alpha based on position in trail (newer = brighter)
                alpha = i / len(points)
                
                # Calculate color with fade
                r = int(color[0] * alpha)
                g = int(color[1] * alpha)
                b = int(color[2] * alpha)
                
                pt1 = points[i - 1]
                pt2 = points[i]
                
                # Line thickness increases towards current position
                thickness = max(1, int(3 * alpha))
                
                cv2.line(frame, pt1, pt2, (r, g, b), thickness)
            
            # Draw heat glow at current position
            if len(points) > 0:
                current = points[-1]
                
                # Glowing effect (multiple circles with decreasing alpha)
                overlay = frame.copy()
                cv2.circle(overlay, current, 20, color, -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                cv2.circle(overlay, current, 12, color, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Solid center dot
                cv2.circle(frame, current, 6, color, -1)
                cv2.circle(frame, current, 8, (255, 255, 255), 1)
        
        return frame
    
    def overlay(self, frame, alpha=0.5):
        """Overlay trails on frame (for heatmap mode)"""
        return self.draw_trails(frame)
    
    def get_density_overlay(self, frame):
        """Get cumulative density visualization"""
        # Create density map from all trail points
        density = np.zeros((self.height, self.width), dtype=np.float32)
        
        for trail in self.trails.values():
            for (x, y) in trail:
                if 0 <= x < self.width and 0 <= y < self.height:
                    cv2.circle(density, (x, y), 15, 1.0, -1)
        
        # Blur for smooth effect
        density = cv2.GaussianBlur(density, (31, 31), 0)
        
        # Normalize
        if density.max() > 0:
            density = density / density.max()
        
        # Apply colormap
        density_8bit = (density * 255).astype(np.uint8)
        colored = cv2.applyColorMap(density_8bit, cv2.COLORMAP_JET)
        
        # Blend
        return cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
    
    def reset(self):
        """Clear all trails"""
        self.trails.clear()
    
    def save(self, filepath, background=None):
        """Save heatmap to file"""
        if background is not None:
            result = self.draw_trails(background.copy())
        else:
            result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            result = self.draw_trails(result)
        
        cv2.imwrite(filepath, result)
        return filepath