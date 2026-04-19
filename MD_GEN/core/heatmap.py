"""
CrowdShield - Heatmap Generator
=================================
Creates a heatmap showing where people spend the most time.

How it works:
  1. Each frame, we add a "heat blob" at each person's position
  2. Over time, areas where people stand/walk frequently get "hotter"
  3. The heatmap slowly "cools down" (decay) so it shows recent activity
  4. Colors: Blue = cool (few people) → Red = hot (many people)

This is useful for:
  - Finding bottlenecks in crowd flow
  - Identifying popular/crowded areas
  - Predicting where stampedes are most likely
"""

import cv2
import numpy as np
import config


class HeatmapGenerator:
    """
    Generates and maintains a running heatmap of person positions.
    
    Usage:
        heatmap_gen = HeatmapGenerator(frame_width=1280, frame_height=720)
        
        # Each frame:
        heatmap_gen.update(people_detections)
        
        # Get colored heatmap overlay
        overlay = heatmap_gen.get_overlay(original_frame, alpha=0.5)
        
        # Get standalone heatmap image
        heatmap_image = heatmap_gen.get_heatmap_image()
    """
    
    def __init__(self, frame_width=1280, frame_height=720):
        """
        Initialize the heatmap.
        
        Args:
            frame_width:  Width of the video frames
            frame_height: Height of the video frames
        """
        self.width = frame_width
        self.height = frame_height
        
        # The "heat" accumulator - a float array where each pixel
        # stores the accumulated heat value
        self.heat_map = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Settings from config
        self.decay = config.HEATMAP_DECAY
        self.intensity = config.HEATMAP_INTENSITY
        self.radius = config.HEATMAP_RADIUS
        self.colormap = config.HEATMAP_COLORMAP
        
        self.frame_count = 0
    
    def update(self, people_detections):
        """
        Add heat at each detected person's position.
        
        Args:
            people_detections: list of Detection objects (people)
        """
        # First, decay the existing heat (so old positions fade)
        self.heat_map *= self.decay
        
        # Add heat blob at each person's foot position
        # (Using bottom-center of bbox as the foot position is more accurate
        #  than center, since a person "occupies" the ground, not the air)
        for det in people_detections:
            x1, y1, x2, y2 = det.bbox
            
            # Bottom-center of bounding box (approximate foot position)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)  # Bottom of bounding box
            
            # Also add some heat at the center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Add Gaussian heat blob at foot position
            self._add_heat_blob(foot_x, foot_y, self.intensity)
            self._add_heat_blob(center_x, center_y, self.intensity * 0.5)
        
        self.frame_count += 1
    
    def _add_heat_blob(self, cx, cy, intensity):
        """
        Add a Gaussian (bell-curve shaped) heat blob at position (cx, cy).
        
        The blob looks like a circular gradient:
        - Hottest at the center
        - Fades to zero at the edges
        
        Args:
            cx, cy:    Center position
            intensity: How much heat to add (higher = hotter)
        """
        # Calculate the area to modify (clipped to image bounds)
        r = self.radius
        x_start = max(0, cx - r)
        x_end = min(self.width, cx + r)
        y_start = max(0, cy - r)
        y_end = min(self.height, cy + r)
        
        if x_start >= x_end or y_start >= y_end:
            return
        
        # Create coordinate grids for the patch
        y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]
        
        # Calculate distance from center
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        
        # Gaussian formula: intensity * e^(-distance²/(2*σ²))
        sigma = r / 3.0
        gaussian = intensity * np.exp(-dist_sq / (2 * sigma ** 2))
        
        # Add to the heat map
        self.heat_map[y_start:y_end, x_start:x_end] += gaussian
    
    def get_heatmap_image(self):
        """
        Convert the heat map to a colored image.
        
        Returns:
            BGR image (numpy array) with the heatmap colors
        """
        # Normalize to 0-255 range
        if self.heat_map.max() > 0:
            normalized = np.clip(self.heat_map / self.heat_map.max() * 255, 0, 255)
        else:
            normalized = self.heat_map
        
        # Convert to uint8
        heat_uint8 = normalized.astype(np.uint8)
        
        # Apply colormap (JET: blue → green → yellow → red)
        colored = cv2.applyColorMap(heat_uint8, self.colormap)
        
        return colored
    
    def get_overlay(self, frame, alpha=0.5):
        """
        Overlay the heatmap on top of the original frame.
        
        Args:
            frame: Original BGR frame
            alpha: Transparency (0=invisible, 1=opaque heatmap)
        
        Returns:
            BGR image with heatmap overlay
        """
        # Resize heat to match frame if needed
        if self.heat_map.shape[:2] != frame.shape[:2]:
            self.resize(frame.shape[1], frame.shape[0])
        
        heatmap_colored = self.get_heatmap_image()
        
        # Only overlay where there's actual heat (avoids blue tint everywhere)
        mask = self.heat_map > (self.heat_map.max() * 0.05) if self.heat_map.max() > 0 else np.zeros_like(self.heat_map, dtype=bool)
        
        result = frame.copy()
        if np.any(mask):
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            result = np.where(
                mask_3ch,
                cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0),
                frame
            )
        
        return result
    
    def resize(self, new_width, new_height):
        """
        Resize the heatmap to match a different frame size.
        
        Args:
            new_width, new_height: New dimensions
        """
        if new_width != self.width or new_height != self.height:
            self.heat_map = cv2.resize(
                self.heat_map, (new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )
            self.width = new_width
            self.height = new_height
    
    def get_hotspots(self, top_n=5):
        """
        Find the hottest areas in the heatmap.
        
        Args:
            top_n: Number of hotspots to return
        
        Returns:
            list of (x, y, intensity) tuples for the hottest points
        """
        if self.heat_map.max() == 0:
            return []
        
        # Blur to find broad hotspots (not individual pixel peaks)
        blurred = cv2.GaussianBlur(self.heat_map, (51, 51), 0)
        
        hotspots = []
        temp = blurred.copy()
        
        for _ in range(top_n):
            max_val = temp.max()
            if max_val < blurred.max() * 0.1:
                break
            
            # Find the peak
            max_loc = np.unravel_index(temp.argmax(), temp.shape)
            y, x = max_loc
            
            hotspots.append((int(x), int(y), float(max_val)))
            
            # Suppress this area so we find the next peak
            cv2.circle(temp, (int(x), int(y)), self.radius * 3, 0, -1)
        
        return hotspots
    
    def reset(self):
        """Clear the heatmap."""
        self.heat_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.frame_count = 0
