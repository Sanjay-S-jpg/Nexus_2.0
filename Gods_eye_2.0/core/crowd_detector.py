# core/crowd_detector.py - Crowd Detection & Counting

import cv2
import numpy as np
from collections import deque


class CrowdDetector:
    """
    Crowd detection using YOLO person detections.
    Counts people, builds density grid, smooths count over time.
    No fake 'hidden people' estimation - shows actual detections only.
    """

    def __init__(self, grid_size=(4, 4)):
        self.grid_size = grid_size
        self.count_history = deque(maxlen=15)

    def count(self, frame, detections=None):
        """
        Count people from YOLO detections.

        Args:
            frame: Video frame (for grid dimensions)
            detections: List of (x, y) center positions from YOLO
        """
        positions = []

        if detections is not None:
            for item in detections:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    positions.append((item[0], item[1]))

        total_count = len(positions)

        # Smooth count over last 15 frames to avoid flicker
        self.count_history.append(total_count)
        smoothed = int(np.mean(self.count_history))

        # Build density grid
        grid = self._get_grid(frame, positions)

        return {
            'count': smoothed,
            'raw_count': total_count,
            'detected': total_count,
            'method': 'YOLO11n',
            'detections': positions,
            'grid': grid,
        }

    def _get_grid(self, frame, positions):
        """Calculate density per grid cell"""
        h, w = frame.shape[:2]
        grid_h, grid_w = self.grid_size
        cell_h, cell_w = h // grid_h, w // grid_w

        grid = np.zeros(self.grid_size, dtype=np.int32)

        for (px, py) in positions:
            gx = min(int(px / cell_w), grid_w - 1)
            gy = min(int(py / cell_h), grid_h - 1)
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                grid[gy, gx] += 1

        return grid

    def draw(self, frame, result, show_all_points=True):
        """Draw detection dots on frame"""
        if show_all_points:
            for (x, y) in result.get('detections', []):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 1)
        return frame