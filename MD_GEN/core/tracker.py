"""
CrowdShield - Multi-Object Tracker
====================================
Tracks people across video frames using a simplified ByteTrack approach.
Each person gets a unique ID that persists as they move through the scene.

How it works:
  1. Each frame, we get new detections from YOLO
  2. We match new detections to existing tracks using IoU (how much boxes overlap)
  3. Matched tracks update their position
  4. Unmatched detections become new tracks
  5. Lost tracks (not matched for a while) get removed

This is simpler than full ByteTrack but works well for our demo.
"""

import numpy as np
from collections import deque
import config


class Track:
    """
    Represents a single tracked person/object.
    
    Attributes:
        track_id:    Unique ID number for this track
        bbox:        Current [x1, y1, x2, y2] bounding box
        confidence:  Detection confidence
        age:         How many frames this track has existed
        hits:        How many times it was matched with a detection
        lost_frames: How many consecutive frames since last match
        history:     Past positions (for drawing trails)
    """
    _next_id = 1  # Class variable to auto-increment IDs
    
    def __init__(self, bbox, confidence, class_id=0):
        self.track_id = Track._next_id
        Track._next_id += 1
        
        self.bbox = list(bbox)              # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.age = 1                        # Total frames existed
        self.hits = 1                       # Times matched
        self.lost_frames = 0                # Consecutive frames without match
        
        # Store history of center positions (for trails/velocity)
        self.history = deque(maxlen=60)     # Last 60 positions (~2 seconds at 30fps)
        cx, cy = self.center
        self.history.append((cx, cy))
    
    @property
    def center(self):
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def height(self):
        """Height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def width(self):
        """Width of the bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    def update(self, bbox, confidence):
        """
        Update this track with a new matched detection.
        
        Args:
            bbox:       New [x1, y1, x2, y2]
            confidence: New detection confidence
        """
        self.bbox = list(bbox)
        self.confidence = confidence
        self.hits += 1
        self.lost_frames = 0
        self.age += 1
        
        cx, cy = self.center
        self.history.append((cx, cy))
    
    def mark_lost(self):
        """Mark this track as not matched this frame."""
        self.lost_frames += 1
        self.age += 1
    
    def get_velocity(self):
        """
        Calculate the velocity (pixels/frame) from recent positions.
        
        Returns:
            (vx, vy) velocity vector, or (0, 0) if not enough history
        """
        if len(self.history) < 2:
            return (0.0, 0.0)
        
        # Average velocity over last 5 frames for stability
        n = min(5, len(self.history))
        positions = list(self.history)[-n:]
        
        vx = (positions[-1][0] - positions[0][0]) / (n - 1)
        vy = (positions[-1][1] - positions[0][1]) / (n - 1)
        
        return (vx, vy)
    
    def get_speed(self):
        """Get the scalar speed (pixels/frame)."""
        vx, vy = self.get_velocity()
        return np.sqrt(vx ** 2 + vy ** 2)


def compute_iou(bbox1, bbox2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    IoU measures how much two boxes overlap:
      - 0.0 = no overlap at all
      - 1.0 = perfect overlap (identical boxes)
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU value (0.0 to 1.0)
    """
    # Find the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_iou_matrix(tracks, detections):
    """
    Compute IoU between all tracks and all detections.
    
    Args:
        tracks:     list of Track objects
        detections: list of Detection objects
    
    Returns:
        numpy array of shape (len(tracks), len(detections)) with IoU values
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    
    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets))
    
    iou_matrix = np.zeros((n_tracks, n_dets))
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = compute_iou(track.bbox, det.bbox)
    
    return iou_matrix


class MultiObjectTracker:
    """
    Tracks multiple objects across frames.
    
    Usage:
        tracker = MultiObjectTracker()
        
        # Each frame:
        tracks = tracker.update(detections)
        
        for track in tracks:
            print(f"Person #{track.track_id} at {track.center}")
    """
    
    def __init__(self):
        self.tracks = []                           # Active tracks
        self.match_thresh = config.TRACKER_MATCH_THRESH
        self.max_lost = config.TRACKER_BUFFER       # Max frames without match
    
    def update(self, detections):
        """
        Update tracker with new frame detections.
        
        This is called every frame with the new YOLO detections.
        It matches detections to existing tracks, creates new tracks,
        and removes old ones.
        
        Args:
            detections: list of Detection objects from YOLODetector
        
        Returns:
            list of active Track objects (each has a unique track_id)
        """
        if len(detections) == 0:
            # No detections - mark all tracks as lost
            for track in self.tracks:
                track.mark_lost()
            # Remove tracks that have been lost too long
            self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]
            return self.tracks
        
        if len(self.tracks) == 0:
            # No existing tracks - create new ones for all detections
            for det in detections:
                new_track = Track(det.bbox, det.confidence, det.class_id)
                det.track_id = new_track.track_id
                self.tracks.append(new_track)
            return self.tracks
        
        # ----- MATCHING STEP -----
        # Compute IoU matrix between all tracks and detections
        iou_matrix = compute_iou_matrix(self.tracks, detections)
        
        # Greedy matching: assign each detection to the best track
        matched_tracks = set()
        matched_dets = set()
        
        # Find matches in order of highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            
            # Find the highest IoU
            max_iou = np.max(iou_matrix)
            if max_iou < self.match_thresh:
                break  # No more good matches
            
            # Get row (track) and column (detection) of the best match
            track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Update the track with the matched detection
            self.tracks[track_idx].update(
                detections[det_idx].bbox,
                detections[det_idx].confidence
            )
            detections[det_idx].track_id = self.tracks[track_idx].track_id
            
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            
            # Remove this match from consideration
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        # ----- HANDLE UNMATCHED -----
        # Unmatched tracks = lost this frame
        for i in range(len(self.tracks)):
            if i not in matched_tracks:
                self.tracks[i].mark_lost()
        
        # Unmatched detections = new objects entering the scene
        for j in range(len(detections)):
            if j not in matched_dets:
                new_track = Track(
                    detections[j].bbox,
                    detections[j].confidence,
                    detections[j].class_id
                )
                detections[j].track_id = new_track.track_id
                self.tracks.append(new_track)
        
        # Remove tracks that have been lost too long
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]
        
        return self.tracks
    
    def get_active_tracks(self):
        """Get only tracks that were matched recently (not lost)."""
        return [t for t in self.tracks if t.lost_frames == 0]
    
    def get_track_by_id(self, track_id):
        """Find a track by its ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_all_velocities(self):
        """Get velocities of all active tracks."""
        return {
            t.track_id: t.get_velocity()
            for t in self.get_active_tracks()
        }
    
    def reset(self):
        """Clear all tracks and reset ID counter."""
        self.tracks = []
        Track._next_id = 1
