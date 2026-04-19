"""
CrowdShield - YOLO Object Detector
====================================
Core detection module using YOLOv8 (Ultralytics).
Detects people, weapons, and other objects in video frames.

YOLOv8 is a state-of-the-art object detection model that can:
  - Detect 80+ object types (COCO dataset)
  - Estimate body pose/skeleton (keypoints)
  - Run on GPU for real-time speed

This module wraps YOLOv8 so other parts of CrowdShield can use it easily.
"""

import numpy as np
from ultralytics import YOLO
import config


class Detection:
    """
    Holds one detected object's information.
    
    Attributes:
        bbox:       [x1, y1, x2, y2] - bounding box coordinates (top-left, bottom-right)
        confidence: float (0.0 to 1.0) - how confident the model is
        class_id:   int - COCO class ID (0=person, 43=knife, etc.)
        class_name: str - human readable name ("person", "knife", etc.)
    """
    def __init__(self, bbox, confidence, class_id, class_name):
        self.bbox = bbox              # [x1, y1, x2, y2]
        self.confidence = confidence  # 0.0 - 1.0
        self.class_id = class_id      # COCO class ID
        self.class_name = class_name  # "person", "knife", etc.
        self.track_id = None          # Assigned by tracker later
    
    @property
    def center(self):
        """Get the center point (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self):
        """Width of the bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self):
        """Height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self):
        """Area of the bounding box in pixels."""
        return self.width * self.height


class PoseDetection:
    """
    Holds one person's pose detection result.
    
    Attributes:
        bbox:       [x1, y1, x2, y2] - bounding box of the person
        confidence: float - detection confidence
        keypoints:  numpy array of shape (17, 3) - [x, y, confidence] for each keypoint
        
    Keypoint indices (COCO format):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    # Keypoint names for reference
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, bbox, confidence, keypoints):
        self.bbox = bbox
        self.confidence = confidence
        self.keypoints = keypoints  # shape: (17, 3) -> [x, y, conf]
        self.track_id = None
    
    @property
    def center(self):
        """Center of the person's bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def height(self):
        """Height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    def get_keypoint(self, name_or_index):
        """
        Get a specific keypoint by name or index.
        
        Args:
            name_or_index: Either a string like "left_wrist" or int like 9
        
        Returns:
            (x, y, confidence) tuple, or None if not detected
        """
        if isinstance(name_or_index, str):
            if name_or_index in self.KEYPOINT_NAMES:
                idx = self.KEYPOINT_NAMES.index(name_or_index)
            else:
                return None
        else:
            idx = name_or_index
        
        if idx < 0 or idx >= len(self.keypoints):
            return None
        
        x, y, conf = self.keypoints[idx]
        if conf < 0.3:  # Keypoint not visible enough
            return None
        return (float(x), float(y), float(conf))


class YOLODetector:
    """
    Main YOLOv8 detection wrapper for CrowdShield.
    
    Usage:
        detector = YOLODetector()
        
        # Detect people and objects
        detections = detector.detect(frame)
        
        # Detect people only  
        people = detector.detect_people(frame)
        
        # Detect with pose estimation
        poses = detector.detect_poses(frame)
    """
    
    def __init__(self):
        """Initialize the YOLO models. They auto-download on first use."""
        print("[YOLODetector] Loading detection model...")
        self.detect_model = YOLO(config.YOLO_DETECT_MODEL)
        
        print("[YOLODetector] Loading pose model...")
        self.pose_model = YOLO(config.YOLO_POSE_MODEL)
        
        # Use GPU if available (CUDA for NVIDIA GPUs)
        self.device = "cuda" if self._check_cuda() else "cpu"
        print(f"[YOLODetector] Using device: {self.device}")
        
        # Use half precision on GPU for faster inference
        self.half = config.USE_HALF_PRECISION and self.device == "cuda"
    
    def _check_cuda(self):
        """Check if NVIDIA GPU (CUDA) is available."""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[YOLODetector] GPU found: {gpu_name}")
            return available
        except ImportError:
            return False
    
    def detect(self, frame, confidence=None, classes=None):
        """
        Detect all objects in a frame.
        
        Args:
            frame:      numpy array (BGR image from OpenCV)
            confidence: minimum confidence threshold (default: from config)
            classes:    list of COCO class IDs to detect (default: all)
        
        Returns:
            list of Detection objects
        """
        if frame is None:
            return []
        
        conf = confidence or config.PERSON_CONFIDENCE
        
        # Run YOLOv8 detection
        results = self.detect_model(
            frame,
            conf=conf,
            iou=config.NMS_IOU_THRESHOLD,
            device=self.device,
            half=self.half,
            classes=classes,
            verbose=False  # Don't print to console every frame
        )
        
        # Convert YOLO results to our Detection objects
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()      # [x1, y1, x2, y2]
                conf_score = float(box.conf[0].cpu().numpy())  # confidence
                cls_id = int(box.cls[0].cpu().numpy())          # class ID
                cls_name = result.names[cls_id]                 # class name
                
                detection = Detection(bbox, conf_score, cls_id, cls_name)
                detections.append(detection)
        
        return detections
    
    def detect_people(self, frame, confidence=None):
        """
        Detect only people in a frame.
        
        Args:
            frame:      numpy array (BGR image)
            confidence: minimum confidence (default: from config)
        
        Returns:
            list of Detection objects (only people)
        """
        return self.detect(
            frame,
            confidence=confidence or config.PERSON_CONFIDENCE,
            classes=[config.PERSON_CLASS_ID]
        )
    
    def detect_weapons(self, frame, confidence=None):
        """
        Detect weapons (knives, scissors) in a frame.
        
        Args:
            frame:      numpy array (BGR image)
            confidence: minimum confidence (default: from config)
        
        Returns:
            list of Detection objects (only weapons)
        """
        return self.detect(
            frame,
            confidence=confidence or config.WEAPON_CONFIDENCE,
            classes=config.WEAPON_CLASS_IDS
        )
    
    def detect_all_relevant(self, frame):
        """
        Detect people AND weapons in one pass (more efficient).
        
        Returns:
            tuple: (people_detections, weapon_detections)
        """
        # Detect both people and weapons in one inference
        all_classes = [config.PERSON_CLASS_ID] + config.WEAPON_CLASS_IDS
        all_detections = self.detect(
            frame,
            confidence=min(config.PERSON_CONFIDENCE, config.WEAPON_CONFIDENCE),
            classes=all_classes
        )
        
        # Split into categories
        people = [d for d in all_detections if d.class_id == config.PERSON_CLASS_ID]
        weapons = [d for d in all_detections if d.class_id in config.WEAPON_CLASS_IDS]
        
        return people, weapons
    
    def detect_poses(self, frame, confidence=None):
        """
        Detect people with pose estimation (skeleton keypoints).
        Used for fight detection and action recognition.
        
        Args:
            frame:      numpy array (BGR image)
            confidence: minimum confidence (default: from config)
        
        Returns:
            list of PoseDetection objects
        """
        if frame is None:
            return []
        
        conf = confidence or config.POSE_CONFIDENCE
        
        # Run YOLOv8-pose
        results = self.pose_model(
            frame,
            conf=conf,
            iou=config.NMS_IOU_THRESHOLD,
            device=self.device,
            half=self.half,
            verbose=False
        )
        
        pose_detections = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            
            boxes = result.boxes
            keypoints = result.keypoints
            
            for i in range(len(boxes)):
                bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
                conf_score = float(boxes[i].conf[0].cpu().numpy())
                kpts = keypoints[i].data[0].cpu().numpy()  # shape: (17, 3)
                
                pose = PoseDetection(bbox, conf_score, kpts)
                pose_detections.append(pose)
        
        return pose_detections
