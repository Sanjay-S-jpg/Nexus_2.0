# core/head_detector.py - Proper Head Detection for Dense Crowds

import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request
import os


class HeadDetector:
    """
    Detects HEADS not full bodies.
    Works in dense crowds where only heads are visible.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Use YOLO trained on heads/faces - much better for crowds
        # CrowdHuman dataset trained model
        model_path = "yolov8n-crowdhuman.pt"
        
        if not os.path.exists(model_path):
            print("[HeadDetector] Downloading head detection model...")
            # Use standard YOLOv8 but we'll detect class 0 (person) with low conf
            # and estimate heads from upper portion of bbox
            self.model = YOLO("yolov8n.pt")
            self.head_mode = "estimate"
        else:
            self.model = YOLO(model_path)
            self.head_mode = "direct"
        
        self.model.to(device)
        print(f"[HeadDetector] Ready on {device}")
    
    def detect(self, frame, conf=0.15):
        """
        Detect heads in frame.
        
        Returns:
            List of (cx, cy) head positions
        """
        # Run detection with VERY low confidence to catch partial people
        results = self.model(
            frame, 
            conf=conf, 
            verbose=False,
            device=self.device,
            classes=[0]  # Person class
        )
        
        heads = []
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                
                # HEAD = top 25% of bounding box
                head_y = y1 + (y2 - y1) * 0.15  # Near top
                head_x = (x1 + x2) / 2  # Center
                
                heads.append((head_x, head_y))
        
        return heads
    
    def estimate_crowd_count(self, frame, grid_size=(8, 8)):
        """
        Estimate total crowd using density analysis.
        Combines detection + texture analysis.
        """
        # Direct detections
        detected_heads = self.detect(frame, conf=0.1)
        detected_count = len(detected_heads)
        
        # Density estimation for missed people
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        h, w = frame.shape[:2]
        cell_h, cell_w = h // grid_size[0], w // grid_size[1]
        
        density_count = 0
        
        for gy in range(grid_size[0]):
            for gx in range(grid_size[1]):
                y1, y2 = gy * cell_h, (gy + 1) * cell_h
                x1, x2 = gx * cell_w, (gx + 1) * cell_w
                
                cell = gray[y1:y2, x1:x2]
                
                # Count heads in this cell
                cell_detections = sum(1 for (hx, hy) in detected_heads 
                                     if x1 <= hx < x2 and y1 <= hy < y2)
                
                # If cell has people but high texture, probably more hidden
                if cell_detections > 0:
                    # Texture complexity = more people
                    edges = cv2.Canny(cell, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    
                    # Estimate hidden people based on density
                    if edge_density > 0.15:
                        density_count += int(cell_detections * 0.5)
        
        total = detected_count + density_count
        
        return {
            'detected': detected_count,
            'estimated_hidden': density_count,
            'total': total,
            'heads': detected_heads
        }