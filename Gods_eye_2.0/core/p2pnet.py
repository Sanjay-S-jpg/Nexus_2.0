# core/p2pnet.py - P2PNet Crowd Counting Model
#
# P2PNet: Point-to-Point Network for Crowd Counting
# Paper: https://arxiv.org/abs/2107.12746
# 
# This is a simplified implementation optimized for real-time use

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from collections import deque
import os

class VGGBackbone(nn.Module):
    """VGG16 backbone for feature extraction"""
    
    def __init__(self):
        super().__init__()
        
        # VGG16 layers (simplified for speed)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.features(x)


class P2PNet(nn.Module):
    """
    P2PNet: Point-to-Point Network
    Predicts a point for each person in the crowd
    """
    
    def __init__(self, num_classes=1, hidden_dim=256, num_queries=500):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Backbone
        self.backbone = VGGBackbone()
        
        # Reduce channels
        self.reduce = nn.Sequential(
            nn.Conv2d(512, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Point prediction head
        self.point_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),  # Confidence map
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Reduce channels
        features = self.reduce(features)
        
        # Predict confidence map
        conf_map = self.point_head(features)
        
        return conf_map


class P2PCrowdCounter:
    """
    P2PNet-based crowd counter for dense crowds.
    
    Features:
    - Point-based detection (each person = one point)
    - Works with dense crowds (100+ people)
    - GPU accelerated
    - Real-time capable
    """
    
    def __init__(self, device='cuda', conf_threshold=0.3):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        
        print(f"[P2PNet] Initializing on {self.device}")
        
        # Initialize model
        self.model = P2PNet().to(self.device)
        self.model.eval()
        
        # Try to load pretrained weights
        self._load_weights()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Smoothing
        self.count_history = deque(maxlen=10)
        self.last_points = []
        
        print("[P2PNet] Ready!")
    
    def _load_weights(self):
        """Load pretrained weights if available"""
        weights_path = "models/p2pnet.pth"
        vgg_path = "models/vgg16_bn.pth"
        
        try:
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("[P2PNet] Loaded pretrained weights")
            elif os.path.exists(vgg_path):
                # Load VGG backbone weights
                vgg_state = torch.load(vgg_path, map_location=self.device)
                
                # Map VGG weights to our backbone
                backbone_state = {}
                for k, v in vgg_state.items():
                    if k.startswith('features.'):
                        backbone_state[k] = v
                
                # Partial load
                model_dict = self.model.backbone.state_dict()
                pretrained_dict = {k: v for k, v in backbone_state.items() 
                                   if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                self.model.backbone.load_state_dict(model_dict, strict=False)
                print(f"[P2PNet] Loaded VGG backbone ({len(pretrained_dict)} layers)")
            else:
                print("[P2PNet] No pretrained weights found, using random init")
                print("[P2PNet] Run download_models.py to get better accuracy")
        except Exception as e:
            print(f"[P2PNet] Warning: Could not load weights: {e}")
    
    def _extract_points(self, conf_map, original_size):
        """Extract person locations from confidence map"""
        h_orig, w_orig = original_size
        
        # Resize confidence map to original size
        conf_map = conf_map.squeeze().cpu().numpy()
        conf_map = cv2.resize(conf_map, (w_orig, h_orig))
        
        # Find local maxima (non-maximum suppression)
        kernel = np.ones((5, 5), np.float32)
        dilated = cv2.dilate(conf_map, kernel)
        local_max = (conf_map == dilated) & (conf_map > self.conf_threshold)
        
        # Get point coordinates
        points = np.where(local_max)
        points = list(zip(points[1], points[0]))  # (x, y) format
        
        return points, conf_map
    
    @torch.no_grad()
    def count(self, frame):
        """
        Count people in frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            dict with count and point locations
        """
        h, w = frame.shape[:2]
        
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))  # Fixed input size for speed
        
        # To tensor
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        conf_map = self.model(img_tensor)
        
        # Extract points
        points, density = self._extract_points(conf_map, (h, w))
        
        # Count
        raw_count = len(points)
        
        # Smooth
        self.count_history.append(raw_count)
        smooth_count = int(np.mean(self.count_history))
        
        self.last_points = points
        
        return {
            'count': smooth_count,
            'raw_count': raw_count,
            'points': points,
            'density_map': density,
            'method': 'P2PNet'
        }
    
    def draw(self, frame, result):
        """Draw detection points on frame"""
        for (x, y) in result.get('points', []):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)
        
        return frame
    
    def get_density_overlay(self, frame, result, alpha=0.5):
        """Get density map overlay"""
        density = result.get('density_map')
        if density is None:
            return frame
        
        # Normalize and colorize
        density_norm = (density * 255).astype(np.uint8)
        density_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        
        # Resize to match frame
        if density_color.shape[:2] != frame.shape[:2]:
            density_color = cv2.resize(density_color, (frame.shape[1], frame.shape[0]))
        
        # Blend
        return cv2.addWeighted(frame, 1 - alpha, density_color, alpha, 0)