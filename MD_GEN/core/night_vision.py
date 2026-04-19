"""
CrowdShield - Night Vision Module
====================================
Automatically enhances low-light/dark video frames for better detection.

Uses CLAHE (Contrast Limited Adaptive Histogram Equalization):
  - Divides the image into tiles
  - Equalizes histogram in each tile independently
  - Limits contrast to prevent over-amplification
  - Result: dark areas become visible, bright areas aren't blown out

This is NOT infrared/thermal - it's software-based enhancement.
Works well for dimly lit CCTV footage.
"""

import cv2
import numpy as np
import config


class NightVisionEnhancer:
    """
    Automatically detects and enhances low-light video frames.
    
    Usage:
        enhancer = NightVisionEnhancer()
        
        # Each frame:
        enhanced, was_dark = enhancer.process(frame)
        # If frame was dark, enhanced = brightened version
        # If frame was bright, enhanced = original (no change)
    """
    
    def __init__(self):
        # Create CLAHE object (reusable for performance)
        self.clahe = cv2.createCLAHE(
            clipLimit=config.NIGHT_VISION_CLIP_LIMIT,
            tileGridSize=config.NIGHT_VISION_TILE_SIZE
        )
        
        self.auto_mode = config.NIGHT_VISION_AUTO
        self.force_enabled = False  # Manual override
        self.brightness_threshold = config.NIGHT_VISION_BRIGHTNESS_THRESH
        
        # Stats
        self.current_brightness = 0
        self.is_dark = False
    
    def process(self, frame):
        """
        Process a frame, enhancing if dark.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            (enhanced_frame, is_dark):
                enhanced_frame: The processed frame (enhanced or original)
                is_dark:        Whether the frame was detected as dark
        """
        if frame is None:
            return frame, False
        
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_brightness = np.mean(gray)
        
        # Determine if enhancement is needed
        self.is_dark = self.current_brightness < self.brightness_threshold
        
        should_enhance = (
            self.force_enabled or 
            (self.auto_mode and self.is_dark)
        )
        
        if not should_enhance:
            return frame, self.is_dark
        
        # ===== APPLY ENHANCEMENT =====
        enhanced = self._enhance(frame)
        
        return enhanced, True
    
    def _enhance(self, frame):
        """
        Apply CLAHE enhancement to a dark frame.
        
        Process:
          1. Convert to LAB color space (separates lightness from color)
          2. Apply CLAHE to the L (lightness) channel only
          3. Optionally apply slight gamma correction for extra brightness
          4. Convert back to BGR
        
        Args:
            frame: BGR image
        
        Returns:
            Enhanced BGR image
        """
        # Convert BGR → LAB
        # LAB has 3 channels:
        #   L = lightness (0=black, 255=white) - we enhance this
        #   A = green-red color component
        #   B = blue-yellow color component
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to the lightness channel
        l_enhanced = self.clahe.apply(l_channel)
        
        # If very dark, apply additional gamma correction
        if self.current_brightness < self.brightness_threshold * 0.5:
            l_enhanced = self._gamma_correction(l_enhanced, gamma=0.7)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Optional: slight denoising for very dark frames
        # (dark frames tend to have more noise when brightened)
        if self.current_brightness < self.brightness_threshold * 0.3:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, None, 5, 5, 7, 21
            )
        
        return enhanced
    
    def _gamma_correction(self, image, gamma=0.7):
        """
        Apply gamma correction to brighten an image.
        
        gamma < 1.0 = brighten
        gamma > 1.0 = darken
        gamma = 1.0 = no change
        
        Args:
            image: Grayscale or single-channel image
            gamma: Gamma value
        
        Returns:
            Gamma-corrected image
        """
        # Build a lookup table for fast gamma correction
        # For each input value (0-255), compute the output value
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def set_auto_mode(self, enabled):
        """Enable or disable automatic dark detection."""
        self.auto_mode = enabled
    
    def set_force_enabled(self, enabled):
        """Force night vision on/off regardless of brightness."""
        self.force_enabled = enabled
    
    def set_brightness_threshold(self, threshold):
        """Set the brightness threshold for auto detection (0-255)."""
        self.brightness_threshold = max(0, min(255, threshold))
    
    def get_brightness_info(self):
        """
        Get current brightness information.
        
        Returns:
            dict with brightness info
        """
        return {
            "current_brightness": round(self.current_brightness, 1),
            "threshold": self.brightness_threshold,
            "is_dark": self.is_dark,
            "auto_mode": self.auto_mode,
            "force_enabled": self.force_enabled
        }
