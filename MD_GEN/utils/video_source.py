"""
CrowdShield - Video Source Handler
===================================
Handles video input from three sources:
  1. System webcam (built-in or USB camera)
  2. IP Camera (phone camera via apps like "IP Webcam" or "DroidCam")
  3. Video files (MP4, AVI, etc.)

Also provides frame preprocessing (resize, flip) and FPS calculation.
"""

import cv2
import time
import threading
import numpy as np
from config import MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT


class VideoSource:
    """
    Manages a single video source (webcam, IP camera, or video file).
    
    Usage:
        source = VideoSource()
        source.open(0)                          # Open webcam
        source.open("http://192.168.1.5:8080/video")  # Open IP camera
        source.open("path/to/video.mp4")        # Open video file
        
        frame = source.read()                    # Get one frame
        source.release()                         # Close when done
    """
    
    def __init__(self):
        self.cap = None                # OpenCV VideoCapture object
        self.source = None             # Current source (int or string)
        self.is_open = False           # Whether the source is currently open
        self.frame_count = 0           # Total frames read
        self.fps = 0.0                 # Current FPS (calculated)
        self._fps_timer = time.time()  # Timer for FPS calculation
        self._fps_frame_count = 0      # Frame counter for FPS calculation
        self.source_width = 0          # Original source width
        self.source_height = 0         # Original source height
        self.source_fps = 0            # Source's native FPS
    
    def open(self, source):
        """
        Open a video source.
        
        Args:
            source: Can be:
                - int (0, 1, 2...) for webcam index
                - str URL for IP camera (e.g., "http://192.168.1.5:8080/video")
                - str file path for video file (e.g., "video.mp4")
        
        Returns:
            True if opened successfully, False otherwise
        """
        # Close any existing source first
        self.release()
        
        self.source = source
        
        # Try to open the video source
        # OpenCV handles all three types with the same VideoCapture class!
        try:
            if isinstance(source, int):
                # Webcam - use DirectShow on Windows for better compatibility
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                print(f"[VideoSource] ERROR: Could not open source: {source}")
                return False
            
            # Read source properties
            self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            self.is_open = True
            self.frame_count = 0
            self._fps_timer = time.time()
            self._fps_frame_count = 0
            
            print(f"[VideoSource] Opened: {source}")
            print(f"[VideoSource] Resolution: {self.source_width}x{self.source_height}")
            print(f"[VideoSource] FPS: {self.source_fps}")
            
            return True
            
        except Exception as e:
            print(f"[VideoSource] ERROR opening source: {e}")
            return False
    
    def read(self):
        """
        Read one frame from the video source.
        
        Returns:
            numpy array (BGR image) if successful, None if failed
        """
        if not self.is_open or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            # For video files, we can loop back to the beginning
            if isinstance(self.source, str) and not self.source.startswith("http"):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                return None
        
        # Resize if frame is too large (saves processing time)
        frame = self._resize_frame(frame)
        
        # Update FPS calculation
        self.frame_count += 1
        self._update_fps()
        
        return frame
    
    def _resize_frame(self, frame):
        """
        Resize frame if it exceeds maximum dimensions.
        Maintains aspect ratio.
        """
        h, w = frame.shape[:2]
        
        # Check if resize is needed
        if w <= MAX_FRAME_WIDTH and h <= MAX_FRAME_HEIGHT:
            return frame
        
        # Calculate new size maintaining aspect ratio
        scale = min(MAX_FRAME_WIDTH / w, MAX_FRAME_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _update_fps(self):
        """Calculate current FPS (updated every second)."""
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_timer
        
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_timer = time.time()
    
    def get_total_frames(self):
        """Get total frames in video file (returns 0 for live sources)."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_current_position(self):
        """Get current frame position in video file."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def seek(self, frame_number):
        """Seek to a specific frame in video file."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self):
        """Release the video source and free resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_open = False
        self.frame_count = 0
    
    def is_file(self):
        """Check if the source is a video file (not live)."""
        if isinstance(self.source, str) and not self.source.startswith(("http", "rtsp")):
            return True
        return False
    
    def __del__(self):
        """Clean up when object is destroyed."""
        self.release()


class ThreadedVideoSource(VideoSource):
    """
    A threaded version of VideoSource for better performance.
    Reads frames in a background thread so the main thread isn't blocked.
    
    Usage:
        source = ThreadedVideoSource()
        source.open(0)
        source.start()          # Start background reading
        
        frame = source.read()    # Always returns the latest frame instantly
        source.stop()            # Stop background reading
        source.release()
    """
    
    def __init__(self):
        super().__init__()
        self._thread = None
        self._running = False
        self._latest_frame = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background frame reading thread."""
        if not self.is_open:
            print("[ThreadedVideoSource] ERROR: Source not open. Call open() first.")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[ThreadedVideoSource] Background reading started.")
    
    def _read_loop(self):
        """Background loop that continuously reads frames."""
        while self._running and self.is_open:
            frame = super().read()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.001)  # Small sleep to avoid busy-waiting on error
    
    def read(self):
        """Get the latest frame (non-blocking)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
    
    def stop(self):
        """Stop the background reading thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[ThreadedVideoSource] Background reading stopped.")
