"""
CrowdShield - Alert System & Database
========================================
Manages security alerts and stores them in a SQLite database.

Features:
  - Alert creation with severity levels
  - SQLite database storage for persistence
  - Alert history with timestamps
  - Sound notifications (Windows beep)
  - Snapshot saving (frame capture at alert time)
  - Alert statistics and summaries
"""

import sqlite3
import time
import datetime
import os
import threading
import cv2
import numpy as np
from collections import deque
import config


class Alert:
    """
    Represents a single security alert.
    
    Attributes:
        alert_type:  str - "stampede", "weapon", "fight", "lost_child", "crowd_surge"
        severity:    str - "CRITICAL", "HIGH", "MEDIUM", "LOW"
        message:     str - Human-readable description
        timestamp:   float - Unix timestamp when alert was created
        frame_path:  str - Path to saved screenshot (if any)
        data:        dict - Additional alert data
    """
    def __init__(self, alert_type, severity, message, data=None, frame=None):
        self.alert_type = alert_type
        self.severity = severity or config.ALERT_SEVERITY.get(alert_type, "MEDIUM")
        self.message = message
        self.timestamp = time.time()
        self.datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data = data or {}
        self.frame_path = None
        self.id = None  # Set by database
        
        # Save frame snapshot if provided
        if frame is not None:
            self._save_snapshot(frame)
    
    def _save_snapshot(self, frame):
        """Save a frame snapshot to disk."""
        try:
            filename = f"alert_{self.alert_type}_{int(self.timestamp)}.jpg"
            filepath = os.path.join(config.REPORTS_DIR, filename)
            cv2.imwrite(filepath, frame)
            self.frame_path = filepath
        except Exception as e:
            print(f"[Alert] Failed to save snapshot: {e}")
    
    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "frame_path": self.frame_path,
            "data": str(self.data)
        }


class AlertDatabase:
    """
    SQLite database for storing and querying alerts.
    
    Usage:
        db = AlertDatabase()
        db.save_alert(alert)
        
        recent = db.get_recent_alerts(limit=10)
        stats = db.get_statistics()
    """
    
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        self._init_database()
    
    def _init_database(self):
        """Create the alerts table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                timestamp REAL,
                datetime_str TEXT,
                frame_path TEXT,
                data TEXT
            )
        """)
        
        # Also create a session log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                source_info TEXT,
                total_alerts INTEGER DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                peak_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert):
        """Save an alert to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (alert_type, severity, message, timestamp, datetime_str, frame_path, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.timestamp,
                alert.datetime_str,
                alert.frame_path,
                str(alert.data)
            ))
            
            alert.id = cursor.lastrowid
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[AlertDatabase] Error saving alert: {e}")
    
    def get_recent_alerts(self, limit=50):
        """Get the most recent alerts."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            print(f"[AlertDatabase] Error reading alerts: {e}")
            return []
    
    def get_alerts_by_type(self, alert_type, limit=50):
        """Get alerts of a specific type."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE alert_type = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (alert_type, limit))
            
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            return []
    
    def get_statistics(self):
        """Get alert statistics summary."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total alerts
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total = cursor.fetchone()[0]
            
            # Alerts by type
            cursor.execute("""
                SELECT alert_type, COUNT(*) as count 
                FROM alerts 
                GROUP BY alert_type
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Alerts by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count 
                FROM alerts 
                GROUP BY severity
            """)
            by_severity = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Recent 24 hours
            day_ago = time.time() - 86400
            cursor.execute("""
                SELECT COUNT(*) FROM alerts 
                WHERE timestamp > ?
            """, (day_ago,))
            last_24h = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total": total,
                "by_type": by_type,
                "by_severity": by_severity,
                "last_24h": last_24h
            }
        except Exception as e:
            return {"total": 0, "by_type": {}, "by_severity": {}, "last_24h": 0}
    
    def clear_all(self):
        """Clear all alerts from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[AlertDatabase] Error clearing: {e}")


class AlertManager:
    """
    Main alert manager that coordinates detection alerts.
    
    Usage:
        manager = AlertManager()
        
        # When something is detected:
        manager.trigger_alert("stampede", "CRITICAL", "Stampede detected!", frame=current_frame)
        
        # Get recent alerts for UI:
        alerts = manager.get_recent(10)
    """
    
    def __init__(self):
        self.database = AlertDatabase()
        
        # In-memory alert buffer (for quick UI access)
        self.recent_alerts = deque(maxlen=config.ALERT_MAX_LOG)
        
        # Cooldown timers per alert type
        self.cooldowns = {}
        
        # Sound notification
        self.sound_enabled = config.ALERT_SOUND_ENABLED
    
    def trigger_alert(self, alert_type, severity=None, message="", data=None, frame=None):
        """
        Create and store a new alert.
        
        Args:
            alert_type: "stampede", "weapon", "fight", "lost_child", "crowd_surge", "crowd_high"
            severity:   "CRITICAL", "HIGH", "MEDIUM", "LOW" (auto from config if None)
            message:    Human-readable alert message
            data:       Additional data dict
            frame:      Video frame to save as snapshot
        
        Returns:
            Alert object, or None if in cooldown
        """
        # Check cooldown
        if not self._check_cooldown(alert_type):
            return None
        
        # Default severity from config
        if severity is None:
            severity = config.ALERT_SEVERITY.get(alert_type, "MEDIUM")
        
        # Create alert
        alert = Alert(alert_type, severity, message, data, frame)
        
        # Save to database
        self.database.save_alert(alert)
        
        # Add to in-memory buffer
        self.recent_alerts.appendleft(alert.to_dict())
        
        # Play sound notification
        if self.sound_enabled and severity in ("CRITICAL", "HIGH"):
            self._play_alert_sound(severity)
        
        print(f"[ALERT] [{severity}] {alert_type}: {message}")
        
        return alert
    
    def _check_cooldown(self, alert_type):
        """Check if enough time has passed since the last alert of this type."""
        current_time = time.time()
        cooldown_duration = config.ALERT_COOLDOWN_DEFAULT
        
        # Custom cooldowns for specific types
        type_cooldowns = {
            "stampede": config.STAMPEDE_COOLDOWN_SEC,
            "fight": config.FIGHT_COOLDOWN_SEC,
            "lost_child": config.CHILD_COOLDOWN_SEC,
        }
        cooldown_duration = type_cooldowns.get(alert_type, config.ALERT_COOLDOWN_DEFAULT)
        
        last_time = self.cooldowns.get(alert_type, 0)
        
        if current_time - last_time < cooldown_duration:
            return False
        
        self.cooldowns[alert_type] = current_time
        return True
    
    def _play_alert_sound(self, severity):
        """Play an alert sound (Windows only, non-blocking)."""
        try:
            import winsound
            
            def _beep():
                if severity == "CRITICAL":
                    # Three quick high beeps for critical
                    for _ in range(3):
                        winsound.Beep(2000, 200)
                        time.sleep(0.1)
                else:
                    # One medium beep for high  
                    winsound.Beep(1500, 300)
            
            # Run in thread so it doesn't block video
            threading.Thread(target=_beep, daemon=True).start()
        except Exception:
            pass  # Silently ignore if winsound not available
    
    def get_recent(self, limit=20):
        """Get recent alerts from memory (fast)."""
        return list(self.recent_alerts)[:limit]
    
    def get_all_from_db(self, limit=100):
        """Get alerts from database (complete history)."""
        return self.database.get_recent_alerts(limit)
    
    def get_statistics(self):
        """Get alert statistics."""
        return self.database.get_statistics()
    
    def clear_all(self):
        """Clear all alerts."""
        self.recent_alerts.clear()
        self.cooldowns.clear()
        self.database.clear_all()
