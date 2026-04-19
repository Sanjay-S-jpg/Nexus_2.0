# alerts/desktop_alert.py - Desktop Notifications

import time
from threading import Thread

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("[Desktop] plyer not installed - desktop notifications disabled")

class DesktopAlert:
    """Desktop notification system"""
    
    def __init__(self, cooldown=30):
        self.cooldown = cooldown
        self.last_alert = {}
        self.enabled = PLYER_AVAILABLE
    
    def _notify_async(self, title, message):
        """Send notification in background"""
        Thread(target=self._notify, args=(title, message), daemon=True).start()
    
    def _notify(self, title, message):
        """Send desktop notification"""
        if not self.enabled:
            return False
        
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="CrowdIntel",
                timeout=5
            )
            return True
        except Exception as e:
            print(f"[Desktop] Error: {e}")
            return False
    
    def stampede_alert(self, risk_score, alert_level):
        """Send stampede alert"""
        key = f"stampede_{alert_level}"
        
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return False
        
        self.last_alert[key] = time.time()
        
        levels = ['SAFE', 'CAUTION', 'WARNING', 'CRITICAL']
        
        self._notify_async(
            f"🚨 STAMPEDE {levels[alert_level]}",
            f"Risk Score: {risk_score}%\nImmediate attention required!"
        )
        return True
    
    def weapon_alert(self, weapon_type, threat_level):
        """Send weapon alert"""
        key = f"weapon_{weapon_type}"
        
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return False
        
        self.last_alert[key] = time.time()
        
        self._notify_async(
            f"🔪 WEAPON DETECTED: {weapon_type}",
            f"Threat Level: {int(threat_level * 100)}%\nSecurity response required!"
        )
        return True
    
    def crowd_alert(self, count):
        """Send crowd density alert"""
        key = "crowd"
        
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return False
        
        self.last_alert[key] = time.time()
        
        self._notify_async(
            "👥 HIGH CROWD DENSITY",
            f"Current count: {count} people\nMonitor situation!"
        )
        return True