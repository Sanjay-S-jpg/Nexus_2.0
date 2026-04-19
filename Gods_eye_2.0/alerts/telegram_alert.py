# alerts/telegram_alert.py - Telegram Notifications

import requests
import cv2
import numpy as np
import time
import os
from threading import Thread

class TelegramAlert:
    """Send alerts via Telegram bot"""
    
    def __init__(self, bot_token, chat_id, cooldown=30):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self.last_alert = {}
        self.enabled = bool(bot_token and chat_id and bot_token != "YOUR_BOT_TOKEN")
        
        if self.enabled:
            print("[Telegram] Alerts enabled")
        else:
            print("[Telegram] Alerts disabled - update config.py with your bot token")
    
    def _send_async(self, message, image=None):
        """Send message in background thread"""
        Thread(target=self._send, args=(message, image), daemon=True).start()
    
    def _send(self, message, image=None):
        """Send message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            if image is not None:
                # Send photo with caption
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                
                # Encode image
                _, buffer = cv2.imencode('.jpg', image)
                
                files = {'photo': ('alert.jpg', buffer.tobytes(), 'image/jpeg')}
                data = {'chat_id': self.chat_id, 'caption': message}
                
                response = requests.post(url, files=files, data=data, timeout=10)
            else:
                # Send text only
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                data = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
                
                response = requests.post(url, data=data, timeout=10)
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"[Telegram] Error: {e}")
            return False
    
    def stampede_alert(self, risk_score, alert_level, frame=None):
        """Send stampede alert"""
        key = f"stampede_{alert_level}"
        
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return False
        
        self.last_alert[key] = time.time()
        
        level_names = ['SAFE', 'CAUTION', 'WARNING', 'CRITICAL']
        emoji = ['✅', '⚠️', '🔶', '🚨'][alert_level]
        
        message = f"""
{emoji} <b>STAMPEDE ALERT</b> {emoji}

<b>Risk Level:</b> {level_names[alert_level]}
<b>Risk Score:</b> {risk_score}%
<b>Time:</b> {time.strftime('%H:%M:%S')}

Immediate attention required!
"""
        
        self._send_async(message, frame)
        return True
    
    def weapon_alert(self, weapon_type, threat_level, frame=None):
        """Send weapon detection alert"""
        key = f"weapon_{weapon_type}"
        
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return False
        
        self.last_alert[key] = time.time()
        
        message = f"""
🔪 <b>WEAPON DETECTED</b> 🔪

<b>Type:</b> {weapon_type}
<b>Threat Level:</b> {int(threat_level * 100)}%
<b>Time:</b> {time.strftime('%H:%M:%S')}

Security response required!
"""
        
        self._send_async(message, frame)
        return True
    
    def crowd_alert(self, count, threshold, frame=None):
        """Send crowd density alert"""
        key = "crowd_density"
        
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return False
        
        self.last_alert[key] = time.time()
        
        message = f"""
👥 <b>CROWD DENSITY ALERT</b> 👥

<b>Current Count:</b> {count} people
<b>Threshold:</b> {threshold} people
<b>Time:</b> {time.strftime('%H:%M:%S')}

Monitor situation closely!
"""
        
        self._send_async(message, frame)
        return True
    
    def test(self):
        """Send test message"""
        message = "✅ <b>CrowdIntel Test</b>\n\nTelegram alerts are working!"
        return self._send(message)