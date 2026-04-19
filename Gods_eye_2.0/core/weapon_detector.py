# core/weapon_detector.py - Weapon Detection (Knife only for now)

import cv2
import numpy as np
import time
from config import WEAPON_CLASSES, WEAPON_CONFIDENCE


class WeaponDetector:
    """
    Detects weapons from YOLO detections.
    Currently: Knife (COCO class 43) only.
    Gun detection requires custom model (future enhancement).
    Context: higher threat if weapon is near a person.
    """

    def __init__(self, cooldown=5.0):
        self.confidence = WEAPON_CONFIDENCE
        self.cooldown = cooldown
        self.last_alert = {}
        self.active_weapons = []

    def detect(self, boxes, classes, confidences, person_positions=None):
        """Process YOLO results and identify weapons."""
        self.active_weapons = []
        new_alerts = []
        current_time = time.time()

        if person_positions is None:
            person_positions = []

        for box, cls, conf in zip(boxes, classes, confidences):
            cls = int(cls)

            if cls not in WEAPON_CLASSES:
                continue
            if conf < self.confidence:
                continue

            weapon_info = WEAPON_CLASSES[cls]
            box_int = [int(b) for b in box]
            cx = (box_int[0] + box_int[2]) // 2
            cy = (box_int[1] + box_int[3]) // 2

            # Check proximity to persons
            near_person = False
            min_distance = float('inf')
            for px, py in person_positions:
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                min_distance = min(min_distance, dist)
                if dist < 150:
                    near_person = True

            # Threat calculation
            base_threat = 0.7  # Knife is always high
            threat = base_threat + (0.2 if near_person else 0) + (conf - 0.3) * 0.3
            threat = min(1.0, max(0.0, threat))

            weapon_data = {
                'type': weapon_info['name'],
                'danger': weapon_info['danger'],
                'color': weapon_info['color'],
                'box': box_int,
                'center': (cx, cy),
                'confidence': float(conf),
                'near_person': near_person,
                'threat': round(threat, 2),
                'distance_to_person': min_distance if min_distance != float('inf') else None,
            }
            self.active_weapons.append(weapon_data)

            # Cooldown check for alerts
            weapon_key = f"{weapon_info['name']}_{cx // 100}_{cy // 100}"
            if current_time - self.last_alert.get(weapon_key, 0) > self.cooldown:
                if threat > 0.3:
                    self.last_alert[weapon_key] = current_time
                    new_alerts.append(weapon_data)

        return {
            'weapons': self.active_weapons,
            'alerts': new_alerts,
            'count': len(self.active_weapons),
            'high_danger': sum(1 for w in self.active_weapons if w['danger'] == 'HIGH'),
        }

    def draw(self, frame):
        """Draw weapon boxes on frame"""
        for weapon in self.active_weapons:
            x1, y1, x2, y2 = weapon['box']
            color = weapon['color']
            threat = weapon['threat']
            thickness = 2 + int(threat * 3)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Corner accents for high danger
            if weapon['danger'] == 'HIGH':
                clen = min(15, (x2 - x1) // 3)
                for (px, py), (dx, dy) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))]:
                    cv2.line(frame, (px, py), (px + dx * clen, py), color, thickness + 1)
                    cv2.line(frame, (px, py), (px, py + dy * clen), color, thickness + 1)

            label = f"{weapon['type']} {int(threat * 100)}%"
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if weapon['near_person']:
                cv2.putText(frame, "THREAT!", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_banner(self, frame):
        """Draw alert banner if high-threat weapons detected"""
        if not self.active_weapons:
            return frame

        h, w = frame.shape[:2]
        high_threats = [wp for wp in self.active_weapons if wp['threat'] > 0.5]

        if high_threats:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            names = ", ".join(set(wp['type'] for wp in high_threats))
            cv2.putText(frame, f"WEAPON ALERT: {names}", (w // 2 - 120, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame