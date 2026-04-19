# core/person_tracker.py - Person Tracking with Re-ID

import cv2
import numpy as np
import time

from config import REID_CONFIG


class TrackedPerson:
    """Represents a tracked individual"""

    def __init__(self, track_id, position, box):
        self.id = track_id
        self.positions = [position]
        self.boxes = [box]
        self.velocities = []
        self.last_seen = time.time()
        self.color = self._generate_color()
        self.status = "ACTIVE"
        self.total_distance = 0
        self.appearance_hist = None
        self.dress_color_name = "Unknown"

    def _generate_color(self):
        np.random.seed(self.id * 42)
        return tuple(int(x) for x in np.random.randint(100, 255, 3))

    def update(self, position, box):
        if self.positions:
            last_pos = self.positions[-1]
            vx = position[0] - last_pos[0]
            vy = position[1] - last_pos[1]
            self.velocities.append((vx, vy))
            if len(self.velocities) > 30:
                self.velocities.pop(0)
            self.total_distance += np.sqrt(vx**2 + vy**2)

        self.positions.append(position)
        self.boxes.append(box)
        self.last_seen = time.time()
        self.status = "ACTIVE"

        if len(self.positions) > 100:
            self.positions.pop(0)
        if len(self.boxes) > 100:
            self.boxes.pop(0)

    @property
    def speed(self):
        if len(self.velocities) < 2:
            return 0
        recent = self.velocities[-5:]
        return np.mean([np.sqrt(vx**2 + vy**2) for vx, vy in recent])

    @property
    def direction(self):
        if len(self.velocities) < 2:
            return 0
        recent = self.velocities[-5:]
        avg_vx = np.mean([v[0] for v in recent])
        avg_vy = np.mean([v[1] for v in recent])
        return np.degrees(np.arctan2(avg_vy, avg_vx))

    @property
    def current_position(self):
        return self.positions[-1] if self.positions else (0, 0)

    @property
    def current_box(self):
        return self.boxes[-1] if self.boxes else (0, 0, 0, 0)


class PersonTracker:
    """
    Person tracker using YOLO BoT-SORT IDs.
    Supports target locking + basic Re-ID (expanded in Step 3).
    """

    def __init__(self):
        self.persons = {}  # track_id -> TrackedPerson
        self.locked_target = None  # current live track id of target
        self.target_memory = {}
        self.locked_profile = None
        self.locked_signature_id = None  # stable user-facing target id
        self.locked_since = None
        self.lost_since = None

        self.hist_bins = REID_CONFIG.get("histogram_bins", 64)
        self.reid_threshold = REID_CONFIG.get("match_threshold", 0.6)
        self.search_radius = REID_CONFIG.get("search_radius", 200)
        self.max_lost_time = REID_CONFIG.get("max_lost_time", 10.0)
        self.reid_grace_seconds = REID_CONFIG.get("grace_seconds", 1.2)
        self.reid_min_similarity = REID_CONFIG.get("min_similarity", 0.80)
        self.reid_min_margin = REID_CONFIG.get("min_margin", 0.08)

    def _safe_box(self, box, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _extract_torso_hist(self, frame, box):
        if frame is None:
            return None
        safe = self._safe_box(box, frame)
        if safe is None:
            return None

        x1, y1, x2, y2 = safe
        h = y2 - y1
        w = x2 - x1
        if h < 20 or w < 10:
            return None

        # Torso crop is more stable than full body (less leg/background variation)
        ty1 = y1 + int(0.20 * h)
        ty2 = y1 + int(0.70 * h)
        tx1 = x1 + int(0.15 * w)
        tx2 = x2 - int(0.15 * w)

        ty1 = max(y1, min(ty1, y2 - 1))
        ty2 = max(ty1 + 1, min(ty2, y2))
        tx1 = max(x1, min(tx1, x2 - 1))
        tx2 = max(tx1 + 1, min(tx2, x2))

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.hist_bins, self.hist_bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _extract_torso_crop(self, frame, box):
        if frame is None:
            return None
        safe = self._safe_box(box, frame)
        if safe is None:
            return None

        x1, y1, x2, y2 = safe
        h = y2 - y1
        w = x2 - x1
        if h < 20 or w < 10:
            return None

        ty1 = y1 + int(0.20 * h)
        ty2 = y1 + int(0.70 * h)
        tx1 = x1 + int(0.15 * w)
        tx2 = x2 - int(0.15 * w)

        ty1 = max(y1, min(ty1, y2 - 1))
        ty2 = max(ty1 + 1, min(ty2, y2))
        tx1 = max(x1, min(tx1, x2 - 1))
        tx2 = max(tx1 + 1, min(tx2, x2))

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None
        return crop

    def _dress_color_name(self, frame, box):
        crop = self._extract_torso_crop(frame, box)
        if crop is None:
            return "Unknown"

        # Robust color estimation:
        # 1) check dark/light dominance first
        # 2) then classify dominant hue only on valid colorful pixels
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].reshape(-1)
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)

        dark_ratio = float(np.mean(v < 60))
        white_ratio = float(np.mean((v > 190) & (s < 35)))
        gray_ratio = float(np.mean((s < 30) & (v >= 60) & (v <= 190)))

        if dark_ratio >= 0.45:
            return "Black"
        if white_ratio >= 0.40:
            return "White"
        if gray_ratio >= 0.40:
            return "Gray"

        # Keep only confidently colored pixels
        color_mask = (s > 45) & (v > 55)
        if np.sum(color_mask) < 20:
            # low-color region: fallback by brightness
            v_mean = float(np.mean(v))
            if v_mean < 85:
                return "Black"
            if v_mean > 180:
                return "White"
            return "Gray"

        h_sel = h[color_mask]
        s_sel = s[color_mask].astype(np.float32)

        # Weighted dominant hue bin (8 bins across [0, 180))
        bins = np.array([0, 8, 18, 33, 78, 96, 132, 155, 180], dtype=np.int32)
        idx = np.digitize(h_sel, bins) - 1
        idx = np.clip(idx, 0, len(bins) - 2)

        weights = np.zeros(len(bins) - 1, dtype=np.float32)
        for i, sat in zip(idx, s_sel):
            weights[i] += sat

        dominant = int(np.argmax(weights))
        h_dom = float((bins[dominant] + bins[dominant + 1]) / 2)

        # Brown is usually low-medium saturation orange region
        s_mean = float(np.mean(s_sel))
        if 10 <= h_dom <= 22 and s_mean < 90:
            return "Brown"

        # OpenCV hue is [0, 179]
        if h_dom < 8 or h_dom >= 170:
            return "Red"
        if h_dom < 18:
            return "Orange"
        if h_dom < 33:
            return "Yellow"
        if h_dom < 78:
            return "Green"
        if h_dom < 96:
            return "Cyan"
        if h_dom < 132:
            return "Blue"
        if h_dom < 155:
            return "Purple"
        return "Pink"

    def _hist_similarity(self, h1, h2):
        if h1 is None or h2 is None:
            return 0.0
        # Correlation in [-1, 1], remap to [0, 1]
        score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        return float((score + 1.0) / 2.0)

    def _predict_target_position(self):
        if not self.target_memory:
            return None

        last_pos = self.target_memory.get("last_position")
        velocity = self.target_memory.get("last_velocity", (0.0, 0.0))
        if last_pos is None:
            return None

        # One-step linear prediction in frame-space
        return (float(last_pos[0] + velocity[0]), float(last_pos[1] + velocity[1]))

    def _distance_score(self, p1, p2, radius):
        if p1 is None or p2 is None:
            return 0.0
        d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        if d >= radius:
            return 0.0
        return 1.0 - (d / max(radius, 1.0))

    def _try_reidentify_locked_target(self, frame):
        if self.locked_target is None:
            return

        # If current locked track is ACTIVE, no need to re-identify.
        # If it exists but is SEARCHING/LOST, still attempt Re-ID to a new ACTIVE track.
        if self.locked_target in self.persons:
            current = self.persons[self.locked_target]
            if current.status == "ACTIVE":
                return

            # Grace period: allow original tracker ID to recover before switching.
            missing_time = time.time() - current.last_seen
            if missing_time < self.reid_grace_seconds:
                return
        if self.locked_profile is None:
            return

        predicted = self._predict_target_position()
        best_track = None
        best_score = 0.0
        second_best = 0.0

        last_box = self.target_memory.get("last_box")
        last_h = None
        if last_box is not None:
            last_h = max(1.0, float(last_box[3] - last_box[1]))

        for track_id, person in self.persons.items():
            if person.status != "ACTIVE":
                continue

            # Don't match target to the same stale track id
            if track_id == self.locked_target:
                continue

            sim = self._hist_similarity(self.locked_profile, person.appearance_hist)
            if sim < self.reid_min_similarity:
                continue

            dist_score = self._distance_score(predicted, person.current_position, self.search_radius)
            if dist_score <= 0.0:
                continue

            # Size consistency check (helps reject nearby different persons)
            size_ok = True
            if last_h is not None:
                cand_h = max(1.0, float(person.current_box[3] - person.current_box[1]))
                ratio = cand_h / last_h
                size_ok = 0.70 <= ratio <= 1.35
            if not size_ok:
                continue

            # Appearance is primary; spatial prediction is secondary
            score = 0.75 * sim + 0.25 * dist_score
            if score > best_score:
                second_best = best_score
                best_score = score
                best_track = track_id
            elif score > second_best:
                second_best = score

        # Reject ambiguous matches to avoid jumping to nearby similar clothing.
        if best_track is None:
            return
        if best_score < self.reid_threshold:
            return
        if (best_score - second_best) < self.reid_min_margin:
            return

        if best_track is not None and best_score >= self.reid_threshold:
            old_target = self.locked_target
            self.locked_target = best_track
            self.lost_since = None

            # refresh locked appearance profile from new match
            matched = self.persons[best_track]
            if matched.appearance_hist is not None:
                self.locked_profile = matched.appearance_hist.copy()

            self.target_memory["last_reid_from"] = old_target
            self.target_memory["last_reid_to"] = best_track
            self.target_memory["last_reid_score"] = round(best_score, 3)

    def update(self, boxes, track_ids, frame=None):
        """Update tracker with new detections from YOLO."""
        current_time = time.time()
        seen_ids = set()

        for box, track_id in zip(boxes, track_ids):
            track_id = int(track_id)
            seen_ids.add(track_id)

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            position = (cx, cy)

            if track_id in self.persons:
                self.persons[track_id].update(position, box)
            else:
                self.persons[track_id] = TrackedPerson(track_id, position, box)

            if frame is not None:
                hist = self._extract_torso_hist(frame, box)
                if hist is not None:
                    self.persons[track_id].appearance_hist = hist
                self.persons[track_id].dress_color_name = self._dress_color_name(frame, box)

        # Mark unseen persons
        for track_id, person in list(self.persons.items()):
            if track_id not in seen_ids:
                time_since = current_time - person.last_seen
                if time_since > 2.0:
                    person.status = "LOST"
                else:
                    person.status = "SEARCHING"

        # Clean up persons lost for more than 10 seconds
        self.persons = {
            tid: p for tid, p in self.persons.items()
            if current_time - p.last_seen < 10.0
        }

        # Keep target memory updated while target is truly ACTIVE
        if self.locked_target is not None and self.locked_target in self.persons and self.persons[self.locked_target].status == "ACTIVE":
            locked_person = self.persons[self.locked_target]
            last_velocity = locked_person.velocities[-1] if locked_person.velocities else (0.0, 0.0)
            self.target_memory.update({
                'id': self.locked_signature_id if self.locked_signature_id is not None else locked_person.id,
                'current_track_id': locked_person.id,
                'last_position': locked_person.current_position,
                'last_box': locked_person.current_box,
                'dress_color': locked_person.dress_color_name,
                'last_velocity': last_velocity,
                'last_seen': current_time,
            })

            # If profile was missing at lock time, refresh once we get a good active crop
            if self.locked_profile is None and locked_person.appearance_hist is not None:
                self.locked_profile = locked_person.appearance_hist.copy()

            self.lost_since = None

        # If target id disappeared, try re-identification by appearance + position
        self._try_reidentify_locked_target(frame)

        # If still missing, mark lost timer
        locked_missing_or_not_active = False
        if self.locked_target is not None:
            if self.locked_target not in self.persons:
                locked_missing_or_not_active = True
            else:
                locked_missing_or_not_active = self.persons[self.locked_target].status != "ACTIVE"

        if locked_missing_or_not_active:
            if self.lost_since is None:
                self.lost_since = current_time

            # Auto-release if gone too long to avoid ghost target loops
            if (current_time - self.lost_since) > self.max_lost_time:
                self.unlock_target()

        return {
            'active_count': sum(1 for p in self.persons.values() if p.status == "ACTIVE"),
            'total_tracked': len(self.persons),
            'persons': self.persons,
        }

    def lock_target(self, track_id, frame=None):
        """Lock onto a specific person. Frame is saved for Re-ID (Step 3)."""
        if track_id in self.persons:
            person = self.persons[track_id]
            self.locked_target = track_id
            self.locked_signature_id = track_id
            self.locked_since = time.time()
            self.lost_since = None

            locked_hist = person.appearance_hist
            if locked_hist is None and frame is not None:
                locked_hist = self._extract_torso_hist(frame, person.current_box)
            self.locked_profile = locked_hist

            last_velocity = person.velocities[-1] if person.velocities else (0.0, 0.0)
            self.target_memory = {
                'id': self.locked_signature_id,
                'current_track_id': track_id,
                'last_position': person.current_position,
                'last_box': person.current_box,
                'dress_color': person.dress_color_name,
                'last_velocity': last_velocity,
                'locked_time': time.time(),
                'last_seen': time.time(),
            }
            return True
        return False

    def unlock_target(self):
        self.locked_target = None
        self.locked_signature_id = None
        self.locked_profile = None
        self.locked_since = None
        self.lost_since = None
        self.target_memory = {}

    def get_target_info(self):
        if self.locked_target is None:
            return None

        if self.locked_target in self.persons:
            person = self.persons[self.locked_target]
            return {
                'id': self.locked_signature_id if self.locked_signature_id is not None else person.id,
                'track_id': person.id,
                'position': person.current_position,
                'box': person.current_box,
                'dress_color': person.dress_color_name,
                'speed': person.speed,
                'direction': person.direction,
                'status': person.status,
                'distance': person.total_distance,
            }

        # Keep only lightweight LOST state data; do not force stale box rendering
        lost_data = {**self.target_memory}
        lost_data['status'] = 'LOST'
        lost_data['track_id'] = None
        if 'position' not in lost_data and 'last_position' in lost_data:
            lost_data['position'] = lost_data['last_position']
        return lost_data

    def find_person_at(self, x, y):
        """Find person at click position"""
        for track_id, person in self.persons.items():
            if person.status != "ACTIVE":
                continue
            box = person.current_box
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return track_id
        return None

    def draw_tracks(self, frame, show_trails=True):
        """Draw dots + trails for each tracked person"""
        for track_id, person in self.persons.items():
            if person.status != "ACTIVE":
                continue

            cx, cy = person.current_position
            color = person.color

            # Trail
            if show_trails and len(person.positions) > 1:
                points = person.positions
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    r = int(color[0] * alpha)
                    g = int(color[1] * alpha)
                    b = int(color[2] * alpha)
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, pt1, pt2, (r, g, b), thickness)

            # Dot
            cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
            cv2.circle(frame, (int(cx), int(cy)), 8, (255, 255, 255), 1)

            # ID label
            cv2.putText(frame, f"#{track_id}", (int(cx) + 10, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def draw_target(self, frame):
        """Draw locked target with highlight"""
        if self.locked_target is None:
            return frame

        target = self.get_target_info()
        if target is None:
            return frame

        status = target.get('status', 'LOST')

        # IMPORTANT: avoid drawing stale/ghost box when target is fully LOST
        if status == "LOST":
            label = f"TARGET #{target.get('id', 'N/A')} [LOST]"
            cv2.rectangle(frame, (10, 10), (260, 40), (0, 0, 180), -1)
            cv2.putText(frame, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            last_pos = target.get('last_position')
            dress_color = target.get('dress_color', 'Unknown')
            if last_pos is not None:
                cx, cy = int(last_pos[0]), int(last_pos[1])
                cv2.circle(frame, (cx, cy), 18, (0, 0, 255), 2)
                cv2.putText(frame, "LAST SEEN", (cx + 22, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                detail = f"Last: ({cx}, {cy})  Dress: {dress_color}"
                text_w = 430
                x1 = max(10, frame.shape[1] - text_w - 10)
                y1 = 10
                x2 = x1 + text_w
                y2 = 40
                cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
                cv2.putText(frame, detail, (x1 + 6, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            return frame

        box = target.get('box', target.get('last_box'))
        if box is None:
            return frame

        x1, y1, x2, y2 = [int(b) for b in box]

        color = {
            "ACTIVE": (0, 255, 255),
            "SEARCHING": (0, 165, 255),
        }.get(status, (0, 0, 255))

        # Thick box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Corner accents
        clen = min(25, (x2 - x1) // 4)
        for (pt, (dx, dy)) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))]:
            cv2.line(frame, pt, (pt[0] + dx * clen, pt[1]), color, 4)
            cv2.line(frame, pt, (pt[0], pt[1] + dy * clen), color, 4)

        # Label
        track_text = f" / TID {target.get('track_id')}" if target.get('track_id') is not None else ""
        label = f"TARGET #{target['id']} [{status}]{track_text}"
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 200, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Speed
        speed = target.get('speed', 0)
        cv2.putText(frame, f"Speed: {speed:.1f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Search animation when searching
        if status == "SEARCHING":
            t = time.time()
            radius = int(40 + (t % 1) * 30)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, center, radius, color, 2)

        return frame