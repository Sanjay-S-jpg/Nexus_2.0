# main.py - Gods Eye 2.0: AI Crowd Stampede Detection System
#
# Features:
#   - Crowd detection & counting (YOLO11n)
#   - Stampede risk analysis (physics-based with temporal window)
#   - Person tracking with Re-ID (BoT-SORT + appearance matching)
#   - Fight detection (pose-based)  [Step 4]
#   - Lost child detection          [Step 5]
#   - Weapon detection (knife/gun)  [Step 6]
#   - Demo mode (simulated crowds)  [Step 7]
#   - Heatmap trails
#   - Dashboard + Telegram/Desktop alerts
#
# Controls:
#   Q - Quit          H - Toggle Heatmap
#   N - Night Vision  R - Reset Target
#   T - Test Alerts   C - Clear Heatmap
#   S - Save Heatmap  D - Demo Mode (1/2/3 for scenarios)
#   CLICK - Lock target on person

import cv2
import numpy as np
import time
import sys
import os

from ultralytics import YOLO

from config import *
from core.crowd_detector import CrowdDetector
from core.person_tracker import PersonTracker
from core.stampede_engine import StampedeEngine
from core.weapon_detector import WeaponDetector
from core.heatmap import Heatmap
from alerts.telegram_alert import TelegramAlert
from alerts.desktop_alert import DesktopAlert
from ui.dashboard import Dashboard
from utils.helpers import apply_night_vision, generate_report


class GodsEye:
    """Main application class for Gods Eye 2.0"""

    def __init__(self, video_source=None):
        print("\n" + "=" * 60)
        print("  GODS EYE 2.0 - AI Crowd Stampede Detection")
        print("=" * 60)

        self.source = video_source or VIDEO_SOURCE

        # ── Load YOLO11n on GPU ──
        print(f"\n[Loading] YOLO11n model on {DEVICE}")
        self.model = YOLO(YOLO_MODEL)
        self.model.to(DEVICE)
        # Warm up the model (first inference is slow, this gets it ready)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, device=DEVICE)
        print(f"[OK] Model loaded and warmed up on {DEVICE}")

        # ── Initialize all detection modules ──
        print("[Loading] Initializing components...")
        self.crowd_detector = CrowdDetector(grid_size=GRID_SIZE)
        self.person_tracker = PersonTracker()
        self.stampede_engine = StampedeEngine()
        self.weapon_detector = WeaponDetector()
        self.heatmap = None  # Created after first frame gives us dimensions
        self.dashboard = Dashboard(width=DASHBOARD_WIDTH, height=DISPLAY_HEIGHT)

        # ── Alerts ──
        self.telegram = TelegramAlert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.desktop = DesktopAlert()

        # ── State flags ──
        self.heatmap_mode = False
        self.night_mode = False
        self.demo_mode = DEMO_ENABLED
        self.running = True

        # ── Stats dict (shared with dashboard) ──
        self.stats = {
            'fps': 0,
            'people_count': 0,
            'peak_count': 0,
            'weapon_count': 0,
            'weapons_detected': 0,
            'tracked_count': 0,
            'max_stampede_risk': 0,
            'total_tracked': 0,
            'start_time': time.time(),
            'detection_method': 'YOLO11n',
        }

        # FPS smoothing (average over last 30 frames)
        self.fps_history = []

        # Mouse click position
        self.click_pos = None
        self._last_target_status = None

        print("[OK] All components initialized!")
        self._print_controls()

    # ─────────────────────────────────────────────
    # CONTROLS HELP
    # ─────────────────────────────────────────────
    def _print_controls(self):
        print("\n" + "-" * 45)
        print("CONTROLS:")
        print("  Q - Quit           H - Heatmap toggle")
        print("  N - Night Vision   R - Release target")
        print("  T - Test Alerts    C - Clear heatmap")
        print("  S - Save Heatmap   D - Demo mode on/off")
        print("  1/2/3 - Demo scenarios (when D is on)")
        print("  CLICK on person - Lock target")
        print("-" * 45 + "\n")

    # ─────────────────────────────────────────────
    # MOUSE
    # ─────────────────────────────────────────────
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_pos = (x, y)

    # ─────────────────────────────────────────────
    # CORE: Process one frame
    # ─────────────────────────────────────────────
    def process_frame(self, frame):
        h, w = frame.shape[:2]

        if self.heatmap is None:
            self.heatmap = Heatmap(w, h)

        ai_frame = frame.copy()
        if self.night_mode:
            ai_frame = apply_night_vision(ai_frame)

        # ── Run YOLO11n tracking (BoT-SORT via config) ──
        tracker_cfg = os.path.join(BASE_DIR, "tracker_config.yaml")
        results = self.model.track(
            ai_frame,
            persist=True,
            verbose=False,
            conf=PERSON_CONFIDENCE,
            device=DEVICE,
            imgsz=640,
            tracker=tracker_cfg,
        )

        # ── Extract all detections ──
        boxes = np.empty((0, 4))
        track_ids = np.array([], dtype=int)
        classes = np.array([], dtype=int)
        confidences = np.array([], dtype=float)
        person_positions = []
        positions_dict = {}
        person_boxes_dict = {}  # tid -> [x1,y1,x2,y2]

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, tid, cls in zip(boxes, track_ids, classes):
                if cls == 0:  # Person
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    person_positions.append((cx, cy))
                    positions_dict[tid] = (cx, cy)
                    person_boxes_dict[tid] = box

        # ── Crowd counting ──
        crowd_result = self.crowd_detector.count(frame, person_positions)
        people_count = crowd_result['count']

        self.stats['people_count'] = people_count
        self.stats['detection_method'] = crowd_result['method']
        if people_count > self.stats['peak_count']:
            self.stats['peak_count'] = people_count

        # ── Person tracking update (always run, even with empty detections) ──
        if len(person_boxes_dict) > 0:
            p_boxes = np.array(list(person_boxes_dict.values()))
            p_ids = np.array(list(person_boxes_dict.keys()))
        else:
            p_boxes = np.empty((0, 4), dtype=float)
            p_ids = np.array([], dtype=int)

        track_result = self.person_tracker.update(p_boxes, p_ids, frame)
        self.stats['tracked_count'] = track_result['active_count']
        self.stats['total_tracked'] = track_result['total_tracked']

        # ── Click to lock target ──
        if self.click_pos:
            x, y = self.click_pos
            target_id = self.person_tracker.find_person_at(x, y)
            if target_id:
                self.person_tracker.lock_target(target_id, frame)
                self.dashboard.add_log(f"TARGET LOCKED: #{target_id}", "TARGET")
            self.click_pos = None

        # ── Stampede detection ──
        grid = crowd_result.get('grid', np.zeros(GRID_SIZE))
        stampede_result = self.stampede_engine.analyze(positions_dict, grid, (w, h))
        self.stats['stampede'] = stampede_result

        if stampede_result['risk_score'] > self.stats['max_stampede_risk']:
            self.stats['max_stampede_risk'] = stampede_result['risk_score']

        if stampede_result['alert_level'] >= 2:
            self.dashboard.add_log(
                f"STAMPEDE RISK: {stampede_result['risk_score']}%", "STAMPEDE"
            )
            self.telegram.stampede_alert(
                stampede_result['risk_score'], stampede_result['alert_level'], frame
            )
            self.desktop.stampede_alert(
                stampede_result['risk_score'], stampede_result['alert_level']
            )

        # ── Weapon detection ──
        weapon_result = self.weapon_detector.detect(
            boxes, classes, confidences, person_positions
        )
        self.stats['weapon_count'] = weapon_result['count']

        for weapon in weapon_result['alerts']:
            self.stats['weapons_detected'] += 1
            self.dashboard.add_log(
                f"WEAPON: {weapon['type']} ({int(weapon['threat']*100)}%)", "WEAPON"
            )
            self.telegram.weapon_alert(weapon['type'], weapon['threat'], frame)
            self.desktop.weapon_alert(weapon['type'], weapon['threat'])

        # ── Update heatmap ──
        self.heatmap.update(positions_dict)

        # ── Feed stats to dashboard ──
        self.stats['density_grid'] = grid
        self.stats['target'] = self.person_tracker.get_target_info()
        self.stats['heatmap_mode'] = self.heatmap_mode
        self.stats['night_mode'] = self.night_mode
        self.stats['demo_mode'] = self.demo_mode

        # Log lost target details once on status transition
        target = self.stats['target']
        curr_status = target.get('status') if target else None
        if curr_status == 'LOST' and self._last_target_status != 'LOST':
            last_pos = target.get('last_position') or target.get('position')
            dress_color = target.get('dress_color', 'Unknown')
            if last_pos is not None:
                lx, ly = int(last_pos[0]), int(last_pos[1])
                self.dashboard.add_log(
                    f"TARGET LOST -> Last ({lx}, {ly}), Dress: {dress_color}",
                    "TARGET"
                )
            else:
                self.dashboard.add_log(
                    f"TARGET LOST -> Dress: {dress_color}",
                    "TARGET"
                )
        self._last_target_status = curr_status

        return crowd_result, stampede_result, weapon_result

    # ─────────────────────────────────────────────
    # DRAW: Render all visualizations on frame
    # ─────────────────────────────────────────────
    def draw_frame(self, frame, crowd_result, stampede_result, weapon_result):
        display = frame.copy()

        if self.night_mode:
            display = apply_night_vision(display)

        if self.heatmap_mode:
            display = self.heatmap.overlay(display, alpha=0.4)

        # Crowd dots
        display = self.crowd_detector.draw(display, crowd_result)

        # Person tracks + trails
        display = self.person_tracker.draw_tracks(display, show_trails=True)

        # Locked target highlight
        display = self.person_tracker.draw_target(display)

        # Weapons
        display = self.weapon_detector.draw(display)
        display = self.weapon_detector.draw_banner(display)

        # Density grid overlay
        h, w = display.shape[:2]
        grid = crowd_result.get('grid', np.zeros(GRID_SIZE))
        grid_h, grid_w = grid.shape
        cell_w, cell_h = w // grid_w, h // grid_h

        for gy in range(grid_h):
            for gx in range(grid_w):
                x1, y1 = gx * cell_w, gy * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                cv2.rectangle(display, (x1, y1), (x2, y2), (40, 40, 40), 1)

                if grid[gy, gx] >= DENSITY_CRITICAL:
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        # Stampede warning banner
        if stampede_result['alert_level'] >= 2:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            text = f"STAMPEDE {stampede_result['alert_name']}: {stampede_result['risk_score']}%"
            cv2.putText(display, text, (w // 2 - 200, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Bottom info bar
        tip = "Click person to track | Q=Quit H=Heatmap D=Demo T=Test"
        cv2.putText(display, tip, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return display

    # ─────────────────────────────────────────────
    # KEYBOARD HANDLER
    # ─────────────────────────────────────────────
    def handle_key(self, key, last_frame):
        if key == ord('q'):
            self.dashboard.add_log("User quit", "INFO")
            self.running = False

        elif key == ord('h'):
            self.heatmap_mode = not self.heatmap_mode
            self.dashboard.add_log(f"Heatmap: {'ON' if self.heatmap_mode else 'OFF'}", "INFO")

        elif key == ord('n'):
            self.night_mode = not self.night_mode
            self.dashboard.add_log(f"Night Vision: {'ON' if self.night_mode else 'OFF'}", "INFO")

        elif key == ord('r'):
            self.person_tracker.unlock_target()
            self.dashboard.add_log("Target released", "TARGET")

        elif key == ord('c'):
            if self.heatmap:
                self.heatmap.reset()
                self.dashboard.add_log("Heatmap cleared", "INFO")

        elif key == ord('s'):
            if self.heatmap and last_frame is not None:
                self.heatmap.save(
                    f"Reports/heatmap_{int(time.time())}.png",
                    background=last_frame
                )
                self.dashboard.add_log("Heatmap saved", "SUCCESS")

        elif key == ord('t'):
            self.dashboard.add_log("Testing alerts...", "INFO")
            self.telegram.test()
            self.desktop.stampede_alert(75, 2)

        elif key == ord('d'):
            self.demo_mode = not self.demo_mode
            self.dashboard.add_log(f"Demo Mode: {'ON' if self.demo_mode else 'OFF'}", "INFO")

    # ─────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            return

        # Determine source type
        if isinstance(self.source, int):
            self.stats['source_type'] = "WEBCAM"
        elif isinstance(self.source, str) and self.source.startswith("http"):
            self.stats['source_type'] = "STREAM"
        else:
            self.stats['source_type'] = "FILE"

        # Setup OpenCV window (NORMAL = resizable, avoids lag)
        window_name = "Gods Eye 2.0 - AI Crowd Surveillance"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, DISPLAY_WIDTH + DASHBOARD_WIDTH, DISPLAY_HEIGHT)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        prev_time = time.time()
        self.dashboard.add_log("System started", "SUCCESS")
        last_frame = None

        while self.running:
            ret, frame = cap.read()

            if not ret:
                if self.stats['source_type'] == "FILE":
                    # Loop video for demo purposes
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.dashboard.add_log("Video looped", "INFO")
                    continue
                continue

            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            last_frame = frame.copy()

            # FPS calculation (smoothed over 30 frames)
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 30.0
            prev_time = curr_time
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            self.stats['fps'] = np.mean(self.fps_history)

            # Process + Draw
            crowd_result, stampede_result, weapon_result = self.process_frame(frame)
            display = self.draw_frame(frame, crowd_result, stampede_result, weapon_result)

            # Dashboard
            dashboard = self.dashboard.draw(self.stats)
            combined = np.hstack((display, dashboard))

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_key(key, last_frame)

        # ── Cleanup ──
        cap.release()
        cv2.destroyAllWindows()

        # ── Generate report ──
        self.stats['duration'] = f"{int(time.time() - self.stats['start_time'])}s"
        self.stats['avg_fps'] = round(np.mean(self.fps_history), 1) if self.fps_history else 0

        heatmap_img = None
        if self.heatmap and last_frame is not None:
            heatmap_img = self.heatmap.overlay(last_frame, alpha=0.6)
        generate_report(self.stats, heatmap_img)

        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"  Duration:        {self.stats['duration']}")
        print(f"  Average FPS:     {self.stats['avg_fps']}")
        print(f"  Peak Crowd:      {self.stats['peak_count']}")
        print(f"  Max Stampede:    {self.stats['max_stampede_risk']}%")
        print(f"  Weapons Found:   {self.stats['weapons_detected']}")
        print(f"  Persons Tracked: {self.stats['total_tracked']}")
        print("=" * 50)


def main():
    source = VIDEO_SOURCE
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg

    app = GodsEye(video_source=source)
    app.run()


if __name__ == "__main__":
    main()