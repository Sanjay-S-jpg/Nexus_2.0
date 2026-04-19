# ui/dashboard.py - Professional Dashboard UI

import cv2
import numpy as np
import time

class Dashboard:
    """
    Professional dashboard with:
    - Live statistics
    - Stampede intelligence panel
    - Zone density map
    - Target tracking info
    - Activity log
    """
    
    def __init__(self, width=400, height=720):
        self.width = width
        self.height = height
        self.logs = []
        self.max_logs = 12
        self.start_time = time.time()
        
    def add_log(self, message, log_type="INFO"):
        """Add entry to activity log"""
        timestamp = time.strftime("%H:%M:%S")
        entry = {
            'time': timestamp,
            'message': message,
            'type': log_type
        }
        self.logs.insert(0, entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop()
        print(f"[{timestamp}] {message}")
    
    def draw(self, stats):
        """
        Draw complete dashboard.
        
        Args:
            stats: dict with all statistics
            
        Returns:
            Dashboard image
        """
        dash = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(self.height):
            shade = int(15 + (i / self.height) * 10)
            dash[i, :] = (shade, shade, shade + 3)
        
        y = 0
        
        # === HEADER ===
        y = self._draw_header(dash, y, stats)
        
        # === LIVE STATISTICS ===
        y = self._draw_live_stats(dash, y, stats)
        
        # === STAMPEDE INTELLIGENCE ===
        y = self._draw_stampede_panel(dash, y, stats)
        
        # === ZONE DENSITY MAP ===
        y = self._draw_density_map(dash, y, stats)
        
        # === TARGET TRACKING ===
        y = self._draw_target_panel(dash, y, stats)
        
        # === MODE INDICATORS ===
        y = self._draw_modes(dash, y, stats)
        
        # === ACTIVITY LOG ===
        y = self._draw_activity_log(dash, y)
        
        # === FOOTER ===
        self._draw_footer(dash)
        
        return dash
    
    def _draw_header(self, dash, y, stats):
        """Draw header section"""
        cv2.rectangle(dash, (0, 0), (self.width, 55), (25, 25, 30), -1)
        cv2.putText(dash, "CROWD INTELLIGENCE", (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(dash, "Real-Time Surveillance System", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        # Status indicator
        color = (0, 255, 0) if stats.get('system_ok', True) else (0, 0, 255)
        cv2.circle(dash, (self.width - 25, 28), 8, color, -1)
        
        return 60
    
    def _draw_live_stats(self, dash, y, stats):
        """Draw live statistics panel"""
        h = 80
        self._draw_panel(dash, y, h, "LIVE STATISTICS")
        
        # People count
        count = stats.get('people_count', 0)
        peak = stats.get('peak_count', 0)
        cv2.putText(dash, f"People: {count}", (20, y + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(dash, f"(Peak: {peak})", (140, y + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # FPS
        fps = stats.get('fps', 0)
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(dash, f"FPS: {int(fps)}", (20, y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1)
        
        # Weapons
        weapons = stats.get('weapon_count', 0)
        w_color = (0, 0, 255) if weapons > 0 else (0, 255, 0)
        cv2.putText(dash, f"Weapons: {weapons}", (100, y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, w_color, 1)
        
        # Tracked
        tracked = stats.get('tracked_count', 0)
        cv2.putText(dash, f"Tracked: {tracked}", (220, y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Detection method
        method = stats.get('detection_method', 'YOLO')
        cv2.putText(dash, f"[{method}]", (320, y + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        return y + h + 10
    
    def _draw_stampede_panel(self, dash, y, stats):
        """Draw stampede intelligence panel"""
        h = 140
        self._draw_panel(dash, y, h, "STAMPEDE INTELLIGENCE")
        
        stampede = stats.get('stampede', {})
        risk = stampede.get('risk_score', 0)
        level = stampede.get('alert_level', 0)
        name = stampede.get('alert_name', 'SAFE')
        color = stampede.get('alert_color', (0, 255, 0))
        
        # Big percentage
        cv2.putText(dash, f"{risk}%", (20, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(dash, name, (100, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Risk bar
        bar_w = self.width - 50
        cv2.rectangle(dash, (20, y + 65), (20 + bar_w, y + 78), (40, 40, 45), -1)
        fill = int((risk / 100) * bar_w)
        if fill > 0:
            cv2.rectangle(dash, (20, y + 65), (20 + fill, y + 78), color, -1)
        
        # Threshold markers
        for thresh in [25, 50, 75]:
            tx = 20 + int((thresh / 100) * bar_w)
            cv2.line(dash, (tx, y + 65), (tx, y + 78), (80, 80, 80), 1)
        
        # Component bars
        components = stampede.get('components', {})
        comp_names = ['COHER', 'ACCEL', 'SPEED', 'SPIKE', 'EDGE']
        comp_keys = ['coherence', 'acceleration', 'velocity', 'spike', 'edge']
        
        bar_start_y = y + 90
        spacing = 70
        
        for i, (name, key) in enumerate(zip(comp_names, comp_keys)):
            value = components.get(key, 0)
            cx = 25 + (i * spacing)
            
            # Label
            cv2.putText(dash, name, (cx - 5, bar_start_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1)
            
            # Vertical bar
            bar_x = cx + 12
            bar_h = 30
            cv2.rectangle(dash, (bar_x, bar_start_y + 5),
                         (bar_x + 10, bar_start_y + 5 + bar_h), (40, 40, 45), -1)
            
            # Fill
            fill_h = int(value * bar_h)
            if fill_h > 0:
                bar_color = (0, 200, 200) if value < 0.5 else (0, 255, 255) if value < 0.8 else (0, 100, 255)
                cv2.rectangle(dash, (bar_x + 1, bar_start_y + 5 + bar_h - fill_h),
                             (bar_x + 9, bar_start_y + 5 + bar_h), bar_color, -1)
            
            # Value
            cv2.putText(dash, f"{value:.1f}", (cx, bar_start_y + 48),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        return y + h + 10
    
    def _draw_density_map(self, dash, y, stats):
        """Draw zone density map"""
        h = 90
        self._draw_panel(dash, y, h, "ZONE DENSITY MAP")
        
        grid = stats.get('density_grid', np.zeros((4, 4)))
        grid_h, grid_w = grid.shape
        
        map_x, map_y = 20, y + 30
        map_w, map_h = self.width - 50, 50
        cell_w, cell_h = map_w // grid_w, map_h // grid_h
        
        max_density = max(grid.max(), 1)
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                cx1, cy1 = map_x + gx * cell_w, map_y + gy * cell_h
                cx2, cy2 = cx1 + cell_w - 2, cy1 + cell_h - 2
                
                density = grid[gy, gx]
                ratio = density / max(max_density, 5)
                
                if ratio < 0.3:
                    color = (0, 100, 0)
                elif ratio < 0.6:
                    color = (0, 180, 0)
                elif ratio < 0.8:
                    color = (0, 200, 200)
                else:
                    color = (0, 0, 200)
                
                cv2.rectangle(dash, (cx1, cy1), (cx2, cy2), color, -1)
                cv2.rectangle(dash, (cx1, cy1), (cx2, cy2), (50, 50, 55), 1)
                
                if density > 0:
                    cv2.putText(dash, str(int(density)),
                               (cx1 + cell_w // 3, cy1 + cell_h // 2 + 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return y + h + 10
    
    def _draw_target_panel(self, dash, y, stats):
        """Draw target tracking panel"""
        h = 70
        self._draw_panel(dash, y, h, "TARGET TRACKING", color=(0, 255, 255))
        
        target = stats.get('target', None)
        
        if target:
            status = target.get('status', 'ACTIVE')
            
            # Status color
            if status == 'ACTIVE':
                status_color = (0, 255, 255)
            elif status == 'SEARCHING':
                pulse = int((time.time() * 4) % 2)
                status_color = (0, 165, 255) if pulse else (0, 200, 255)
            else:
                status_color = (0, 0, 255)
            
            cv2.circle(dash, (self.width - 25, y + 35), 8, status_color, -1)
            
            # Info
            cv2.putText(dash, f"ID: {target.get('id', 'N/A')}", (20, y + 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(dash, f"Speed: {target.get('speed', 0):.1f}", (90, y + 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Status badge
            cv2.rectangle(dash, (200, y + 28), (280, y + 48), status_color, -1)
            cv2.putText(dash, status, (205, y + 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
            # Position
            pos = target.get('position', (0, 0))
            cv2.putText(dash, f"Pos: ({pos[0]}, {pos[1]})", (20, y + 58),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        else:
            cv2.circle(dash, (self.width - 25, y + 35), 8, (60, 60, 60), -1)
            cv2.putText(dash, "No target - Click to track", (20, y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return y + h + 10
    
    def _draw_modes(self, dash, y, stats):
        """Draw mode indicators"""
        h = 35
        cv2.rectangle(dash, (10, y), (self.width - 10, y + h), (30, 30, 35), -1)
        
        modes = [
            ('HEAT', stats.get('heatmap_mode', False)),
            ('NIGHT', stats.get('night_mode', False)),
            ('DEMO', stats.get('demo_mode', False))
        ]
        
        mode_x = 20
        for name, is_on in modes:
            color = (0, 255, 255) if is_on else (50, 50, 55)
            text_color = (0, 0, 0) if is_on else (80, 80, 80)
            
            cv2.rectangle(dash, (mode_x, y + 5), (mode_x + 55, y + 28), color, -1 if is_on else 1)
            cv2.putText(dash, name, (mode_x + 8, y + 21),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            mode_x += 62
        
        # Source indicator
        source = stats.get('source_type', 'VIDEO')
        cv2.putText(dash, source, (mode_x + 30, y + 21),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        
        return y + h + 10
    
    def _draw_activity_log(self, dash, y):
        """Draw activity log"""
        remaining_h = self.height - y - 35
        self._draw_panel(dash, y, remaining_h, "ACTIVITY LOG", color=(255, 255, 0))
        
        log_y = y + 35
        for i, entry in enumerate(self.logs):
            if log_y > self.height - 50:
                break
            
            # Color based on type
            if entry['type'] == 'STAMPEDE':
                color = (0, 165, 255)
            elif entry['type'] == 'WEAPON':
                color = (0, 0, 255)
            elif entry['type'] == 'TARGET':
                color = (0, 255, 255)
            elif entry['type'] == 'SUCCESS':
                color = (0, 255, 0)
            else:
                color = (140, 140, 140)
            
            text = f"[{entry['time']}] {entry['message']}"
            cv2.putText(dash, text[:45], (15, log_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
            log_y += 16
        
        return log_y
    
    def _draw_footer(self, dash):
        """Draw footer"""
        cv2.line(dash, (10, self.height - 28), (self.width - 10, self.height - 28), (50, 50, 55), 1)
        
        runtime = int(time.time() - self.start_time)
        mins, secs = runtime // 60, runtime % 60
        
        cv2.putText(dash, f"Runtime: {mins}m {secs}s", (15, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
        cv2.putText(dash, "Q=Quit H=Heat T=Test", (self.width - 140, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
    
    def _draw_panel(self, dash, y, h, title, color=(255, 255, 0)):
        """Draw a panel with title"""
        cv2.rectangle(dash, (10, y), (self.width - 10, y + h), (30, 30, 35), -1)
        cv2.rectangle(dash, (10, y), (self.width - 10, y + h), (50, 50, 55), 1)
        cv2.putText(dash, title, (20, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)