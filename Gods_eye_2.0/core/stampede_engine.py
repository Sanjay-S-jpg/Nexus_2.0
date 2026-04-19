# core/stampede_engine.py - Stampede Detection Engine (Temporal + Gated)

import numpy as np
from collections import deque

from config import STAMPEDE_THRESHOLDS, ALERT_LEVELS


class StampedeEngine:
    """
    Robust stampede detection designed to avoid false positives.

    Key idea:
    A stampede is not just "people close together". It usually has:
    - Large affected crowd fraction
    - Sudden acceleration / density change
    - High movement coherence (same direction)
    - Sustained duration over multiple frames
    """

    def __init__(self, history_size=90):
        self.history_size = history_size
        self.thresholds = dict(STAMPEDE_THRESHOLDS)

        self.speed_history = deque(maxlen=history_size)
        self.density_history = deque(maxlen=history_size)
        self.coherence_history = deque(maxlen=history_size)
        self.nn_distance_history = deque(maxlen=history_size)

        self.position_history = {}  # track_id -> deque([(x, y), ...])
        self.missing_counts = {}    # track_id -> missing frame count

        self.current_risk = 0.0
        self.alert_level = 0
        self.components = {}
        self.sustained_counter = 0

    def _update_track_histories(self, positions):
        current_ids = set(positions.keys())

        for track_id, (cx, cy) in positions.items():
            if track_id not in self.position_history:
                self.position_history[track_id] = deque(maxlen=15)
            self.position_history[track_id].append((cx, cy))
            self.missing_counts[track_id] = 0

        # Age out missing tracks
        for track_id in list(self.position_history.keys()):
            if track_id not in current_ids:
                self.missing_counts[track_id] = self.missing_counts.get(track_id, 0) + 1
                if self.missing_counts[track_id] > 45:
                    self.position_history.pop(track_id, None)
                    self.missing_counts.pop(track_id, None)

    def _calculate_motion(self, positions):
        velocities = []
        moving_vectors = []

        for track_id, (cx, cy) in positions.items():
            if track_id not in self.position_history:
                continue
            trace = self.position_history[track_id]
            if len(trace) < 2:
                continue

            px, py = trace[-2]
            vx = cx - px
            vy = cy - py
            speed = float(np.hypot(vx, vy))

            velocities.append(speed)
            if speed > 2.0:
                moving_vectors.append((vx, vy, speed))

        avg_speed = float(np.mean(velocities)) if velocities else 0.0
        moving_count = len(moving_vectors)

        # Coherence: magnitude of average unit vector
        if moving_count >= 3:
            units = [(vx / s, vy / s) for vx, vy, s in moving_vectors if s > 1e-6]
            if len(units) >= 3:
                ux = float(np.mean([u[0] for u in units]))
                uy = float(np.mean([u[1] for u in units]))
                coherence = float(np.hypot(ux, uy))
            else:
                coherence = 0.0
        else:
            coherence = 0.0

        return avg_speed, coherence, moving_count

    def _calculate_acceleration(self, avg_speed):
        if len(self.speed_history) < 1:
            return 0.0
        prev_speed = self.speed_history[-1]
        return abs(avg_speed - prev_speed)

    def _calculate_density_change(self, grid):
        current_density = float(np.max(grid)) if grid.size > 0 else 0.0
        if len(self.density_history) < 1:
            return 0.0, current_density

        prev_density = self.density_history[-1]
        if prev_density <= 0:
            return 0.0, current_density

        change = (current_density - prev_density) / max(prev_density, 1.0)
        return max(0.0, float(change)), current_density

    def _calculate_edge_pressure(self, positions, frame_size):
        if not positions:
            return 0.0

        w, h = frame_size
        edge_margin = 0.15
        edge_count = 0

        for cx, cy in positions.values():
            if (
                cx < w * edge_margin
                or cx > w * (1 - edge_margin)
                or cy < h * edge_margin
                or cy > h * (1 - edge_margin)
            ):
                edge_count += 1

        return edge_count / max(len(positions), 1)

    def _median_nearest_neighbor_distance(self, positions):
        coords = list(positions.values())
        if len(coords) < 3:
            return 0.0

        dists = []
        for i, (x1, y1) in enumerate(coords):
            nearest = float("inf")
            for j, (x2, y2) in enumerate(coords):
                if i == j:
                    continue
                d = float(np.hypot(x2 - x1, y2 - y1))
                if d < nearest:
                    nearest = d
            if nearest < float("inf"):
                dists.append(nearest)

        if not dists:
            return 0.0
        return float(np.median(dists))

    def _calculate_compression(self, current_nn):
        if len(self.nn_distance_history) < 1 or current_nn <= 0:
            return 0.0

        prev_nn = self.nn_distance_history[-1]
        if prev_nn <= 0:
            return 0.0

        # Positive when people are getting closer quickly
        compression = (prev_nn - current_nn) / max(prev_nn, 1.0)
        return max(0.0, float(compression))

    def _get_alert_from_risk(self, risk):
        # ALERT_LEVELS format: {level: {name, color, threshold}}
        selected_level = 0
        for level in sorted(ALERT_LEVELS.keys()):
            if risk >= ALERT_LEVELS[level]["threshold"]:
                selected_level = level
        info = ALERT_LEVELS[selected_level]
        return selected_level, info["name"], info["color"]

    def analyze(self, positions, grid, frame_size):
        """Analyze crowd state and return stampede risk (0-100)."""
        self._update_track_histories(positions)

        people_count = len(positions)

        avg_speed, coherence, moving_count = self._calculate_motion(positions)
        acceleration = self._calculate_acceleration(avg_speed)
        density_change, current_density = self._calculate_density_change(grid)
        edge_pressure = self._calculate_edge_pressure(positions, frame_size)

        current_nn = self._median_nearest_neighbor_distance(positions)
        compression = self._calculate_compression(current_nn)

        affected_ratio = moving_count / max(people_count, 1)

        # Normalized scores
        velocity_score = min(1.0, avg_speed / max(self.thresholds["velocity"], 1e-6))
        acceleration_score = min(1.0, acceleration / max(self.thresholds["acceleration"], 1e-6))
        coherence_score = min(1.0, coherence / max(self.thresholds["coherence"], 1e-6))
        spike_score = min(1.0, density_change / max(self.thresholds["density_change"], 1e-6))
        edge_score = min(1.0, edge_pressure / max(self.thresholds["edge_pressure"], 1e-6))
        compression_score = min(1.0, compression / 0.25)
        scale_score = min(1.0, affected_ratio / 0.60)

        # Hard gates to suppress false positives
        gate_people = people_count >= self.thresholds["min_people"]
        gate_scale = affected_ratio >= 0.60
        gate_flow = (
            coherence >= self.thresholds["coherence"]
            and avg_speed >= self.thresholds["velocity"] * 0.60
        )
        gate_trigger = (
            acceleration >= self.thresholds["acceleration"]
            or density_change >= self.thresholds["density_change"]
            or compression >= 0.20
            or (avg_speed >= self.thresholds["velocity"] and coherence >= 0.75)
        )

        stampede_frame = gate_people and gate_scale and gate_flow and gate_trigger

        if stampede_frame:
            self.sustained_counter = min(
                self.sustained_counter + 1,
                int(self.thresholds["sustained_frames"]),
            )
        else:
            self.sustained_counter = max(0, self.sustained_counter - 2)

        sustained_ratio = self.sustained_counter / max(self.thresholds["sustained_frames"], 1)

        # Weighted base risk
        risk = (
            velocity_score * 0.18
            + acceleration_score * 0.18
            + coherence_score * 0.20
            + spike_score * 0.14
            + edge_score * 0.10
            + compression_score * 0.10
            + scale_score * 0.10
        ) * 100.0

        # Gate penalties (important for reducing false alarms)
        if not gate_people:
            risk *= 0.20
        if gate_people and not gate_scale:
            risk *= 0.45
        if gate_people and gate_scale and not gate_flow:
            risk *= 0.55

        # Temporal boosting only when risky behavior persists
        if stampede_frame:
            risk += 45.0 * sustained_ratio
            risk = max(risk, 48.0 + 45.0 * sustained_ratio)

        # Smooth risk and decay faster when crowd is small
        if people_count < self.thresholds["min_people"]:
            self.current_risk = 0.60 * self.current_risk + 0.40 * risk
        else:
            self.current_risk = 0.78 * self.current_risk + 0.22 * risk

        self.current_risk = float(np.clip(self.current_risk, 0, 100))

        self.alert_level, alert_name, alert_color = self._get_alert_from_risk(self.current_risk)

        self.components = {
            "velocity": round(velocity_score, 2),
            "acceleration": round(acceleration_score, 2),
            "coherence": round(coherence_score, 2),
            "spike": round(spike_score, 2),
            "edge": round(edge_score, 2),
            "compression": round(compression_score, 2),
            "scale": round(scale_score, 2),
            "sustain": round(sustained_ratio, 2),
        }

        # Update histories after metrics computed
        self.speed_history.append(avg_speed)
        self.density_history.append(current_density)
        self.coherence_history.append(coherence)
        if current_nn > 0:
            self.nn_distance_history.append(current_nn)

        return {
            "risk_score": int(self.current_risk),
            "alert_level": self.alert_level,
            "alert_name": alert_name,
            "alert_color": alert_color,
            "components": self.components,
            "people_count": people_count,
            "moving_count": moving_count,
            "affected_ratio": round(affected_ratio, 2),
            "avg_speed": round(avg_speed, 1),
            "acceleration": round(acceleration, 2),
            "coherence": round(coherence, 2),
            "density_change": round(density_change, 2),
            "edge_pressure": round(edge_pressure, 2),
            "compression": round(compression, 2),
            "sustained_frames": self.sustained_counter,
            "sustained_target": int(self.thresholds["sustained_frames"]),
            "stampede_frame": stampede_frame,
            "max_density": int(np.max(grid)) if grid.size > 0 else 0,
        }

    def reset(self):
        self.speed_history.clear()
        self.density_history.clear()
        self.coherence_history.clear()
        self.nn_distance_history.clear()
        self.position_history.clear()
        self.missing_counts.clear()
        self.current_risk = 0.0
        self.alert_level = 0
        self.sustained_counter = 0