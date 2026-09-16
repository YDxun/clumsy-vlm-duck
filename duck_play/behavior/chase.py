"""Chase-ball state machine (autonomous): SEARCH -> ALIGN -> APPROACH ->
FOOT_SELECT -> KICK -> (re-detect). Uses the same kick geometry as the
validated fixture (test_kick_physics.py): ball ~0.09 m in front of the chosen
foot and ~+/-0.042 m lateral before requesting kickL/kickR.
"""
from __future__ import annotations
import math
from ..config import CHASE


class ChaseController:
    def __init__(self, cfg=None):
        self.cfg = dict(CHASE)
        if cfg:
            self.cfg.update(cfg)
        self.state = "SEARCH"        # SEARCH|ALIGN|APPROACH|FOOT_SELECT|KICK|DOWN
        self.state_time = 0.0
        self.touch_count = 0
        self.kick_attempts = 0
        self.side = None             # kickL | kickR chosen for current approach
        self._pending_kick = None

    def _clip(self, v, lo, hi):
        return max(lo, min(hi, v))

    def decide(self, perception, sim, dt):
        """Return (cmd[3], kick_request|None, note). Pure decision; no physics."""
        self.state_time += dt
        ball = perception.ball
        cfg = self.cfg

        if self.state == "DOWN":
            return [0.0, 0.0, 0.0], None, "down"

        if ball is None or not math.isfinite(ball.range_m):
            self.state = "SEARCH"
            return [0.0, 0.0, cfg["search_wz_radps"]], None, "search"

        yaw, rng = ball.bearing_yaw, ball.range_m

        if self.state in ("FOOT_SELECT", "KICK"):
            return [0.0, 0.0, 0.0], None, self.state

        # --- choose foot + desired lateral offset once reasonably close ---
        if rng <= cfg["side_switch_range_m"]:
            ks = cfg.get("kick_side", "auto")
            if ks == "left":
                self.side = "kickL"
            elif ks == "right":
                self.side = "kickR"
            else:
                self.side = "kickL" if yaw >= 0.0 else "kickR"
            ty = cfg["side_offset_m"] if self.side == "kickL" else -cfg["side_offset_m"]
        else:
            self.side = None
            ty = 0.0
        yaw_d = math.atan2(ty, max(rng, cfg["kick_range_m"] * 0.7))
        err = yaw - yaw_d

        if abs(err) > cfg["align_ang_rad"]:
            self.state = "ALIGN"
            wz = self._clip(cfg["kp_yaw"] * err, -cfg["wz_max"], cfg["wz_max"])
            return [0.0, 0.0, wz], None, "align"

        self.state = "APPROACH"
        if rng <= cfg["kick_range_m"]:
            self.state = "FOOT_SELECT"
            side = self.side or ("kickL" if yaw >= 0.0 else "kickR")
            self._pending_kick = side
            return [0.0, 0.0, 0.0], side, "foot_select"

        vx = self._clip(cfg["kp_vx"] * (rng - cfg["kick_range_m"]), 0.0, cfg["vx_max"])
        if rng > cfg["kick_range_m"]:
            vx = max(vx, cfg["approach_min_vx"])
        wz = self._clip(cfg["kp_yaw"] * err, -cfg["wz_max"], cfg["wz_max"])
        return [vx, 0.0, wz], None, "approach"

    def mark_kick_started(self):
        self.state = "KICK"
        self.state_time = 0.0
        self._pending_kick = None

    def mark_kick_done(self, touched: bool):
        self.kick_attempts += 1
        if touched:
            self.touch_count += 1
        self.state = "SEARCH"
        self.state_time = 0.0
