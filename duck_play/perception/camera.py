"""Duck head pinhole camera model (W2).

Derives camera centre/axes from the duck head body pose and a fixed mounting
pitch, projects world 3D points to head-cam pixels. Rendering stays on the
same approximate view; labels come from this exact model (no colour guessing).
"""
from __future__ import annotations
import math
import numpy as np


class DuckHeadCam:
    def __init__(self, width=320, height=240, hfov_deg=62.0,
                 pitch_up_deg=None, fwd_m=0.09, up_m=0.03):
        self.w = width
        self.h = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        import os as _os
        self.pitch = math.radians(pitch_up_deg if pitch_up_deg is not None else float(_os.environ.get('DUCK_CAM_PITCH', '30')))
        self.fwd_m = fwd_m
        self.up_m = up_m
        self.hfov = math.radians(hfov_deg)   # IMX219 ~62 deg horizontal
        vfov = 2.0 * math.atan(math.tan(self.hfov / 2.0) * height / width)
        self.fx = (width / 2.0) / math.tan(self.hfov / 2.0)
        self.fy = (height / 2.0) / math.tan(vfov / 2.0)

    def pose(self, sim):
        d = sim.data
        pos = d.xpos[sim.head_id].copy()
        R = d.xmat[sim.head_id].reshape(3, 3)
        fwd = R[:, 0].copy()
        fwd /= np.linalg.norm(fwd)
        C = pos + fwd * self.fwd_m + np.array([0.0, 0.0, self.up_m])
        f = fwd * math.cos(self.pitch) + np.array([0.0, 0.0, 1.0]) * math.sin(self.pitch)
        f /= np.linalg.norm(f)
        ref = np.array([0.0, 0.0, 1.0])
        # MuJoCo camera image right is forward x world-up, not world-up x forward.
        right = np.cross(f, ref)
        n = np.linalg.norm(right)
        if n < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= n
        up = np.cross(right, f)
        return C, f, right, up

    def project(self, xyz, sim):
        C, f, right, up = self.pose(sim)
        v = xyz - C
        zc = float(np.dot(v, f))
        if zc <= 0.05:
            return None
        xc = float(np.dot(v, right))
        yc = float(np.dot(v, up))
        u = self.cx + self.fx * xc / zc
        vv = self.cy - self.fy * yc / zc
        if not (0 <= u < self.w and 0 <= vv < self.h):
            return None
        return u, vv




