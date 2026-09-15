#!/usr/bin/env python3
"""
MicroDuck cloud display server — replay renderer + web host + command relay.

The RDK X5 runs the authoritative sim (physics + policy + behaviors) on-board
and streams qpos at 50 Hz. This server replays that state into its own MuJoCo
model purely to RENDER the video and duck-cam for the web page — MuJoCo runs
here for display, the control truth lives on the X5.

  browser --wss /ws--> commands, mode, actions
  browser <--wss /ws-- video frames + telemetry
  X5      --wss /board--> state stream (qpos, duck pose, ...)
  X5      <--wss /board--  command relay (cmd / action / reset)
  X5      <--wss /vision-- duck-cam JPEG frames (server-rendered)
  X5      --wss /vision--> BPU detections + auto-follow steering

When no board is attached, a local fallback sim runs so the page still demos.
"""

import asyncio
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import onnxruntime as ort
from PIL import Image
import io
from aiohttp import web, WSMsgType
from local_kick import LocalKick, KICK_POLICIES, TRICK_POLICIES

HERE = Path(__file__).resolve().parent
SCENE_XML = HERE / "microduck_rl/src/mjlab_microduck/robot/microduck/scene_pretty.xml"
POLICY_DIR = HERE / "microduck/policies"

# --- Harness Duck C1 integration (opt-in: DUCKAGENT_ENABLE=1) ---
DUCKAGENT_ENABLE = os.environ.get("DUCKAGENT_ENABLE") == "1"
DUCK_VLM_ENABLE = os.environ.get("DUCK_VLM_ENABLE", "1") == "1"

# --- Verified DuckVLM scene pack selection ---
DUCK_SCENE_ID = os.environ.get("DUCK_SCENE_ID", "").strip()
if not DUCK_SCENE_ID:
    _scene_file = HERE / ".duck_scene_id"
    if _scene_file.exists():
        DUCK_SCENE_ID = _scene_file.read_text(encoding="utf-8").strip()
SCENE_SPEC = None
if DUCK_SCENE_ID:
    try:
        from duck_vlm.scenes import catalog as _scene_catalog
        SCENE_SPEC = _scene_catalog().load(DUCK_SCENE_ID)
        SCENE_XML = SCENE_SPEC.xml
    except Exception as exc:
        print(f"[scene] load failed for {DUCK_SCENE_ID}: {exc}", flush=True)
        DUCK_SCENE_ID = ""
if DUCKAGENT_ENABLE and not DUCK_SCENE_ID:
    try:
        from duck_agent import scenes as _da_scenes
        SCENE_XML = _da_scenes.ensure()
    except Exception as exc:
        print(f"[agent] scene override failed, keep default: {exc}", flush=True)
        DUCKAGENT_ENABLE = False


DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

PHYS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION
FRAME_MAIN = b"\x01"
FRAME_HEADCAM = b"\x02"
AUTOCMD_STALE_S = 0.6


def quat_rotate_inverse(quat, vec):
    w = quat[0]; xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


class LocalSim:
    """Fallback physics when no X5 board is attached (server-side demo)."""
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.model.opt.timestep = PHYS_DT
        self.data = mujoco.MjData(self.model)
        kt = 0.3459739511711113; lim = kt * 1.75
        self.model.actuator_forcerange[:, 0] = -lim
        self.model.actuator_forcerange[:, 1] = lim
        self.model.actuator_forcelimited[:] = 1
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "yaw_roll_motion")
        self.ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.imu_gyro_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
        self.n_joints = self.model.nu
        self.joint_qpos_idx = [int(self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]) for i in range(self.model.nu)]
        self.joint_qvel_idx = [int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]) for i in range(self.model.nu)]
        # Duck head joints, addressable directly so head actions can override the
        # walking policy's head targets without touching locomotion.
        self.head_ctrl_idx = {}
        for _i in range(self.model.nu):
            _jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                       int(self.model.actuator_trnid[_i, 0]))
            if _jname in ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll'):
                self.head_ctrl_idx[_jname] = _i
        self.default_pose = DEFAULT_POSE[: self.n_joints]
        fj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qpos_adr = int(self.model.jnt_qposadr[fj])
        bj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos_adr = int(self.model.jnt_qposadr[bj])
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.reset()

    def reset(self):
        d = self.data
        if self.model.nkey > 0:
            # Scene packs ship an `initial` keyframe holding the scene-correct robot
            # spawn and object layout. Using it keeps tasks.yaml robot_spawn honest
            # instead of resetting every scene to the workspace origin.
            mujoco.mj_resetDataKeyframe(self.model, d, 0)
            d.qvel[:] = 0
            d.ctrl[:] = self.default_pose
            mujoco.mj_forward(self.model, d)
            self.last_action = np.zeros(self.n_joints, dtype=np.float32)
            return
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qpos_adr:self.qpos_adr+7] = [0, 0, 0.125, 1, 0, 0, 0]
        for i, qi in enumerate(self.joint_qpos_idx):
            d.qpos[qi] = self.default_pose[i]
        d.qpos[self.ball_qpos_adr:self.ball_qpos_adr+7] = [0.9, 0.0, 0.035, 1, 0, 0, 0]
        d.qvel[:] = 0
        d.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.model, d)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)


class PolicyBank:
    """Server-side fallback policies (delay-trained, used only when no X5)."""
    def __init__(self):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; so.log_severity_level = 3
        self.sessions = {}
        for name in ("alpha_walking", "alpha_stand", *KICK_POLICIES.values(), *TRICK_POLICIES.values()):
            try:
                s = ort.InferenceSession(str(POLICY_DIR / f"{name}.onnx"), sess_options=so,
                                         providers=["CPUExecutionProvider"])
                assert s.get_inputs()[0].shape == [1, 61], 'expected input [1,61]'
                assert s.get_outputs()[0].shape == [1, 14], 'expected output [1,14]'
                self.sessions[name] = (s, s.get_inputs()[0].name, s.get_outputs()[0].name)
            except Exception as exc:
                if name not in KICK_POLICIES.values():
                    raise
                print(f'[policy] {name} unavailable: {exc}', flush=True)

    def infer(self, name, obs):
        s, i, o = self.sessions[name]
        result = s.run([o], {i: obs.reshape(1, -1)})[0].squeeze(0)
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError(f'{name}: invalid action output')
        return result


class Server:
    def __init__(self):
        self.sim = LocalSim()
        self.policies = PolicyBank()
        self.scene_id = DUCK_SCENE_ID
        self.scene_spec = SCENE_SPEC
        self.viewers = set()
        self.board = None                 # X5 control websocket
        self.vision = None
        self.board_state = None           # latest state packet from the X5
        self.board_last_at = 0.0
        self.cmd_manual = np.zeros(3, dtype=np.float32)
        self.cmd_auto = np.zeros(3, dtype=np.float32)
        self.cmd_auto_at = 0.0
        self.control_mode = "manual"
        self.dets = []
        self.dets_src = "-"
        self.dets_hsv = None
        self.dets_ms = 0.0
        self.fps_out = 0.0
        self.local_time = 0.0
        self.kick = LocalKick()
        self.local_policy = 'alpha_stand'
        self.local_command = np.zeros(3, dtype=np.float32)
        self.last_main_jpg = b""
        self.last_main_at = 0.0
        self.last_scene_jpg = b""
        self.last_scene_at = 0.0
        self.last_headcam_jpg = b""
        self.last_headcam_at = 0.0

        self.render_data = mujoco.MjData(self.sim.model)
        self.render_exec = ThreadPoolExecutor(max_workers=1)
        self._cam_renderer = {}
        self.cam = {"azimuth": 150.0, "elevation": -16.0, "distance": 0.62}
        self.duck = None
        self.vlm = None
        if DUCKAGENT_ENABLE:
            try:
                from duck_agent.simhost import SimHost
                self.duck = SimHost(self)
                print("[agent] Harness Duck host attached (mode='agent' to drive)", flush=True)
            except Exception as exc:
                print(f"[agent] init failed: {exc}", flush=True)
                self.duck = None
        if DUCK_VLM_ENABLE:
            try:
                from duck_vlm.host import DuckVlmHost
                self.vlm = DuckVlmHost(self)
                print("[vlm] Duck action-token harness attached (mode='vlm' to drive)", flush=True)
            except Exception as exc:
                print(f"[vlm] init failed: {exc}", flush=True)
                self.vlm = None


    # ---------------- replay / render ----------------
    def _sync_render_state(self):
        """Point the render model at the authoritative state (X5 stream or local sim)."""
        if self.board_state is not None:
            qp = self.board_state.get("qpos")
            qv = self.board_state.get("qvel")
            if qp is not None:
                self.render_data.qpos[:] = qp
                if qv is not None:
                    self.render_data.qvel[:] = qv
                mujoco.mj_forward(self.sim.model, self.render_data)
        else:
            mujoco.mj_copyData(self.render_data, self.sim.model, self.sim.data)

    def _jpeg(self, rgb, quality=86):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def render_main(self):
        if "main" not in self._cam_renderer:
            r = mujoco.Renderer(self.sim.model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.sim.trunk_id
            cam.lookat = [0, 0, 0.08]
            self._cam_renderer["main"] = (r, cam)
        r, cam = self._cam_renderer["main"]
        cam.distance = float(self.cam.get("distance", 0.62))
        cam.azimuth = float(self.cam.get("azimuth", 150))
        cam.elevation = float(self.cam.get("elevation", -16))
        r.update_scene(self.render_data, camera=cam)
        return self._jpeg(r.render())

    def render_overview(self):
        """Whole-floor overview (fixed scene camera) for the demo composite.

        The third-person tracking view only frames the duck, so the demo could
        not show where the duck was in the map. This uses the scene's
        `cam_overhead` camera so the full floor plan and the duck are both visible.
        """
        key = "overview"
        if key not in self._cam_renderer:
            r = mujoco.Renderer(self.sim.model, height=360, width=640)
            cam = mujoco.MjvCamera()
            cid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_overhead")
            if cid >= 0:
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cid
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = self.sim.trunk_id
                cam.lookat = [0.0, 0.0, 0.08]
                cam.distance = 4.5
                cam.azimuth = 130.0
                cam.elevation = -70.0
            self._cam_renderer[key] = (r, cam)
        r, cam = self._cam_renderer[key]
        r.update_scene(self.render_data, camera=cam)
        return self._jpeg(r.render(), quality=80)

    def render_headcam(self):
        if "head" not in self._cam_renderer:
            r = mujoco.Renderer(self.sim.model, height=240, width=320)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam_renderer["head"] = (r, cam)
        r, cam = self._cam_renderer["head"]
        pos = self.render_data.xpos[self.sim.head_id].copy()
        fwd = self.render_data.xmat[self.sim.head_id].reshape(3, 3)[:, 0].copy()
        fwd /= np.linalg.norm(fwd)
        campos = pos + fwd * 0.09 + np.array([0, 0, 0.03])
        tilt = math.radians(20)
        v = fwd * math.cos(tilt) + np.array([0.0, 0.0, -1.0]) * math.sin(tilt)
        v /= np.linalg.norm(v)
        L = 0.5
        cam.lookat = campos + v * L; cam.distance = L
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.asin(float(np.clip(v[2], -1, 1))))
        r.update_scene(self.render_data, camera=cam)
        return self._jpeg(r.render(), quality=76)

    # ---------------- command relay ----------------
    def effective_cmd(self):
        if self.control_mode == "follow":
            if time.monotonic() - self.cmd_auto_at < AUTOCMD_STALE_S:
                return self.cmd_auto
            return np.zeros(3, dtype=np.float32)
        return self.cmd_manual

    async def _send_to_board(self, obj):
        if self.board is not None:
            try:
                await self.board.send_str(json.dumps(obj))
                return True
            except Exception:
                pass
        return False

    def local_step(self):
        d = self.sim.data
        if self.duck is not None and self.board is None and self.control_mode == "agent":
            self.cmd_manual[:] = self.duck.tick(self.local_time)
        if self.vlm is not None and self.board is None and self.control_mode == "vlm":
            self.cmd_manual[:] = self.vlm.tick(self.local_time)
        gravity = quat_rotate_inverse(d.xquat[self.sim.trunk_id].copy(), np.array([0, 0, -1.0]))
        gyro = d.sensordata[self.sim.imu_gyro_adr:self.sim.imu_gyro_adr+3].astype(np.float32)
        policy, cmd = self.kick.select(CONTROL_DT, self.effective_cmd(),
                                      float(np.linalg.norm(d.qvel[:2])),
                                      float(np.linalg.norm(gyro)), -float(gravity[2]))
        obs = np.concatenate([
            gyro, gravity.astype(np.float32),
            (d.qpos[self.sim.joint_qpos_idx] - self.sim.default_pose).astype(np.float32),
            d.qvel[self.sim.joint_qvel_idx].astype(np.float32), self.sim.last_action,
            np.concatenate([cmd, np.zeros(10, dtype=np.float32)]),
        ]).astype(np.float32)
        action = self.policies.infer(policy, obs)
        self.sim.last_action = action.copy()
        d.ctrl[:] = self.sim.default_pose + action
        if self.vlm is not None and self.control_mode == 'vlm':
            head_delta = self.vlm.head_override()
            if head_delta:
                for _jname, _delta in head_delta.items():
                    _idx = self.sim.head_ctrl_idx.get(_jname)
                    if _idx is not None:
                        d.ctrl[_idx] = float(self.sim.default_pose[_idx]) + float(_delta)
        self.local_policy, self.local_command = policy, cmd.copy()
        for _ in range(DECIMATION):
            mujoco.mj_step(self.sim.model, d)
        self.local_time += CONTROL_DT

    async def action_events(self):
        while self.kick.events:
            event = self.kick.events.pop(0)
            # A failed cycle must not resume an unsafe held velocity automatically.
            if event['status'] == 'failed':
                self.cmd_manual[:] = 0
                self.cmd_auto[:] = 0
            for viewer in list(self.viewers):
                try:
                    await viewer.send_json(event)
                except Exception:
                    pass

    # ---------------- loops ----------------
    async def command_loop(self):
        """Forward the current command to the X5 at a fixed ~25 Hz heartbeat so
        a dropped packet is always re-sent within 40 ms — commands never get
        silently lost (which is what made fwd/back feel unresponsive)."""
        while True:
            cmd = self.effective_cmd()
            if self.board is not None:
                await self._send_to_board({"type": "cmd", "vx": float(cmd[0]), "vy": float(cmd[1]), "vyaw": float(cmd[2])})
            await asyncio.sleep(0.04)

    async def local_sim_loop(self):
        """Only drives the local fallback sim when no X5 board is attached."""
        while True:
            if self.board is None:
                self.local_step()
                await self.action_events()
                await asyncio.sleep(CONTROL_DT * 0.9)
            else:
                await asyncio.sleep(0.1)

    async def frame_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            if self.viewers:
                self._sync_render_state()
                jpg = await loop.run_in_executor(self.render_exec, self.render_main)
                self.last_main_jpg = jpg
                dead = []
                for ws in list(self.viewers):
                    try:
                        await ws.send_bytes(FRAME_MAIN + jpg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self.viewers.discard(ws)
            await asyncio.sleep(1 / 20)

    async def headcam_loop(self):
        loop = asyncio.get_running_loop()
        frame_i = 0
        while True:
            if self.viewers or self.vision is not None or (self.vlm is not None and self.control_mode == "vlm"):
                self._sync_render_state()
                jpg = await loop.run_in_executor(self.render_exec, self.render_headcam)
                self.last_headcam_jpg = jpg
                self.last_headcam_at = time.monotonic()
                # Demo video: stitch the duck-cam with a whole-floor overview faster
                # than the once-per-decision frame so the clip is smooth and longer.
                if self.vlm is not None and getattr(self.vlm, "recording", lambda: False)():
                    frame_i += 1
                    # Third-person tracking shot every frame so the duck stays large
                    # and smooth; the whole-floor overview refreshes every other frame.
                    self.last_main_jpg = await loop.run_in_executor(self.render_exec, self.render_main)
                    self.last_main_at = time.monotonic()
                    if frame_i % 2 == 1:
                        self.last_scene_jpg = await loop.run_in_executor(self.render_exec, self.render_overview)
                        self.last_scene_at = time.monotonic()
                    if self.last_scene_jpg and self.last_main_jpg:
                        self.vlm.record_demo_frame()
                if self.viewers:
                    for ws in list(self.viewers):
                        try:
                            await ws.send_bytes(FRAME_HEADCAM + jpg)
                        except Exception:
                            pass
                if self.vision is not None:
                    try:
                        await self.vision.send_bytes(jpg)
                    except Exception:
                        pass
            await asyncio.sleep(1 / 8)

    async def telemetry_loop(self):
        while True:
            if self.viewers:
                if self.board_state is not None:
                    bs = self.board_state
                    st = {
                        "type": "state", "mode": "board",
                        "control_mode": self.control_mode,
                        "vision_connected": self.vision is not None,
                        "cmd": [round(float(x), 3) for x in self.effective_cmd()],
                        "achieved": bs.get("achieved", [0, 0, 0]),
                        "policy": bs.get("policy", "-"),
                        "behavior": bs.get("behavior"),
                        "board_connected": True,
                        "board_rtt_ms": 0,
                        "ctrl_hz": 50.0,
                        "stalls": 0,
                        "sim_time": bs.get("t", 0),
                        "fallen": bs.get("fallen", False),
                        "ball": bs.get("ball"),
                        "duck": bs.get("duck"),
                        "rpy": bs.get("rpy"),
                        "jact": 0,
                        "dets": self.dets, "dets_src": self.dets_src,
                        "dets_ms": self.dets_ms, "hsv": self.dets_hsv,
                    }
                else:
                    d = self.sim.data
                    trunk = d.xpos[self.sim.trunk_id]
                    g = quat_rotate_inverse(d.xquat[self.sim.trunk_id].copy(), np.array([0,0,-1.0]))
                    vw = np.array(d.qvel[0:3])
                    vb = quat_rotate_inverse(d.qpos[3:7], vw)
                    st = {
                        "type": "state", "mode": "local",
                        "control_mode": self.control_mode,
                        "vision_connected": self.vision is not None,
                        "cmd": [round(float(x), 3) for x in self.effective_cmd()],
                        "achieved": [round(float(vb[0]),3), round(float(vb[1]),3), round(float(d.qvel[5]),3)],
                        "policy": self.local_policy,
                        "behavior": self.kick.name,
                        "behavior_phase": self.kick.phase,
                        "applied_cmd": [round(float(x), 3) for x in self.local_command],
                        "available_actions": [name for name, policy in {**KICK_POLICIES, **TRICK_POLICIES}.items()
                                              if policy in self.policies.sessions],
                        "board_connected": False,
                        "board_rtt_ms": 0,
                        "ctrl_hz": 50.0,
                        "stalls": 0,
                        "sim_time": round(self.local_time, 2),
                        "fallen": bool(g[2] > -0.5),
                        "ball": [round(float(x),3) for x in d.xpos[self.sim.ball_body]],
                        "duck": [round(float(x),3) for x in trunk],
                        "rpy": [0,0,0], "jact": 0,
                        "dets": self.dets, "dets_src": self.dets_src,
                        "dets_ms": self.dets_ms, "hsv": self.dets_hsv,
                    }
                if self.duck is not None:
                    try:
                        st["agent"] = self.duck.status()
                    except Exception:
                        pass
                if self.vlm is not None:
                    try:
                        st["vlm"] = self.vlm.status()
                    except Exception:
                        pass
                msg = json.dumps(st)
                for ws in list(self.viewers):
                    try:
                        await ws.send_str(msg)
                    except Exception:
                        pass
            await asyncio.sleep(0.05)

    # ---------------- websocket handlers ----------------
    async def ws_viewer(self, request):
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        self.viewers.add(ws)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                t = data.get("type")
                if t == "event" and self.duck is not None:
                    from duck_agent.events import make_event as _mk
                    ev = _mk(source=data.get("source", "button"),
                             kind=data.get("kind", "summon"),
                             text=data.get("text", ""),
                             origin=data.get("origin"),
                             force_wake=bool(data.get("force_wake", False)))
                    self.duck.push_event(ev)
                    await ws.send_json({"type": "agent_ack", "kind": ev.kind,
                                        "intent": ev.intent, "text": ev.text})
                elif t == "cam":
                    for k in ("azimuth", "elevation", "distance"):
                        if k in data:
                            try:
                                self.cam[k] = float(data[k])
                            except Exception:
                                pass
                elif t == "wake_cfg" and self.duck is not None:
                    from duck_agent import events as _ev
                    _ev.WAKE.update({k: v for k, v in data.items()
                                     if k in ("wake_names", "wake_on_any")})
                    await ws.send_json({"type": "agent_ack", "wake": dict(_ev.WAKE)})
                elif t == "vlm_config" and self.vlm is not None:
                    await ws.send_json({"type": "vlm_state", "state": self.vlm.configure(data.get("config") or {})})
                elif t == "vlm_control" and self.vlm is not None:
                    state = self.vlm.control(data.get("action", "status"), data)
                    if data.get("action") == "start":
                        self.control_mode = "vlm"
                        self.cmd_manual[:] = 0
                        self.cmd_auto[:] = 0
                    await ws.send_json({"type": "vlm_state", "state": state})
                elif t == "vlm_human_action" and self.vlm is not None:
                    await ws.send_json({"type": "vlm_state", "state": self.vlm.human_action(data.get("token", ""))})
                if t == "cmd":
                    v = np.array([float(data.get("vx", 0)), float(data.get("vy", 0)),
                                  float(data.get("vyaw", 0))], dtype=np.float32)
                    self.cmd_manual = np.clip(v, [-0.45, -0.3, -1.5], [0.35, 0.3, 1.5])
                elif t == "mode" and data.get("mode") in ("manual", "follow", "agent", "vlm"):
                    self.control_mode = data["mode"]
                    if self.control_mode in ("agent", "vlm"):
                        self.cmd_manual[:] = 0
                        self.cmd_auto[:] = 0
                    if self.control_mode == "vlm" and self.vlm is not None:
                        await ws.send_json({"type": "vlm_state", "state": self.vlm.status()})
                    if self.control_mode == "agent" and self.duck is not None:
                        self.cmd_manual[:] = 0
                        self.cmd_auto[:] = 0
                        await ws.send_json({"type": "agent_ack", "mode": "agent",
                                            "note": "Agent 模式：召唤/文本指令驱动鸭子"})
                elif t == "action":
                    name = data.get('name', '')
                    if self.board is not None:
                        sent = await self._send_to_board({'type': 'action', 'name': name})
                        await ws.send_json({'type': 'action_status', 'name': name,
                                            'status': 'forwarded' if sent else 'failed',
                                            'executor': 'x5', 'message': 'Sent to X5; execution not yet confirmed.' if sent else 'X5 send failed.'})
                    else:
                        gravity = quat_rotate_inverse(self.sim.data.xquat[self.sim.trunk_id].copy(), np.array([0, 0, -1.0]))
                        await ws.send_json(self.kick.request(name, self.policies.sessions, gravity[2] > -.5))
                elif t == "reset":
                    await self._send_to_board({"type": "reset"})
                    self.kick.finish('cancelled', 'Reset cancelled the kick.')
                    self.cmd_manual[:] = 0
                    self.cmd_auto[:] = 0
                    self.sim.reset()
                    await self.action_events()
        finally:
            self.viewers.discard(ws)
        return ws

    async def ws_board(self, request):
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024, heartbeat=15)
        await ws.prepare(request)
        if self.board is not None:
            try:
                await self.board.close()
            except Exception:
                pass
        self.board = ws
        self.kick.finish('cancelled', 'X5 connected; control transferred to board.')
        await self.action_events()
        self.board_state = None
        print(f"[board] X5 brain attached from {request.remote}", flush=True)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    if data.get("type") == "state":
                        self.board_state = data
                        self.board_last_at = time.monotonic()
        finally:
            if self.board is ws:
                self.board = None
                self.board_state = None
            print("[board] X5 brain detached", flush=True)
        return ws

    async def ws_vision(self, request):
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=15)
        await ws.prepare(request)
        if self.vision is not None:
            try:
                await self.vision.close()
            except Exception:
                pass
        self.vision = ws
        print(f"[vision] X5 vision attached from {request.remote}", flush=True)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                t = data.get("type")
                if t == "dets":
                    self.dets = data.get("dets", [])
                    self.dets_src = data.get("track_src", "-")
                    self.dets_hsv = data.get("hsv")
                    self.dets_ms = float(data.get("infer_ms", 0))
                elif t == "auto_cmd":
                    self.cmd_auto = np.clip(np.array([
                        float(data.get("vx", 0)), 0.0, float(data.get("vyaw", 0))],
                        dtype=np.float32), [-0.3, -0.3, -1.5], [0.3, 0.3, 1.5])
                    self.cmd_auto_at = time.monotonic()
        finally:
            if self.vision is ws:
                self.vision = None
            print("[vision] detached", flush=True)
        return ws

    async def api_scenes(self, request):
        from duck_vlm.scenes import catalog
        try:
            cat = catalog()
            scenes = []
            for sid in cat.list_ids():
                spec = cat.load(sid)
                scenes.append({"id": sid, "tasks": [t.get("id") for t in spec.list_tasks()]})
            return web.json_response({"current": self.scene_id or None, "scenes": scenes})
        except Exception as exc:
            return web.json_response({"current": self.scene_id or None, "scenes": [], "error": str(exc)})

    async def api_scene_select(self, request):
        from duck_vlm.scenes import catalog
        try:
            data = await request.json()
        except Exception:
            data = {}
        scene_id = str(data.get("scene_id") or "").strip()
        try:
            spec = catalog().load(scene_id)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        (HERE / ".duck_scene_id").write_text(scene_id + "\n", encoding="utf-8")
        if data.get("restart", True):
            def _restart():
                subprocess.Popen(
                    ["/usr/bin/python3", str(HERE / "service.py"), "restart"],
                    cwd=str(HERE), env=os.environ.copy(), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True, close_fds=True,
                )
            asyncio.get_running_loop().call_later(0.8, _restart)
        return web.json_response({"ok": True, "scene_id": spec.scene_id,
                                  "restarting": bool(data.get("restart", True))})

    async def http_vlm(self, request):
        return web.FileResponse(HERE / "web/vlm.html")

    async def api_vlm_state(self, request):
        if self.vlm is None:
            return web.json_response({"error": "DuckVLM disabled"}, status=503)
        return web.json_response(self.vlm.status())

    async def api_vlm_control(self, request):
        if self.vlm is None:
            return web.json_response({"error": "DuckVLM disabled"}, status=503)
        try:
            data = await request.json()
        except Exception:
            data = {}
        action = data.get("action", "status")
        state = self.vlm.control(action, data)
        if action == "start":
            self.control_mode = "vlm"
            self.cmd_manual[:] = 0
            self.cmd_auto[:] = 0
        return web.json_response(state)

    async def api_vlm_human(self, request):
        if self.vlm is None:
            return web.json_response({"error": "DuckVLM disabled"}, status=503)
        try:
            data = await request.json()
        except Exception:
            data = {}
        return web.json_response(self.vlm.human_action(data.get("token", "")))

    async def api_vlm_snapshot(self, request):
        view = (request.query.get("view") or "head").lower()
        jpg = bytes(self.last_main_jpg or b"") if view == "main" else bytes(self.last_headcam_jpg or b"")
        if not jpg:
            self._sync_render_state()
            fn = self.render_main if view == "main" else self.render_headcam
            jpg = await asyncio.get_running_loop().run_in_executor(self.render_exec, fn)
        return web.Response(body=jpg, content_type="image/jpeg")

    async def http_index(self, request):
        return web.FileResponse(HERE / "web/index.html")

    async def http_duck(self, request):
        return web.FileResponse(HERE / "web/duck_panel.html")

    async def http_health(self, request):
        return web.json_response({"ok": True,
                                  "board": self.board is not None,
                                  "vision": self.vision is not None})


    async def agent_out_loop(self):
        """Broadcast agent decisions + mirrored kick statuses to viewers."""
        last_n = 0
        while True:
            if self.duck is not None:
                dec = self.duck.agent.decisions
                for i in range(last_n, len(dec)):
                    for v in list(self.viewers):
                        try:
                            await v.send_json({"type": "agent_decision",
                                               "t": dec[i]["t"], "text": dec[i]["text"]})
                        except Exception:
                            pass
                last_n = len(dec)
                for m in self.duck.drain_outbox():
                    for v in list(self.viewers):
                        try:
                            await v.send_json(m)
                        except Exception:
                            pass
            await asyncio.sleep(0.2)

async def main():
    server = Server()
    app = web.Application()
    app.router.add_get("/", server.http_index)
    app.router.add_get("/duck", server.http_duck)
    app.router.add_get("/healthz", server.http_health)
    app.router.add_get("/api/scenes", server.api_scenes)
    app.router.add_post("/api/scenes/select", server.api_scene_select)
    app.router.add_get("/vlm", server.http_vlm)
    app.router.add_get("/api/vlm/state", server.api_vlm_state)
    app.router.add_post("/api/vlm/control", server.api_vlm_control)
    app.router.add_post("/api/vlm/human", server.api_vlm_human)
    app.router.add_get("/api/vlm/snapshot", server.api_vlm_snapshot)
    app.router.add_get("/ws", server.ws_viewer)
    app.router.add_get("/board", server.ws_board)
    app.router.add_get("/vision", server.ws_vision)
    app.router.add_static("/fonts", HERE / "web/fonts", show_index=False)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.environ.get("SIM_PORT", "8080")))
    await site.start()
    print("display server on :8080 (X5-authoritative replay + local fallback)", flush=True)

    await asyncio.gather(
        server.command_loop(),
        server.local_sim_loop(),
        server.frame_loop(),
        server.headcam_loop(),
        server.telemetry_loop(),
        server.agent_out_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
