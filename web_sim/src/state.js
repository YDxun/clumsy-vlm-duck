/**
 * 本体感知 + 目标可达性 —— `duck_vlm/state.py` + `duck_vlm/scenes.py` 的浏览器版。
 *
 * 最重要的一条设计（用户明确要求过）：**鸭子没看到目标时，不许用上帝视角真值。**
 * 只有目标真的落在头摄画面里、并且视线上没有遮挡，才把距离/方位写进提示词；
 * 否则那些字段是 `not_found`，模型只能靠转头扫视去找。
 * 鸭子身上没有深度传感器，这条规则就是“仿真真值不许偷偷泄漏给 VLM”的闸门。
 */

function quatRotateInverse(quat, vec) {
  const w = quat[0], x = quat[1], y = quat[2], z = quat[3];
  const t = [2 * (y * vec[2] - z * vec[1]), 2 * (z * vec[0] - x * vec[2]), 2 * (x * vec[1] - y * vec[0])];
  return [vec[0] - w * t[0] + (y * t[2] - z * t[1]),
          vec[1] - w * t[1] + (z * t[0] - x * t[2]),
          vec[2] - w * t[2] + (x * t[1] - y * t[0])];
}

export class DuckStateSensor {
  /**
   * @param {object} opts
   * @param {object} opts.duck DuckSim
   * @param {object|null} opts.scene SceneIndex
   * @param {object|null} opts.view DuckView —— 给了就用**渲染器**判断目标有没有被挡住
   *        （这个 WASM 构建的 mj_ray 拿不到 geomid，见 view.maskVisible 的注释）
   */
  constructor({ duck, scene = null, view = null }) {
    this.duck = duck;
    this.scene = scene;
    this.view = view;
    this._cache = new Map();
    this.mujoco = duck.mujoco;
    this.model = duck.model;
    this.data = duck.data;
    this._BODY = duck.mujoco.mjtObj.mjOBJ_BODY.value;
  }

  /** 目标名 → body id。先问场景元数据（能理解自然语言标签），再直接当 body 名查。 */
  resolveTarget(targetName) {
    const raw = String(targetName || "ball").trim();
    const key = raw.toLowerCase();
    if (this._cache.has(key)) return this._cache.get(key);
    let body = this.scene ? this.scene.resolveBody(raw) : null;
    if (!body && key === "ball") body = "ball";
    let id = -1;
    if (body) id = this.mujoco.mj_name2id(this.model, this._BODY, body);
    if (id < 0) id = this.mujoco.mj_name2id(this.model, this._BODY, key);
    const out = id < 0 ? null : id;
    this._cache.set(key, out);
    return out;
  }

  /**
   * 采一帧本体状态。sim_time 由调用方给（单位秒）。
   * @returns {object} DuckState（字段名与 Python types.DuckState 一致）
   */
  snapshot(targetName = "ball", simTime = 0) {
    const d = this.data, t = this.duck.bodies.trunk;
    const trunk = [d.xpos[t * 3], d.xpos[t * 3 + 1], d.xpos[t * 3 + 2]];
    const quat = d.xquat.subarray(t * 4, t * 4 + 4);
    const m = d.xmat;
    const heading = Math.atan2(m[t * 9 + 3], m[t * 9 + 0]);
    const gravity = quatRotateInverse(quat, [0, 0, -1]);
    const upright = -gravity[2];
    const gyro = this.duck.gyroAdr;
    const state = {
      simTime: Number(simTime), x: trunk[0], y: trunk[1], z: trunk[2],
      headingRad: heading,
      linearSpeedMps: Math.hypot(d.qvel[0], d.qvel[1]),
      angularSpeedRps: Math.hypot(d.sensordata[gyro], d.sensordata[gyro + 1], d.sensordata[gyro + 2]),
      upright, fallen: upright < 0.5,
      targetName: targetName || "ball",
      targetVisible: false, targetRangeM: null, targetBearingRad: null,
      targetElevationRad: null, targetUv: null, targetWorldXyz: null,
    };

    const id = this.resolveTarget(targetName);
    if (id === null) return state;
    const world = [d.xpos[id * 3], d.xpos[id * 3 + 1], d.xpos[id * 3 + 2]];
    const local = quatRotateInverse(quat, [world[0] - trunk[0], world[1] - trunk[1], world[2] - trunk[2]]);
    const inFront = local[0] > 0.03;
    let uv = inFront ? this.duck.projectToHeadCam(world) : null;
    if (uv) {
      if (this.view) {
        // 画面里一个像素都没有 = 真的看不见（出画或被挡住）
        const mask = this.view.maskVisible(this.duck.geomIndicesUnderBody(id));
        if (mask.count < 2) uv = null;
      } else if (this.duck.isOccluded(world, id)) {
        uv = null;
      }
    }
    state.targetUv = uv ? [uv.u, uv.v] : null;
    state.targetVisible = uv !== null;
    if (state.targetVisible) {
      // 只有在头摄真的拍到目标时才允许把仿真真值写进提示词。
      state.targetWorldXyz = world;
      state.targetRangeM = Math.hypot(local[0], local[1], local[2]);
      state.targetBearingRad = Math.atan2(local[1], local[0]);
      state.targetElevationRad = Math.atan2(local[2], Math.hypot(local[0], local[1]));
    }
    return state;
  }

  /** 提示词里的 ROBOT STATE 一行（格式与 Python DuckState.prompt_text 完全一致）。 */
  static promptText(s) {
    const parts = [
      `body=(${s.x.toFixed(2)},${s.y.toFixed(2)},${s.z.toFixed(2)})`,
      `heading=${s.headingRad.toFixed(2)}rad`,
      `speed=${s.linearSpeedMps.toFixed(2)}m/s`,
      `turn_rate=${s.angularSpeedRps.toFixed(2)}rad/s`,
      `upright=${s.upright.toFixed(2)}`,
      "fallen=" + (s.fallen ? "yes" : "no"),
    ];
    if (s.targetRangeM === null) {
      parts.push(`target(${s.targetName})=not_found`);
    } else {
      parts.push(`target(${s.targetName})=range ${s.targetRangeM.toFixed(2)}m, ` +
                 `bearing ${s.targetBearingRad.toFixed(2)}rad, ` +
                 `visible=${s.targetVisible ? "yes" : "no"}`);
    }
    return parts.join(" | ");
  }
}
