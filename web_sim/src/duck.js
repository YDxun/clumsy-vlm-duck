/**
 * 浏览器里的鸭子：MuJoCo WASM 物理 + onnxruntime-web 策略。
 *
 * 与 `_dev_cockpit/sim_server.py` 的 LocalSim + local_step 一一对应：
 *   61 维观测 = 陀螺(3) + 重力方向(3) + 关节位置偏差(14) + 关节速度(14)
 *                + 上一步动作(14) + 指令(3) + 零填充(10)
 *   50 Hz 控制，每个控制步 4 次物理子步；ctrl = DEFAULT_POSE + action。
 *
 * 两条从 Python 端搬过来的经验（都在 README 里记了原因）：
 *   - 执行器力矩上限必须**烘进 XML**，因为 JS 侧改不了模型（见 flatten_scene.py）；
 *   - mjtObj 枚举必须**从模块读**，硬编码会把陀螺仪观测弄成恒 0，鸭子一走就摔。
 */
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web";

export const HOME = Float64Array.from([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491,
  0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530]);
export const PHYS_DT = 0.005, DECIMATION = 4, CONTROL_DT = PHYS_DT * DECIMATION;
export const NJ = 14, OBS_DIM = 61;

/**
 * onnxruntime-web 的运行环境设置。浏览器里必须调用一次：
 *   - `wasmPaths` 指向 dist/ 目录（`ort.min.mjs` 是“外部 wasm”版本，运行时才去取 .wasm）；
 *   - `numThreads = 1`：多线程需要 SharedArrayBuffer（要求 COOP/COEP 响应头），
 *     单线程让页面可以随便挂静态托管，也顺带保证推理结果可复现。
 * Node 侧的验证脚本不需要调用（默认值即可）。
 */
export function configureOrt({ wasmPaths, numThreads = 1, proxy = false } = {}) {
  if (!ort.env || !ort.env.wasm) throw new Error("onnxruntime-web 版本不含 ort.env.wasm");
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  ort.env.wasm.numThreads = numThreads;
  ort.env.wasm.proxy = proxy;
}

export function mjObj(mujoco, name) {
  const e = mujoco.mjtObj && mujoco.mjtObj[name];
  if (e && typeof e === "object" && typeof e.value === "number") return e.value;
  if (typeof e === "number") return e;
  throw new Error(`无法解析 mjtObj.${name}`);
}

function quatRotateInverse(quat, vec) {
  const w = quat[0], x = quat[1], y = quat[2], z = quat[3];
  const t = [2 * (y * vec[2] - z * vec[1]), 2 * (z * vec[0] - x * vec[2]), 2 * (x * vec[1] - y * vec[0])];
  return [vec[0] - w * t[0] + (y * t[2] - z * t[1]),
          vec[1] - w * t[1] + (z * t[0] - x * t[2]),
          vec[2] - w * t[2] + (x * t[1] - y * t[0])];
}

/** 四元数 (w,x,y,z) -> 3x3 旋转矩阵，行主序展平成 9 个元素（与 MuJoCo 的 xmat 同存储方式）。 */
export function quatToMat(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

export class DuckSim {
  constructor({ mujoco, model, data }) {
    Object.assign(this, { mujoco, model, data });
    const OBJ_BODY = mjObj(mujoco, "mjOBJ_BODY"), OBJ_SENSOR = mjObj(mujoco, "mjOBJ_SENSOR");
    this.bodies = {
      trunk: mujoco.mj_name2id(model, OBJ_BODY, "trunk_base"),
      head: mujoco.mj_name2id(model, OBJ_BODY, "yaw_roll_motion"),
      ball: mujoco.mj_name2id(model, OBJ_BODY, "ball"),
    };
    this.ballGeom = this.geomIndicesUnderBody(this.bodies.ball);
    this.gyroAdr = model.sensor_adr[mujoco.mj_name2id(model, OBJ_SENSOR, "imu_ang_vel")];
    this.jointQpos = []; this.jointQvel = [];
    for (let i = 0; i < model.nu; i++) {
      const jid = model.actuator_trnid[i * 2];
      this.jointQpos.push(model.jnt_qposadr[jid]);
      this.jointQvel.push(model.jnt_dofadr[jid]);
    }
    this.lastAction = new Float64Array(NJ);
    this.policy = null;
    this.steps = 0;
    // 场景的 vis.global_.fovy。WASM 绑定不保证暴露嵌套结构体，读不到就用 MuJoCo 默认 45。
    this.fovy = 45;
    try {
      const v = model.vis?.global_?.fovy;
      if (typeof v === "number" && v > 0) this.fovy = v;
    } catch { /* 绑定没暴露 vis：保持默认值 */ }
    this.reset();
  }

  /** body 的祖先链（含自身），用于按“躯干子树”筛选 geom —— 渲染验证与视觉 affordance 都要用。 */
  bodyChain(bodyId) {
    const chain = [];
    for (let b = bodyId; b > 0; b = this.model.body_parentid[b]) chain.push(b);
    return chain;
  }

  /**
   * 属于某个 body **及其所有后代**的 geom 下标。
   * 注意方向：鸭子的腿和头是 `trunk_base` 的后代（trunk_base -> yaw2roll -> hip_l -> …），
   * 所以必须沿 body_parentid 往上找祖先，不能只比较 bodyid。
   * （写反过一次：只拿到 trunk_base 自身挂着 12 个外壳，漏掉腿/头的 69 个。）
   */
  geomIndicesUnderBody(bodyId) {
    const out = [];
    for (let g = 0; g < this.model.ngeom; g++) {
      for (let b = this.model.geom_bodyid[g]; b > 0; b = this.model.body_parentid[b]) {
        if (b === bodyId) { out.push(g); break; }
      }
    }
    return out;
  }

  /** 鸭子本体的全部 geom（trunk_base 子树，共 81 个）；worldbody 上的地面/墙不算。 */
  get duckGeoms() {
    return this.geomIndicesUnderBody(this.bodies.trunk);
  }

  /** 与 Python LocalSim.reset() 一致：keyframe + 零速 + 额定姿态。 */
  reset() {
    this.mujoco.mj_resetDataKeyframe(this.model, this.data, 0);
    this.data.qvel.fill(0);
    this.data.ctrl.set(HOME);
    this.mujoco.mj_forward(this.model, this.data);
    this.lastAction.fill(0);
    this.steps = 0;
  }

  async loadPolicy(url) {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    this.policy = await ort.InferenceSession.create(buf);
    this.policyIn = this.policy.inputNames[0];
    this.policyOut = this.policy.outputNames[0];
    return this.policy;
  }

  upright() {
    const t = this.bodies.trunk;
    const g = quatRotateInverse(this.data.xquat.subarray(t * 4, t * 4 + 4), [0, 0, -1]);
    return -g[2];
  }

  get pose() {
    const t = this.bodies.trunk, p = this.data.xpos;
    const m = this.data.xmat;
    const heading = Math.atan2(m[t * 9 + 3], m[t * 9 + 0]);   // 与 Python state.py 同一算法
    return { x: p[t * 3], y: p[t * 3 + 1], z: p[t * 3 + 2], heading };
  }

  get ballPose() {
    const b = this.bodies.ball, p = this.data.xpos;
    return { x: p[b * 3], y: p[b * 3 + 1], z: p[b * 3 + 2] };
  }

  /** 头体位姿（躯干坐标系下的真实姿态，不是相机）。 */
  headPose() {
    const h = this.bodies.head, p = this.data.xpos;
    const fwd = [this.data.xmat[h * 9], this.data.xmat[h * 9 + 1], this.data.xmat[h * 9 + 2]];
    const yaw = Math.atan2(fwd[1], fwd[0]);
    const pitch = Math.asin(Math.max(-1, Math.min(1, fwd[2])));
    return { x: p[h * 3], y: p[h * 3 + 1], z: p[h * 3 + 2], fwd, yaw, pitch };
  }

  /**
   * 鸭子第一人称相机 —— **逐字复刻** `_dev_cockpit/sim_server.py::render_headcam`：
   *   位置 = 头体原点 + 朝向*0.09 + 世界上方*0.03
   *   视线 = 朝向再向下俯 20°（固定俯角，**不跟着身体俯仰走**）
   *   垂直 FOV = 模型里的 vis.global_.fovy（本场景 45°）
   *
   * 这三个数字为什么重要：将来 VLM 看到的就是这张图，web 端和 Python 端必须一致，
   * 否则“同一句指令在两边表现不同”会变成没法排查的玄学。
   * （我自己先写错过一次：用了头体的 pitch 当相机俯仰、FOV 写死 62°，
   *   结果画面 2/3 是天空，地平线掉到画面下方 67% 处。）
   */
  headCamPose() {
    const h = this.bodies.head, p = this.data.xpos;
    const fwd = [this.data.xmat[h * 9], this.data.xmat[h * 9 + 1], this.data.xmat[h * 9 + 2]];
    const pos = [p[h * 3] + fwd[0] * 0.09, p[h * 3 + 1] + fwd[1] * 0.09, p[h * 3 + 2] + 0.03];
    const t = (20 * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
    let dir = [fwd[0] * c, fwd[1] * c, fwd[2] * c - s];
    const n = Math.hypot(dir[0], dir[1], dir[2]);
    dir = dir.map((v) => v / n);
    return { pos, dir, fovy: this.fovy ?? 45 };
  }

  /**
   * 场景里定义的固定相机（`cam_overhead` / `cam_workspace`），直接读模型，不硬编码。
   * 返回 MuJoCo 世界坐标下的 { pos, forward, up, right, fovy }。
   */
  fixedCamera(name) {
    const id = this.mujoco.mj_name2id(this.model, mjObj(this.mujoco, "mjOBJ_CAMERA"), name);
    if (id < 0) return null;
    const m = this.model;
    const R = quatToMat(m.cam_quat.subarray(id * 4, id * 4 + 4));
    return {
      id,
      pos: [m.cam_pos[id * 3], m.cam_pos[id * 3 + 1], m.cam_pos[id * 3 + 2]],
      right: [R[0], R[3], R[6]],      // 行主序矩阵的第 0 列
      up: [R[1], R[4], R[7]],         // 第 1 列
      forward: [-R[2], -R[5], -R[8]], // 相机看的是 -z 轴
      fovy: m.cam_fovy[id],
    };
  }

  /** 一个控制步：观测 -> 策略 -> 力矩 -> DECIMATION 次物理子步。 */
  async stepAsync(cmd = [0, 0, 0]) {
    const obs = this.buildObs(cmd);
    const out = await this.policy.run({ [this.policyIn]: new ort.Tensor("float32", obs, [1, OBS_DIM]) });
    const action = out[this.policyOut].data;
    this.lastAction.set(action);
    for (let i = 0; i < NJ; i++) this.data.ctrl[i] = HOME[i] + action[i];
    for (let i = 0; i < DECIMATION; i++) this.mujoco.mj_step(this.model, this.data);
    this.steps += 1;
    return action;
  }

  buildObs(cmd = [0, 0, 0]) {
    const { data } = this;
    const t = this.bodies.trunk;
    const gravity = quatRotateInverse(data.xquat.subarray(t * 4, t * 4 + 4), [0, 0, -1]);
    const obs = new Float32Array(OBS_DIM);
    for (let i = 0; i < 3; i++) obs[i] = data.sensordata[this.gyroAdr + i];
    obs.set(gravity, 3);
    for (let i = 0; i < NJ; i++) obs[6 + i] = data.qpos[this.jointQpos[i]] - HOME[i];
    for (let i = 0; i < NJ; i++) obs[20 + i] = data.qvel[this.jointQvel[i]];
    obs.set(this.lastAction, 34);
    obs[48] = cmd[0] || 0; obs[49] = cmd[1] || 0; obs[50] = cmd[2] || 0;
    return obs;
  }

  static async load({ sceneUrl, mjcfBase, wasmUrl }) {
    const mujoco = await loadMujoco(wasmUrl ? { locateFile: () => wasmUrl } : undefined);
    const xml = await (await fetch(sceneUrl)).text();
    const dir = mjcfBase.replace(/\/$/, "");
    const vfs = new mujoco.MjVFS();
    vfs.addBuffer("scene.xml", new TextEncoder().encode(xml));
    for (const name of [...xml.matchAll(/<mesh\s+file="([^"]+)"/g)].map((m) => m[1])) {
      const buf = new Uint8Array(await (await fetch(`${dir}/assets/${name}`)).arrayBuffer());
      vfs.addBuffer(`assets/${name}`, buf);
    }
    const model = mujoco.MjModel.from_xml_string(xml, vfs);
    return new DuckSim({ mujoco, model, data: new mujoco.MjData(model) });
  }
}
