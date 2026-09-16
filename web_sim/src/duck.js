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
/** 头部四个关节（见场景包 metadata.yaml 的 robot.head_joints）。 */
export const HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];

/**
 * 步态地板与达成率 —— **本仓库实测值**，和 quackd 公开的常数互相印证。
 *
 * 在 3 秒定速测试里量出来的（`tools/_probe_gait.mjs`）：
 *
 *   指令      实际线速度   实际角速度   达成率
 *   前进 0.10     0.002        —         2%     ← 基本原地抖动
 *   前进 0.30     0.125        —        42%
 *   前进 0.40     0.171        —        43%
 *   右转 0.80       —        -0.056      7%     ← 「原地转没用」的真相
 *   右转 1.10       —        -0.083      8%
 *   右转 1.15       —        -0.703     61%     ← 拐点：跨过去才真的转
 *   右转 1.50       —        -0.940     63%     ← 我们 TURN token 用的值
 *   右转 2.00       —        -0.940     47%     ← 物理转速饱和在 0.94 rad/s，再大是浪费
 *
 * 也就是说：走路策略有个**步态地板**，低于它的指令会被策略吃掉（腿在动，身体不走），
 * 高于地板的指令也只能拿到约 42% 的达成率。quackd 的 `GAIT_FLOOR` +
 * `ACHIEVED_FRACTION = 0.42` 测出来是同一回事，两边独立复现。
 *
 * 处理办法照 quackd：低于地板时把**整条 twist 一起**抬到地板（不是逐轴抬），
 * 这样「转 0.8」会变成「转 1.0」并且真的转起来，而不是原地抖。
 */
// 达成率 0.42 与 quackd 公布的 ACHIEVED_FRACTION 一致（就写在上面那张表里，不再单独导出一个常量）
export const GAIT_FLOOR = { vx: 0.22, vy: 0.30, wz: 1.15 };
export const CMD_MAX = { vx: 0.40, vy: 0.30, wz: 1.50 };

/** 把一条速度指令抬到步态地板之上；全零指令保持全零。 */
export function normalizeTwist(cmd) {
  const [vx, vy, wz] = [cmd[0] || 0, cmd[1] || 0, cmd[2] || 0];
  if (vx === 0 && vy === 0 && wz === 0) return [0, 0, 0];
  // 各轴按自己的地板算「离地板还有多远」，取最大的那个缩放量，整条 twist 一起放大
  let scale = 1;
  if (vx !== 0) scale = Math.max(scale, GAIT_FLOOR.vx / Math.abs(vx));
  if (vy !== 0) scale = Math.max(scale, GAIT_FLOOR.vy / Math.abs(vy));
  if (wz !== 0) scale = Math.max(scale, GAIT_FLOOR.wz / Math.abs(wz));
  return [
    Math.max(-CMD_MAX.vx, Math.min(CMD_MAX.vx, vx * scale)),
    Math.max(-CMD_MAX.vy, Math.min(CMD_MAX.vy, vy * scale)),
    Math.max(-CMD_MAX.wz, Math.min(CMD_MAX.wz, wz * scale)),
  ];
}

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

/**
 * 某个 body 的**前向**（局部 x 轴在世界里的方向）。
 *
 * MuJoCo 的 xmat 是行主序的旋转矩阵 R，世界坐标 = R @ 局部坐标，
 * 所以“局部 x 轴在世界里的方向”是 R 的**第 0 列** = 展平下标 0, 3, 6。
 * 写成 [0], [1], [2] 就变成第 0 行（= 转置），小角度下几乎看不出差别，
 * 一旦抬头/低头几十度，方向就直接反了 —— 这正是当初“低头”看起来像“抬头”的原因。
 */
export function bodyForward(xmat, bodyId) {
  const o = bodyId * 9;
  return [xmat[o], xmat[o + 3], xmat[o + 6]];
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
    const OBJ_JOINT = mjObj(mujoco, "mjOBJ_JOINT");
    this.headCtrl = {};
    for (let i = 0; i < model.nu; i++) {
      const jid = model.actuator_trnid[i * 2];
      this.jointQpos.push(model.jnt_qposadr[jid]);
      this.jointQvel.push(model.jnt_dofadr[jid]);
      const jname = mujoco.mj_id2name(model, OBJ_JOINT, jid);
      if (HEAD_JOINTS.includes(jname)) this.headCtrl[jname] = i;
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

  /**
   * 把某个带自由关节的物体（球、方块、杯子…）放到指定位置。
   * 自由关节名字是 `<body>_free`，qpos 地址直接来自 `jnt_qposadr`，不用猜偏移。
   */
  placeBody(bodyName, { x = 0, y = 0, z = 0.05, yaw = 0 } = {}) {
    const jid = this.mujoco.mj_name2id(this.model, mjObj(this.mujoco, "mjOBJ_JOINT"), `${bodyName}_free`);
    if (jid < 0) return false;
    const adr = this.model.jnt_qposadr[jid];
    const q = this.data.qpos;
    q[adr] = x; q[adr + 1] = y; q[adr + 2] = z;
    const h = yaw / 2;
    q[adr + 3] = Math.cos(h); q[adr + 4] = 0; q[adr + 5] = 0; q[adr + 6] = Math.sin(h);
    const dadr = this.model.jnt_dofadr[jid];
    for (let k = 0; k < 6; k++) this.data.qvel[dadr + k] = 0;
    this.mujoco.mj_forward(this.model, this.data);
    return true;
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

  /**
   * 装载策略。换场景时会重新建 DuckSim，但 ONNX 会话本身与场景无关，
   * 所以允许把已有会话传进来复用（否则每换一次场景都要重新解析一遍 ONNX）。
   */
  async loadPolicy(url, { session = null } = {}) {
    if (session) {
      this.policy = session;
    } else {
      this.policy = await DuckSim.createPolicySession(url);
    }
    this.policyIn = this.policy.inputNames[0];
    this.policyOut = this.policy.outputNames[0];
    this.policyName = (url || "").split("/").pop().replace(/\.onnx$/, "") || null;
    return this.policy;
  }

  /**
   * 热切换策略：技能 token（翻滚/跳舞/踢球…）就是这么实现的
   * —— 切到它自己的 ONNX，跑一段，再切回走路。所有策略共用同一套 61 维观测契约，
   * 所以任意时刻都能换人接管，不需要重置仿真。
   */
  setPolicy(session, name = null) {
    if (!session) return this.policyName;
    if (session === this.policy) { if (name) this.policyName = name; return this.policyName; }
    this.policy = session;
    this.policyIn = session.inputNames[0];
    this.policyOut = session.outputNames[0];
    this.lastAction.fill(0);            // 换策略时清掉上一步动作，别把别的策略的输出喂进去
    if (name) this.policyName = name;
    return this.policyName;
  }

  /** 单独建一个 ONNX 会话（和场景无关，可以跨场景复用）。 */
  static async createPolicySession(url) {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    return ort.InferenceSession.create(buf);
  }

  /**
   * 释放 WASM 侧的模型与数据。embind 的对象不会因为 JS 引用消失就马上归还内存，
   * 连续换场景时不主动释放，MuJoCo 堆会一直涨。
   */
  dispose() {
    try { this.model?.delete?.(); } catch { /* 已经被 FinalizationRegistry 回收 */ }
    try { this.data?.delete?.(); } catch { /* 同上 */ }
    this.model = null;
    this.data = null;
  }

  upright() {
    const t = this.bodies.trunk;
    const g = quatRotateInverse(this.data.xquat.subarray(t * 4, t * 4 + 4), [0, 0, -1]);
    return -g[2];
  }

  /** 角速度大小（rad/s）：陀螺仪三轴的模。站定判定和状态面板都要用。 */
  angularSpeed() {
    const g = this.gyroAdr;
    return Math.hypot(this.data.sensordata[g], this.data.sensordata[g + 1], this.data.sensordata[g + 2]);
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

  /** body 名 → 内部 id（找不到返回 -1）。判据里写的都是名字，得先翻译一下。 */
  bodyId(name) {
    if (this._bodyIdCache?.has(name)) return this._bodyIdCache.get(name);
    const id = this.mujoco.mj_name2id(this.model, mjObj(this.mujoco, "mjOBJ_BODY"), name);
    (this._bodyIdCache || (this._bodyIdCache = new Map())).set(name, id);
    return id;
  }

  /** 任意 body 的世界坐标（场景判据要用：目标物体、区域中心都按 body 名引用）。 */
  bodyPose(name) {
    const b = this.bodyId(name);
    if (b < 0) return null;
    const p = this.data.xpos;
    return { x: p[b * 3], y: p[b * 3 + 1], z: p[b * 3 + 2] };
  }

  /** body 的水平速度大小（m/s）：与 Python 判据 `speed_xy_le` 用同一份 cvel。 */
  bodySpeedXY(name) {
    const b = this.bodyId(name);
    if (b < 0) return null;
    const c = this.data.cvel;
    return Math.hypot(c[b * 6 + 3], c[b * 6 + 4]);
  }

  /** 头体位姿（躯干坐标系下的真实姿态，不是相机）。 */
  headPose() {
    const h = this.bodies.head, p = this.data.xpos;
    const fwd = bodyForward(this.data.xmat, h);
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
    const fwd = bodyForward(this.data.xmat, h);
    const pos = [p[h * 3] + fwd[0] * 0.09, p[h * 3 + 1] + fwd[1] * 0.09, p[h * 3 + 2] + 0.03];
    const t = (20 * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
    let dir = [fwd[0] * c, fwd[1] * c, fwd[2] * c - s];
    const n = Math.hypot(dir[0], dir[1], dir[2]);
    dir = dir.map((v) => v / n);
    return { pos, dir, fovy: this.fovy ?? 45 };
  }

  /**
   * 头摄的完整相机基。渲染和「目标可不可见」的判断共用这一份数学，
   * 免得出现“画面里有球、状态却说看不见”这种自相矛盾。
   *
   * right = forward × 世界上方（与 `duck_play/perception/camera.py` 的 DuckHeadCam 同约定：
   * MuJoCo 相机图像右方向是 forward×up，不是 up×forward）。
   *
   * 注意两个 FOV 故意不同：
   *   - 渲染用 `fovy`（= 模型的 vis.global_.fovy，45°），因为 VLM 一直看的就是这个构图；
   *   - 可见性判断用 `hfovDeg`（= DuckHeadCam 的 62° 水平 → 48.5° 垂直），
   *     因为 Python 端的 range/bearing 标签就是按这个模型发的。
   *   3.5° 的差是上游本来就有的，这里保持两边一致而不是各改各的。
   */
  headCamBasis() {
    const { pos, dir } = this.headCamPose();
    const z = [0, 0, 1];
    let right = [dir[1] * z[2] - dir[2] * z[1], dir[2] * z[0] - dir[0] * z[2], dir[0] * z[1] - dir[1] * z[0]];
    let n = Math.hypot(right[0], right[1], right[2]);
    if (n < 1e-6) right = [1, 0, 0];
    else right = right.map((v) => v / n);
    const up = [right[1] * dir[2] - right[2] * dir[1],
                right[2] * dir[0] - right[0] * dir[2],
                right[0] * dir[1] - right[1] * dir[0]];
    return { pos, forward: dir, right, up, fovy: this.fovy ?? 45, hfovDeg: 62 };
  }

  /** 世界点是否落在头摄视野内；落在里面则返回归一化像素坐标 (u,v) ∈ [0,1)。 */
  projectToHeadCam(xyz, { hfovDeg = 62, aspect = 4 / 3, znear = 0.05 } = {}) {
    const { pos, forward: f, right: r, up: u } = this.headCamBasis();
    const v = [xyz[0] - pos[0], xyz[1] - pos[1], xyz[2] - pos[2]];
    const zc = v[0] * f[0] + v[1] * f[1] + v[2] * f[2];
    if (zc <= znear) return null;
    const xc = v[0] * r[0] + v[1] * r[1] + v[2] * r[2];
    const yc = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    const hfov = (hfovDeg * Math.PI) / 180;
    const fx = 0.5 / Math.tan(hfov / 2);                 // 归一化：半宽 = 1
    const vfov = 2 * Math.atan(Math.tan(hfov / 2) / aspect);
    const fy = 0.5 / Math.tan(vfov / 2);
    const sx = 0.5 + (fx * xc) / zc;
    const sy = 0.5 - (fy * yc) / zc;
    if (sx < 0 || sx >= 1 || sy < 0 || sy >= 1) return null;
    return { u: sx, v: sy, depth: zc };
  }

  /**
   * 视线是否被场景几何挡住 —— 逐条对应 Python `DuckStateSensor._is_occluded`：
   *   命中距离 < 0 或者 ≥ 目标距离-0.02 → 没挡住（打到目标本身或更远）
   *   命中距离 < 0.05                     → 是鸭子自己头颈的壳，不算遮挡
   *   打到的 geom 属于目标 body            → 没挡住
   *
   * WASM 绑定的 mj_ray 要 9 个参数（第 9 个是版本差异多出来的，传 null 即可，
   * 第 8 个 geomid 是输出用的 Int32Array）。
   */
  isOccluded(targetXyz, targetBodyId) {
    const { pos } = this.headCamBasis();
    const dv = [targetXyz[0] - pos[0], targetXyz[1] - pos[1], targetXyz[2] - pos[2]];
    const dist = Math.hypot(dv[0], dv[1], dv[2]);
    if (dist < 1e-6) return false;
    const dir = dv.map((v) => v / dist);
    const gid = new Int32Array([-1]);
    let hit;
    try {
      hit = this.mujoco.mj_ray(this.model, this.data, new Float64Array(pos), new Float64Array(dir), null, 1, -1, gid, null);
    } catch {
      return false;   // 绑定不可用时按“没遮挡”处理，与 Python 的 try/except 一致
    }
    if (!(hit >= 0) || hit >= dist - 0.02) return false;
    if (hit < 0.05) return false;
    if (gid[0] < 0) return false;
    return this.model.geom_bodyid[gid[0]] !== targetBodyId;
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
  /**
   * 一个控制步。
   * @param {number[]} cmd 速度指令 [vx, vy, wz]
   * @param {object|null} headDelta 头部关节偏移（LOOK_DOWN 之类），
   *        在走路策略算出动作**之后**覆盖对应执行器 —— 与 Python sim_server 的做法一致：
   *        d.ctrl[head_idx] = DEFAULT_POSE[head_idx] + delta
   */
  async stepAsync(cmd = [0, 0, 0], headDelta = null) {
    // 低于步态地板的指令先抬到地板，否则鸭子只会原地抖腿（见 GAIT_FLOOR 的注释）
    const twist = normalizeTwist(cmd);
    this.lastTwist = twist;
    const obs = this.buildObs(twist);
    const out = await this.policy.run({ [this.policyIn]: new ort.Tensor("float32", obs, [1, OBS_DIM]) });
    const action = out[this.policyOut].data;
    this.lastAction.set(action);
    for (let i = 0; i < NJ; i++) this.data.ctrl[i] = HOME[i] + action[i];
    if (headDelta) {
      for (const [name, delta] of Object.entries(headDelta)) {
        const idx = this.headCtrl[name];
        if (idx !== undefined) this.data.ctrl[idx] = HOME[idx] + delta;
      }
    }
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

  /**
   * @param {object} opts
   * @param {string} opts.sceneUrl 场景 XML 的地址
   * @param {string} opts.mjcfBase 场景目录（meshdir 相对它解析）
   * @param {string} [opts.meshBase] **网格单独放别处时的基址**。默认同 mjcfBase。
   *   存在的理由：机器人网格是 CC BY-SA-NC 的硬件设计文件，公开站点**不分发**它们，
   *   而是运行时从上游固定 commit 取（见 THIRD_PARTY_NOTICES.md）。这样 NC/SA 的约束
   *   落在访客自己的使用上，与我们分发的东西无关。
   */
  static async load({ sceneUrl, mjcfBase, meshBase = null, wasmUrl }) {
    const mujoco = await loadMujoco(wasmUrl ? { locateFile: () => wasmUrl } : undefined);
    const xml = await (await fetch(sceneUrl)).text();
    const dir = mjcfBase.replace(/\/$/, "");
    // mesh 目录从 XML 里读，不写死：多个场景共用一个 _shared 目录（否则每个场景都要
    // 复制 20 MB 机器人网格）。MuJoCo 的 VFS 把路径当字符串，`../_shared/x.stl`
    // 只要注册时用同一个字符串就行。浏览器 URL 里带 `..` 会自己规范化，不用管。
    const meshdir = (xml.match(/meshdir="([^"]+)"/) || [, "assets"])[1];
    // 显式给 meshBase 时，它就是"网格所在的完整目录"（例如上游 assets 的 raw URL）；
    // 否则按 XML 的 meshdir 相对场景目录解析（本地/自包含模式）。
    const meshDir = meshBase ? meshBase.replace(/\/$/, "") : `${dir}/${meshdir}`;
    const vfs = new mujoco.MjVFS();
    vfs.addBuffer("scene.xml", new TextEncoder().encode(xml));
    for (const name of [...xml.matchAll(/<mesh\s+file="([^"]+)"/g)].map((m) => m[1])) {
      // 注意：URL 用 meshDir，**VFS 里的路径仍然按 XML 里的 meshdir 注册**
      // （MuJoCo 拿 meshdir 拼出来找，路径必须对得上）
      const buf = new Uint8Array(await (await fetch(`${meshDir}/${name}`)).arrayBuffer());
      vfs.addBuffer(`${meshdir}/${name}`, buf);
    }
    const model = mujoco.MjModel.from_xml_string(xml, vfs);
    return new DuckSim({ mujoco, model, data: new mujoco.MjData(model) });
  }
}
