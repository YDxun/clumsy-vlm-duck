/**
 * 闭环验证：官方 MuJoCo WASM（物理）+ onnxruntime-web（策略）能否跑起我们的鸭子。
 *
 * 复刻 `_dev_cockpit/sim_server.py` 的 local_step：
 *   61 维观测 = 陀螺(3) + 重力方向(3) + 关节位置偏差(14) + 关节速度(14) + 上一步动作(14) + 指令(3) + 零填充(10)
 *   50 Hz 控制，每个控制步 4 次物理子步；ctrl = DEFAULT_POSE + action。
 *
 * 渲染仍走 Python/EGL（本脚本只验证"物理+策略"这条决定性的链路）。
 * 运行：node web_sim/tools/verify_policy.mjs [policy] [seconds]
 */
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web";
import { loadScene } from "./load_scene.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const REPO = path.resolve(ROOT, "..");

// 与 Python 端一字不差（sim_server.py）
const DEFAULT_POSE = Float64Array.from([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
  0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530]);
const PHYS_DT = 0.005, DECIMATION = 4, CONTROL_DT = PHYS_DT * DECIMATION;
const OBS_DIM = 61, NJ = 14;

// 从 WASM 模块读 mjtObj 枚举值。曾经把 SENSOR 硬编码成 12，
// 而 12 是 mjOBJ_HFIELD —— 结果 sensor_adr[undefined] 退化成 undefined，
// 陀螺仪观测恒为 0，策略失去角速度反馈，鸭子一走就摔。
const mjObj = (mujoco, name) => {
  const e = mujoco.mjtObj && mujoco.mjtObj[name];
  if (e && typeof e === "object" && typeof e.value === "number") return e.value;
  if (typeof e === "number") return e;
  throw new Error(`无法解析 mjtObj.${name}`);
};
const FORCE_LIMIT = 0.3459739511711113 * 1.75;

const policyName = process.argv[2] || "alpha_stand";
const seconds = Number(process.argv[3] || 6);
const vx = Number(process.argv[4] || 0);          // 前进指令 m/s（0 = 站立）
const wz = Number(process.argv[5] || 0);          // 转向指令 rad/s
const sceneId = "duck_workspace_v1";

function quatRotateInverse(quat, vec) {
  const w = quat[0], x = quat[1], y = quat[2], z = quat[3];
  const t = [2 * (y * vec[2] - z * vec[1]), 2 * (z * vec[0] - x * vec[2]), 2 * (x * vec[1] - y * vec[0])];
  return [vec[0] - w * t[0] + (y * t[2] - z * t[1]),
          vec[1] - w * t[1] + (z * t[0] - x * t[2]),
          vec[2] - w * t[2] + (x * t[1] - y * t[0])];
}

async function main() {
  const mujoco = await loadMujoco({
    locateFile: () => pathToFileURL(path.join(REPO, "web_sim/node_modules/@mujoco/mujoco/mujoco.wasm")).href,
  });

  const sceneDir = path.join(ROOT, "assets", sceneId);
  const { model, data } = await loadScene(mujoco, sceneDir, { keyframe: -1 });

  // 注意：LocalSim 在 Python 里会把执行器力矩上限覆写成 ±0.605454，
  // 但官方 WASM 的 model.actuator_forcerange / actuator_forcelimited 在 JS 侧不可写
  // （抛 Embind 的 `unknown type ...memory_viewIbEE`）。
  // 所以这条覆写已经在 flatten_scene.py 里烘进 XML，运行时无需、也无法再改。
  // 同样的原因：model.opt.timestep 直接取自 XML（<option timestep=0.005>）。

  const OBJ_BODY = mjObj(mujoco, "mjOBJ_BODY");
  const OBJ_SENSOR = mjObj(mujoco, "mjOBJ_SENSOR");
  const trunk = mujoco.mj_name2id(model, OBJ_BODY, "trunk_base");
  const gyroAdr = model.sensor_adr[mujoco.mj_name2id(model, OBJ_SENSOR, "imu_ang_vel")];

  // actuator -> joint 映射（actuator_trnid 是扁平数组，每执行器 2 个槽）
  const jq = [], jv = [];
  for (let i = 0; i < model.nu; i++) {
    const jid = model.actuator_trnid[i * 2];
    jq.push(model.jnt_qposadr[jid]);
    jv.push(model.jnt_dofadr[jid]);
  }

  // 策略
  const policyPath = path.join(ROOT, "assets", "policies", `${policyName}.onnx`);
  const session = await ort.InferenceSession.create(await readFile(policyPath));
  const inputName = session.inputNames[0], outputName = session.outputNames[0];

  mujoco.mj_resetDataKeyframe(model, data, 0);
  data.ctrl.set(DEFAULT_POSE);
  mujoco.mj_forward(model, data);

  const last = new Float64Array(NJ);
  const cmd = new Float64Array([vx, 0, wz]);
  const obs = new Float32Array(OBS_DIM);
  const steps = Math.round(seconds / CONTROL_DT);
  const trace = [];
  let sawNaN = false;

  for (let s = 0; s < steps; s++) {
    const q = data.xquat.subarray(trunk * 4, trunk * 4 + 4);
    const g = quatRotateInverse(q, [0, 0, -1]);
    obs.set(data.sensordata.subarray(gyroAdr, gyroAdr + 3), 0);
    obs.set(g, 3);
    for (let i = 0; i < NJ; i++) obs[6 + i] = data.qpos[jq[i]] - DEFAULT_POSE[i];
    for (let i = 0; i < NJ; i++) obs[20 + i] = data.qvel[jv[i]];
    obs.set(last, 34);
    obs.set(cmd, 48);                    // 49..60 保持 0
    if (!obs.every(Number.isFinite)) { sawNaN = true; break; }

    const out = await session.run({ [inputName]: new ort.Tensor("float32", obs, [1, OBS_DIM]) });
    const action = out[outputName].data;
    last.set(action);
    for (let i = 0; i < NJ; i++) data.ctrl[i] = DEFAULT_POSE[i] + action[i];
    for (let i = 0; i < DECIMATION; i++) mujoco.mj_step(model, data);

    if (s % Math.round(0.5 / CONTROL_DT) === 0) {
      const gg = quatRotateInverse(data.xquat.subarray(trunk * 4, trunk * 4 + 4), [0, 0, -1]);
      trace.push([Number((s * CONTROL_DT).toFixed(2)), Number((-gg[2]).toFixed(3))]);
    }
  }

  const gg = quatRotateInverse(data.xquat.subarray(trunk * 4, trunk * 4 + 4), [0, 0, -1]);
  const upright = -gg[2];
  const x = data.xpos[trunk * 3];
  console.log("=== WASM 物理 + onnxruntime-web 策略（闭环）===");
  console.log(`场景 / 策略     : ${sceneId} / ${policyName}.onnx`);
  console.log(`观测维度        : ${obs.length}（期望 ${OBS_DIM}）`);
  console.log(`控制步          : ${steps} 步 / ${seconds}s（50 Hz，每步 ${DECIMATION} 次物理子步）`);
  console.log(`upright 轨迹    : ${JSON.stringify(trace)}`);
  console.log(`最终 upright    : ${upright.toFixed(4)}   前进 x = ${x.toFixed(4)} m`);
console.log(`指令            : vx=${vx} wz=${wz}`);
  const moved = x - 0;   // 起点 x ≈ 0
  const ok = !sawNaN && upright > 0.9 && (vx === 0 || moved > 0.05);
  console.log(ok ? "\n结果: PASS — WASM 物理与 ONNX 策略在浏览器技术栈上闭环成立（鸭子站立稳定）"
                 : "\n结果: FAIL — 需要排查");
  process.exit(ok ? 0 : 1);
}
main().catch((e) => { console.error("异常:", e); process.exit(1); });
