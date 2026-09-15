/**
 * 无浏览器验证：官方 MuJoCo WASM 能否加载我们**摊平后的场景**，物理是否与 Python 端一致。
 *
 * 这是"浏览器端仿真"的第一道关卡：物理不通，后面渲染和决策都不用谈。
 * 参照 Python 端同场景 200 步的参考值（mujoco 3.12.0）：
 *   trunk = (0.13079, 0.00058, 0.05089)   ball = (0.90000, 0.00000)
 *
 * 运行：node web_sim/tools/verify_wasm.mjs [sceneId]
 */
import { readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import loadMujoco from "@mujoco/mujoco";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const REPO = path.resolve(ROOT, "..");
const sceneId = process.argv[2] || "duck_workspace_v1";
const sceneDir = path.join(ROOT, "assets", sceneId);

const REFERENCE = {
  duck_workspace_v1: { trunk: [0.13079, 0.00058, 0.05089], ball: [0.9, 0.0], expect: 200 },
};

async function main() {
  if (!existsSync(sceneDir)) {
    console.error(`找不到摊平后的场景：${sceneDir}\n先运行: python web_sim/tools/flatten_scene.py ${sceneId}`);
    process.exit(2);
  }

  // 1) 加载 WASM 模块（Node 下需要显式告诉它 wasm 在哪）
  const wasmPath = path.join(REPO, "web_sim/node_modules/@mujoco/mujoco/mujoco.wasm");
  const t0 = Date.now();
  const mujoco = await loadMujoco({ locateFile: () => pathToFileURL(wasmPath).href });
  const tLoad = Date.now() - t0;

  // 2) 把场景与 mesh 塞进 MjVFS：主 XML 在根，mesh 在 assets/ 下（与 XML 的 meshdir 对应）
  const xml = await readFile(path.join(sceneDir, "scene.xml"), "utf8");
  const meshFiles = await readdir(path.join(sceneDir, "assets"));
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer("scene.xml", new TextEncoder().encode(xml));
  let bytes = 0;
  for (const f of meshFiles) {
    const buf = await readFile(path.join(sceneDir, "assets", f));
    bytes += buf.length;
    vfs.addBuffer(`assets/${f}`, buf);
  }

  // 3) 用 XML 字符串 + VFS 建模
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);

  // 4) 关键对象是否解析出来（mj_name2id 的第 1 个参数是对象类型：1=body）
  const OBJ_BODY = 1;
  const ids = {};
  for (const n of ["trunk_base", "ball", "obj_cube_red", "beacon_target"]) {
    ids[n] = mujoco.mj_name2id(model, OBJ_BODY, n);
  }

  // 5) 与 Python 端一致：keyframe 复位 + 200 步
  mujoco.mj_resetDataKeyframe(model, data, 0);
  const steps = 200;
  for (let i = 0; i < steps; i++) mujoco.mj_step(model, data);

  const trunk = [data.xpos[ids.trunk_base * 3], data.xpos[ids.trunk_base * 3 + 1], data.xpos[ids.trunk_base * 3 + 2]];
  const ball = [data.xpos[ids.ball * 3], data.xpos[ids.ball * 3 + 1]];

  const ref = REFERENCE[sceneId];
  const close = (a, b, tol = 5e-4) => Math.abs(a - b) <= tol;
  const trunkOk = ref ? trunk.every((v, i) => close(v, ref.trunk[i])) : null;
  const ballOk = ref ? ball.every((v, i) => close(v, ref.ball[i])) : null;

  console.log("=== MuJoCo WASM × 我们的场景 ===");
  console.log(`场景            : ${sceneId}`);
  console.log(`WASM 模块加载   : ${tLoad} ms`);
  console.log(`mesh            : ${meshFiles.length} 个, ${(bytes / 1048576).toFixed(2)} MB`);
  console.log(`model           : nbody=${model.nbody} ngeom=${model.ngeom} nu=${model.nu} nq=${model.nq}`);
  console.log(`关键 body id    : ${JSON.stringify(ids)}`);
  console.log(`${steps} 步后 trunk   : [${trunk.map((v) => v.toFixed(5)).join(", ")}]`);
  console.log(`${steps} 步后 ball    : [${ball.map((v) => v.toFixed(5)).join(", ")}]`);
  if (ref) {
    console.log(`Python 参考 trunk: [${ref.trunk.join(", ")}]  ball: [${ref.ball.join(", ")}]`);
    console.log(`一致性          : trunk ${trunkOk ? "OK" : "DIFF"}   ball ${ballOk ? "OK" : "DIFF"}`);
  }
  const ok = Object.values(ids).every((v) => v >= 0) && (trunkOk !== false) && (ballOk !== false);
  console.log(ok ? "\n结果: PASS — 官方 MuJoCo WASM 可以直接加载我们的场景，物理与 Python 端一致"
                 : "\n结果: FAIL — 需要排查");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => { console.error("异常:", e); process.exit(1); });
