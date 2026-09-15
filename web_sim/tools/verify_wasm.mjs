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
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import loadMujoco from "@mujoco/mujoco";
import { loadScene } from "./load_scene.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const REPO = path.resolve(ROOT, "..");
const sceneId = process.argv[2] || "duck_workspace_v1";
const sceneDir = path.join(ROOT, "assets", sceneId);

const REFERENCE = {
  duck_workspace_v1: { trunk: [0.13079, 0.00058, 0.05089], ball: [0.9, 0.0], expect: 200 },
};

/** 与 flatten_scene.py 的 DEFAULT_POSE 一致。 */
const HOME = Float64Array.from([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491,
  0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530]);

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

  // 2) 把场景与 mesh 塞进 MjVFS（mesh 目录从 XML 的 meshdir 读，见 load_scene.mjs）
  const { model, data, meshCount, meshdir } = await loadScene(mujoco, sceneDir);

  // 4) 关键对象是否解析出来（mj_name2id 的第 1 个参数是对象类型：1=body）
  const OBJ_BODY = 1;
  const ids = {};
  for (const n of ["trunk_base", "ball", "obj_cube_red", "beacon_target"]) {
    ids[n] = mujoco.mj_name2id(model, OBJ_BODY, n);
  }

  // 5) 跑摊平时由 Python 生成的那套参考协议（见 flatten_scene.py 的 REFERENCE_PROTOCOL）
  const ref = JSON.parse(await readFile(path.join(sceneDir, "reference.json"), "utf8"));
  mujoco.mj_resetDataKeyframe(model, data, 0);
  data.qvel.fill(0);
  data.ctrl.set(HOME);
  mujoco.mj_forward(model, data);
  const steps = 200;
  for (let i = 0; i < steps; i++) mujoco.mj_step(model, data);

  const trunk = [data.xpos[ids.trunk_base * 3], data.xpos[ids.trunk_base * 3 + 1], data.xpos[ids.trunk_base * 3 + 2]];
  const close = (a, b, tol = 5e-4) => Math.abs(a - b) <= tol;
  const trunkOk = trunk.every((v, i) => close(v, ref.trunk[i]));
  // 逐 body 比：比只看 trunk 强得多，而且任何场景都适用
  const bodyDiffs = [];
  for (const [name, xyz] of Object.entries(ref.bodies)) {
    const b = mujoco.mj_name2id(model, OBJ_BODY, name);
    if (b < 0) { bodyDiffs.push([name, "缺失"]); continue; }
    for (let k = 0; k < 3; k++) {
      const d = Math.abs(data.xpos[b * 3 + k] - xyz[k]);
      if (d > 5e-4) bodyDiffs.push([name, d.toExponential(1)]);
    }
  }
  // 所有 geom 的位置哈希：一条就能覆盖整车，不会漏掉某个关节
  const hashOf = (arr) => {
    const s = Array.from(arr).map((v) => Number(v).toFixed(6)).join(",");
    return createHash("sha256").update(s).digest("hex").slice(0, 16);
  };
  const wasmHash = hashOf(data.geom_xpos);
  const hashOk = wasmHash === ref.geomXposHash;

  console.log("=== MuJoCo WASM × 我们的场景 ===");
  console.log(`场景            : ${sceneId}`);
  console.log(`参考协议        : ${ref.protocol}`);
  console.log(`WASM 模块加载   : ${tLoad} ms`);
  console.log(`mesh            : ${meshCount} 个（共享目录 ${meshdir}）`);
  console.log(`model           : nbody=${model.nbody} ngeom=${model.ngeom} nu=${model.nu} nq=${model.nq}`);
  console.log(`${steps} 步后 trunk   : [${trunk.map((v) => v.toFixed(5)).join(", ")}]`);
  console.log(`Python 参考 trunk: [${ref.trunk.map((v) => v.toFixed(5)).join(", ")}]`);
  console.log(`一致性          : trunk ${trunkOk ? "OK" : "DIFF"} | ` +
              `${Object.keys(ref.bodies).length} 个 body ${bodyDiffs.length ? "DIFF " + JSON.stringify(bodyDiffs.slice(0, 4)) : "OK"} | ` +
              `geom 位置哈希 ${hashOk ? "OK" : `DIFF ${wasmHash} vs ${ref.geomXposHash}`}`);
  const ok = trunkOk && bodyDiffs.length === 0 && hashOk;
  console.log(ok ? "\n结果: PASS — 官方 MuJoCo WASM 的物理与 Python 端逐步一致（不只 trunk，全部 body + 全部 geom）"
                 : "\n结果: FAIL — 需要排查");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => { console.error("异常:", e); process.exit(1); });
