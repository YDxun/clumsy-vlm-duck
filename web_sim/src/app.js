/**
 * 步骤① 验证页：把 MuJoCo WASM 的鸭子画出来。
 *
 * 这一版有意**不含决策层**——只证明下面这条链路在浏览器里成立：
 *   XML+STL / ONNX 资源  ->  MuJoCo WASM 物理  ->  onnxruntime-web 策略
 *                        ->  three.js 按 geom_xpos/geom_xmat 画出来
 *
 * 页面另外暴露一组 `window.__sim` 钩子，给 tools/verify_render.mjs 用
 * （headless Chrome 里做像素级断言，而不是靠肉眼看截图）。
 */
import * as THREE from "three";
import { DuckSim, configureOrt, CONTROL_DT } from "./duck.js";
import { DuckView } from "./view.js";

const SCENE = "duck_workspace_v1";
const POLICY = "alpha_walking";

const $ = (id) => document.getElementById(id);
const boot = $("boot"), bootMsg = $("boot-msg"), logEl = $("log");
const lines = [];
function log(msg) {
  lines.push(msg);
  if (lines.length > 200) lines.shift();
  logEl.textContent = lines.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

let duck = null, view = null, ready = false;
const control = { running: true, cmd: [0, 0, 0] };

// ---------------------------------------------------------------- 像素工具
// 在页面内部分析画面。把 Uint8ClampedArray 整体搬过 CDP 太慢也没必要，
// 所以快照存在页面里，只把结论（占比/质心/差异）返回给验证脚本。
const snaps = new Map();
const CLEAR = [11, 17, 24];   // 与 view.js 的 setClearColor(0x0b1118) 一致

function grab(name) {
  const c = $("canvas");
  const w = c.width, h = c.height;
  const ctx2d = grab._ctx || (grab._ctx = document.createElement("canvas").getContext("2d", { willReadFrequently: true }));
  grab._ctx.canvas.width = w; grab._ctx.canvas.height = h;
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.drawImage(c, 0, 0, w, h);
  const img = ctx2d.getImageData(0, 0, w, h);
  const rec = { w, h, data: img.data };
  if (name) snaps.set(name, rec);
  return rec;
}

function stats(rec, { from = null } = {}) {
  const { w, h, data } = rec;
  const colors = new Set();
  let fg = 0, sx = 0, sy = 0, changed = 0, csx = 0, csy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      colors.add((r << 16) | (g << 8) | b);
      if (Math.abs(r - CLEAR[0]) + Math.abs(g - CLEAR[1]) + Math.abs(b - CLEAR[2]) > 24) {
        fg++; sx += x; sy += y;
      }
      if (from) {
        const j = i;
        const d = Math.abs(r - from.data[j]) + Math.abs(g - from.data[j + 1]) + Math.abs(b - from.data[j + 2]);
        if (d > 24) { changed++; csx += x; csy += y; }
      }
    }
  }
  const n = w * h;
  return {
    width: w, height: h, pixels: n,
    uniqueColors: colors.size,
    foreground: fg, foregroundFrac: fg / n,
    centroid: fg ? { x: sx / fg, y: sy / fg } : null,
    changed, changedFrac: from ? changed / n : null,
    changedCentroid: changed ? { x: csx / changed, y: csy / changed } : null,
  };
}

// ---------------------------------------------------------------- 主循环
async function loop() {
  let last = performance.now(), acc = 0, frames = 0, fpsAt = last;
  while (true) {
    await new Promise((r) => requestAnimationFrame(r));
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!ready) continue;
    if (control.running) {
      acc = Math.min(acc + dt, 0.2);
      let n = 0;
      while (acc >= CONTROL_DT && n < 5) {
        await duck.stepAsync(control.cmd);
        acc -= CONTROL_DT;
        n++;
      }
    }
    view.frame();
    if (++frames >= 10) {
      $("hud-fps").textContent = (frames / ((now - fpsAt) / 1000)).toFixed(0);
      frames = 0; fpsAt = now;
    }
    $("hud-steps").textContent = String(duck.steps);
    $("s-time").textContent = `${(duck.steps * CONTROL_DT).toFixed(2)} s`;
    $("s-pos").textContent = `${duck.pose.x.toFixed(3)}, ${duck.pose.y.toFixed(3)}`;
    $("s-head").textContent = `${((duck.pose.heading * 180) / Math.PI).toFixed(1)}°`;
    $("s-up").textContent = duck.upright().toFixed(3);
    $("s-ball").textContent = `${duck.ballPose.x.toFixed(2)}, ${duck.ballPose.y.toFixed(2)}`;
  }
}

async function boot_() {
  try {
    configureOrt({
      // ort.min.mjs 是“外部 wasm”构建，运行时才去取 ort-wasm-*.wasm，
      // 必须把 dist 目录绝对 URL 告诉它（末尾斜杠不能省）。
      wasmPaths: new URL("../node_modules/onnxruntime-web/dist/", import.meta.url).href,
      numThreads: 1,
    });
    const t0 = performance.now();
    duck = await DuckSim.load({ sceneUrl: `./assets/${SCENE}/scene.xml`, mjcfBase: `./assets/${SCENE}/` });
    log(`[场景] ${SCENE} 已加载：ngeom=${duck.model.ngeom} nbody=${duck.model.nbody} nu=${duck.model.nu}`);
    log(`[物理] WASM 就绪 ${(performance.now() - t0).toFixed(0)} ms（含 mesh 解析）`);

    const t1 = performance.now();
    await duck.loadPolicy(`./assets/policies/${POLICY}.onnx`);
    log(`[策略] ${POLICY}.onnx 就绪 ${(performance.now() - t1).toFixed(0)} ms`);

    view = new DuckView({ canvas: $("canvas"), THREE, duck });
    log(`[渲染] 建了 ${view.meshes.length} 个 geom mesh（共 ${duck.model.ngeom} 个）`);
    view.frame();

    $("s-scene").textContent = SCENE;
    $("s-policy").textContent = POLICY;
    boot.style.display = "none";
    ready = true;
    window.__duckReady = true;
    loop();
  } catch (e) {
    window.__duckError = String((e && e.stack) || e);
    bootMsg.textContent = String(e);
    log(`[错误] ${e}`);
    console.error(e);
  }
}

// ---------------------------------------------------------------- 控件
for (const b of document.querySelectorAll("[data-mode]")) {
  b.onclick = () => {
    for (const o of document.querySelectorAll("[data-mode]")) o.classList.toggle("on", o === b);
    view && view.setMode(b.dataset.mode);
  };
}
for (const b of document.querySelectorAll("[data-cmd]")) {
  b.onclick = () => {
    if (!duck) return;
    control.running = true;
    control.cmd = { stand: [0, 0, 0], fwd: [0.3, 0, 0], turn: [0, 0, 0.8] }[b.dataset.cmd];
    log(`[指令] ${b.textContent} -> vx=${control.cmd[0]} vy=${control.cmd[1]} wz=${control.cmd[2]}`);
  };
}
$("pause").onclick = (e) => {
  control.running = !control.running;
  e.target.textContent = control.running ? "暂停" : "继续";
  e.target.classList.toggle("on", !control.running);
};

// ---------------------------------------------------------------- 测试钩子
window.__duckReady = false;
window.__sim = {
  get duck() { return duck; },
  get view() { return view; },
  scene: SCENE, policy: POLICY,
  pause() { control.running = false; },
  resume() { control.running = true; },
  setMode(m) { view.setMode(m); view.frame(); },
  /** 停下来、重置、按固定动作跑 n 个控制步（确定性，不受 rAF 节奏影响）。 */
  async runSteps(n, cmd = [0, 0, 0]) {
    control.running = false;
    for (let i = 0; i < n; i++) await duck.stepAsync(cmd);
    view.frame();
    return duck.steps;
  },
  reset() { duck.reset(); view.frame(); },
  render() { view.frame(); },
  grab(name) { const r = grab(name); return stats(r); },
  diff(a, b) { return stats(snaps.get(b), { from: snaps.get(a) }); },
  snapshotInfo(a) { const s = snaps.get(a); return s ? { width: s.w, height: s.h } : null; },
  pixel(x, y) {
    const r = snaps.get("_live") || null;
    const s = r || grab();
    const i = (y * s.w + x) * 4;
    return [s.data[i], s.data[i + 1], s.data[i + 2], s.data[i + 3]];
  },
  /** MuJoCo 世界坐标 -> 屏幕像素（经 world 组 z-up→y-up 变换后投影）。 */
  projectPoint(xyz) {
    // world 组绕 x 轴 -90°：MuJoCo (X,Y,Z) -> three.js (X, Z, -Y)
    const v = new THREE.Vector3(xyz[0], xyz[2], -xyz[1]);
    v.project(view.camera);
    const c = $("canvas");
    return { x: (v.x + 1) / 2 * c.width, y: (1 - v.y) / 2 * c.height, ndcZ: v.z };
  },
  /** 某个 geom 的中心在屏幕上的像素位置。注意参数是 **geom 下标**，不是 body id。 */
  projectGeom(g) {
    const p = duck.data.geom_xpos;
    return this.projectPoint([p[g * 3], p[g * 3 + 1], p[g * 3 + 2]]);
  },
  /** 鸭子所有 geom 的世界坐标均值（“鸭子在哪”的一个几何代理），及其屏幕投影。 */
  duckAnchor() {
    const p = duck.data.geom_xpos, gs = duck.duckGeoms;
    const m = [0, 0, 0];
    for (const g of gs) for (let k = 0; k < 3; k++) m[k] += p[g * 3 + k] / gs.length;
    return { world: m, screen: this.projectPoint(m) };
  },
  /** 球心的世界坐标与屏幕投影。 */
  ballAnchor() {
    const b = duck.ballGeom[0], p = duck.data.geom_xpos;
    const m = [p[b * 3], p[b * 3 + 1], p[b * 3 + 2]];
    return { world: m, screen: this.projectPoint(m) };
  },
  geomInfo(g) {
    const m = duck.model;
    return { type: m.geom_type[g], bodyid: m.geom_bodyid[g], dataid: m.geom_dataid[g],
             rgba: [0, 1, 2, 3].map((i) => m.geom_rgba[g * 4 + i]) };
  },
  duckGeoms() { return duck.duckGeoms; },
  ballGeoms() { return duck.ballGeom; },
  /** 临时隐藏某些 geom（验证“画出来的东西确实来自物理状态”时做 A/B 差分用）。 */
  setHidden(geoms) {
    const set = new Set(geoms);
    for (const { index, mesh } of view.meshes) mesh.visible = !set.has(index);
  },
  /** 返回每个 mesh 的 matrixWorld 与 MuJoCo 位姿的偏差（矩阵行序/轴向对不对）。 */
  matrixCheck(sample = 24) {
    const { model, data } = duck;
    const out = [];
    const step = Math.max(1, Math.floor(view.meshes.length / sample));
    for (let k = 0; k < view.meshes.length; k += step) {
      const { index, mesh } = view.meshes[k];
      // MuJoCo mesh_vert 是 mesh 局部坐标；这里抽第 0 个顶点做对比
      let local;
      if (model.geom_type[index] === 7 && model.geom_dataid[index] >= 0) {
        const id = model.geom_dataid[index], at = model.mesh_vertadr[id] * 3;
        local = new THREE.Vector3(model.mesh_vert[at], model.mesh_vert[at + 1], model.mesh_vert[at + 2]);
      } else {
        local = new THREE.Vector3(1, 0, 0);   // 基本体：局部单位向量
      }
      const inScene = mesh.localToWorld(local.clone());
      const M = data.geom_xmat, p = data.geom_xpos, o = index * 9, q = index * 3;
      const lx = local.x, ly = local.y, lz = local.z;
      const wx = M[o] * lx + M[o + 1] * ly + M[o + 2] * lz + p[q];
      const wy = M[o + 3] * lx + M[o + 4] * ly + M[o + 5] * lz + p[q + 1];
      const wz = M[o + 6] * lx + M[o + 7] * ly + M[o + 8] * lz + p[q + 2];
      const err = Math.hypot(inScene.x - wx, inScene.y - wz, inScene.z + wy);
      out.push({ geom: index, type: model.geom_type[index], err });
    }
    return { max: Math.max(...out.map((o) => o.err)), samples: out };
  },
};

boot_();
