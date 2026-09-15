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
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DuckSim, configureOrt, CONTROL_DT } from "./duck.js";
import { DuckView } from "./view.js";
import { SceneIndex } from "./scene.js";
import { DuckAgent } from "./agent.js";
import { PROVIDERS } from "./llm.js";
import { ActionInterpreter } from "./interpreter.js";
import { ACTION_SPECS } from "./actions.js";

const SCENE = "duck_workspace_v1";
const POLICY = "alpha_walking";
const ASSET_BASE = "./assets";

const $ = (id) => document.getElementById(id);
const boot = $("boot"), bootMsg = $("boot-msg"), logEl = $("log");
const lines = [];
function log(msg) {
  lines.push(msg);
  if (lines.length > 200) lines.shift();
  logEl.textContent = lines.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

let duck = null, view = null, sceneIndex = null, agent = null, ready = false;
let manifest = null, sceneBase = ASSET_BASE;
let activeSceneId = null;
const policySessions = new Map();   // ONNX 会话与场景无关，换场景时复用
const policySessionsById = {};      // 策略名 -> session，技能 token 靠它热切换
const sceneCache = new Map();       // 场景 id -> DuckSim，换回来秒开
const SCENE_CACHE_MAX = 3;          // 三个场景都留得住；再多就得靠 dispose() 腾地方
const control = { running: true, cmd: [0, 0, 0] };
const driver = { agent: false };      // true = 由决策层下发指令，false = 手动按钮
const manualIt = new ActionInterpreter();   // 手动按钮也走同一个解释器：按 token 走，不按速度走
let switchingScene = false;           // 换场景期间停掉渲染与物理，CPU 全给 MuJoCo 解析网格

// ---------------------------------------------------------------- 像素工具
// 在页面内部分析画面。把 Uint8ClampedArray 整体搬过 CDP 太慢也没必要，
// 所以快照存在页面里，只把结论（占比/质心/差异）返回给验证脚本。
const snaps = new Map();
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
  // 背景参考色**从画面顶部几行采样**，不能写死清屏色：
  // 场景美化之后背景是天空渐变，写死 (11,17,24) 会把整幅画都算成前景（校验抓到过这个回归）。
  let bg = [0, 0, 0];
  const bgRows = Math.min(4, h);
  for (let y = 0; y < bgRows; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      bg[0] += data[i] / (bgRows * w);
      bg[1] += data[i + 1] / (bgRows * w);
      bg[2] += data[i + 2] / (bgRows * w);
    }
  }
  const colors = new Set();
  let fg = 0, sx = 0, sy = 0, changed = 0, csx = 0, csy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      colors.add((r << 16) | (g << 8) | b);
      if (Math.abs(r - bg[0]) + Math.abs(g - bg[1]) + Math.abs(b - bg[2]) > 24) {
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
    background: bg.map((v) => Math.round(v)),
    uniqueColors: colors.size,
    foreground: fg, foregroundFrac: fg / n,
    centroid: fg ? { x: sx / fg, y: sy / fg } : null,
    changed, changedFrac: from ? changed / n : null,
    changedCentroid: changed ? { x: csx / changed, y: csy / changed } : null,
  };
}

// ---------------------------------------------------------------- 主循环
/**
 * 装载 policies.json 里列出的所有策略。
 * 缺文件的直接跳过（策略文件不进仓库，克隆下来可能只有一部分），
 * 页面上对应的技能 token 会自动消失 —— 和 Python 的 available_tokens() 同一条原则。
 */
async function loadPolicies() {
  let list = [];
  try {
    const res = await fetch("./policies.json");
    if (res.ok) list = (await res.json()).policies || [];
  } catch { /* 没有清单就只加载默认走路策略 */ }
  const results = [];
  for (const p of list) {
    const url = `${ASSET_BASE}/policies/${p.file}`;
    try {
      const session = policySessions.get(url) || await DuckSim.createPolicySession(url);
      policySessions.set(url, session);
      policySessionsById[p.id] = session;
      results.push(p.id);
    } catch {
      if (p.required) results.push(`${p.id}（缺失，必需）`);
    }
  }
  log(`[策略] 可用 ${results.length} 个：${results.join(", ")}`);
  return results;
}

/** 读场景清单。可以指向任意静态托管（这就是「自定义导入场景」的入口）。 */
async function loadManifest(url = "./scenes.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`读不到场景清单 ${url}（HTTP ${res.status}）`);
  const data = await res.json();
  if (!Array.isArray(data.scenes) || !data.scenes.length) throw new Error(`${url} 里没有 scenes`);
  // 清单与场景目录的相对关系：清单在 web_sim/ 下，场景在 web_sim/assets/ 下
  const base = new URL(url, location.href);
  sceneBase = new URL(base.href.replace(/[^/]*$/, "") + "assets", base).href.replace(/\/$/, "");
  if (data.assetBase) sceneBase = new URL(data.assetBase, base).href.replace(/\/$/, "");
  return data;
}

/**
 * 激活一个场景。这是「换场景不用重启」的实现：
 * 重建 DuckSim（物理模型），复用 ONNX 会话与渲染器，只把 mesh 换掉。
 */
async function activateScene(id) {
  const entry = manifest.scenes.find((s) => s.id === id) || manifest.scenes[0];
  const t0 = performance.now();
  const dir = entry.dir || entry.id;
  const base = `${sceneBase}/${dir}`;
  const prevDuck = duck;
  switchingScene = true;
  try {
    let next = sceneCache.get(entry.id) || null;
    if (next) {
      sceneCache.delete(entry.id);     // LRU：刚用过的排到最后
      sceneCache.set(entry.id, next);
      next.reset();                    // 复用的模型要回到关键帧，别接着上一局的状态
      log(`[场景] ${entry.id} 命中缓存（解析网格是换场景唯一的大头，一次约 10 s）`);
    } else {
      next = await DuckSim.load({ sceneUrl: `${base}/scene.xml`, mjcfBase: `${base}/` });
      while (sceneCache.size >= SCENE_CACHE_MAX) {
        const [oldId, old] = sceneCache.entries().next().value;
        sceneCache.delete(oldId);
        if (old !== duck) old.dispose();   // 绝不释放正在用的那个
      }
      sceneCache.set(entry.id, next);
    }
    const tModel = performance.now();

    const policyUrl = `${ASSET_BASE}/policies/${POLICY}.onnx`;
    if (!policySessions.has(policyUrl)) {
      policySessions.set(policyUrl, await DuckSim.createPolicySession(policyUrl));
    }
    await next.loadPolicy(policyUrl, { session: policySessions.get(policyUrl) });
    next.policies = policySessionsById;

    const [metadata, tasks] = await Promise.all([
      fetch(`${base}/metadata.json`).then((r) => (r.ok ? r.json() : {})),
      fetch(`${base}/tasks.json`).then((r) => (r.ok ? r.json() : {})),
    ]);
    const idx = new SceneIndex(metadata, tasks);

    const prevConfig = agent ? { ...agent.config } : null;
    activeSceneId = entry.id;
    duck = next;
    sceneIndex = idx;
    if (!view) {
      view = new DuckView({ canvas: $("canvas"), THREE, OrbitControls, duck });
      log(`[渲染] 建了 ${view.meshes.length} 个 geom mesh（共 ${duck.model.ngeom} 个）`);
    } else {
      view.rebuild(duck);
    }
    const tView = performance.now();
    agent = new DuckAgent({ duck, view, scene: idx });
    agent.policies = policySessionsById;
    agent.interpreter.policySwitch = (name, token) => {
      const session = policySessionsById[name];
      if (!session) return duck.policyName;
      const done = duck.setPolicy(session, name);
      if (token) log(`[技能] ${token} 接管：切到策略 ${name}`);
      return done;
    };
    manualIt.policySwitch = agent.interpreter.policySwitch;
    if (prevConfig) Object.assign(agent.config, prevConfig);
    agent.onRecord = renderRecord;

    $("scene-select").value = entry.id;
    $("s-scene").textContent = entry.id;
    const info = `${entry.title || entry.id}${entry.description ? " · " + entry.description : ""}`;
    $("scene-info").textContent =
      `${info}｜几何 ${duck.model.ngeom} · body ${duck.model.nbody} · 任务 ${idx.listTasks().length} 条 · ` +
      `实体 ${idx.entityNames().length} 个 · ${((entry.meshBytes || 0) / 1048576).toFixed(1)} MB 网格`;
    log(`[场景] ${entry.id} 切换完成 ${(performance.now() - t0).toFixed(0)} ms`);
    log(`       拆开看：解析模型 ${(tModel - t0).toFixed(0)} ms ｜ 重建渲染 ${(tView - tModel).toFixed(0)} ms ｜ 其余 ${(performance.now() - tView).toFixed(0)} ms`);
    return entry;
  } finally {
    switchingScene = false;
    // 旧模型只在**不在缓存里**时才释放。放进缓存的模型要留着复用，
    // 手滑把它 delete 掉的话，下次命中缓存拿到的就是个已释放对象。
    const stillCached = [...sceneCache.values()].includes(prevDuck);
    if (prevDuck && prevDuck !== duck && !stillCached) prevDuck.dispose();
  }
}

async function loop() {
  let last = performance.now(), acc = 0, frames = 0, fpsAt = last;
  while (true) {
    await new Promise((r) => requestAnimationFrame(r));
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!ready) continue;
    if (switchingScene) continue;      // 换场景时把这一帧整个让出去
    if (control.running) {
      acc = Math.min(acc + dt, 0.2);
      let n = 0;
      while (acc >= CONTROL_DT && n < 5) {
        if (driver.agent && agent) {
          const out = agent.tick(CONTROL_DT);
          await duck.stepAsync(out.cmd, out.headDelta);
          $("s-phase").textContent = out.phase;
        } else if (manualIt.busy) {
          const out = manualIt.tick(CONTROL_DT);
          await duck.stepAsync(out.command, manualIt.headOverride());
        } else {
          await duck.stepAsync(control.cmd);
        }
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
    $("s-decisions").textContent = String(agent ? agent.records.length : 0);
    if (agent && agent.minRange != null) $("s-minrange").textContent = `${agent.minRange.toFixed(2)} m`;
    if (duck.lastTwist) {
      $("s-twist").textContent = duck.lastTwist.map((v) => v.toFixed(2)).join(", ");
    }
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
    await loadPolicies();
    manifest = await loadManifest("./scenes.json");
    log(`[清单] scenes.json：${manifest.scenes.length} 个场景 —— ${manifest.scenes.map((s) => s.id).join(", ")}`);
    const want = new URLSearchParams(location.search).get("scene");
    await activateScene(want || manifest.scenes[0].id);
    log(`[物理] WASM 就绪 ${(performance.now() - t0).toFixed(0)} ms（含 mesh 解析）`);
    view.frame();

    $("s-policy").textContent = POLICY;
    boot.style.display = "none";
    ready = true;
    window.__duckReady = true;
    setupUi();
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
    driver.agent = false;          // 手动遥控优先：先把手动模式抢回来
    control.running = true;
    // 手动按钮下发的是**离散 token**，和 VLM 用的是同一套词表、同一个解释器。
    // 以前下发的是原始速度（前进 0.3 会一直走到撞墙、原地转 0.8 低于步态地板根本不动）。
    const token = b.dataset.cmd;
    const res = manualIt.start(token);
    log(`[手动] ${token} -> 速度指令 ${res.command.join(",")}（${ACTION_SPECS[token]?.durationS ?? 0}s）`);
  };
}
$("pause").onclick = (e) => {
  control.running = !control.running;
  e.target.textContent = control.running ? "暂停" : "继续";
  e.target.classList.toggle("on", !control.running);
};

// ---------------------------------------------------------------- 页面 UI
const LS_KEY = "duckvlm.config.v1";

/** 决策日志卡片：把「模型看到的图 + 它选了什么 + 规则有没有改它」摊开给用户看。 */
function renderRecord(rec) {
  const box = $("decisions");
  if (box.firstChild && box.firstChild.className === "hint") box.textContent = "";
  const card = document.createElement("div");
  card.className = "card " + (rec.source === "vlm" ? "vlm" : "rule") + (rec.rules?.length ? " overridden" : "");
  const bits = [];
  if (rec.provider) bits.push(`${rec.provider}/${rec.model}`);
  if (rec.latencyMs != null) bits.push(`${rec.latencyMs.toFixed(0)} ms`);
  if (rec.rules?.length) bits.push(`规则接管：${rec.rules.join(", ")}`);
  if (rec.note && rec.note !== "ok") bits.push(rec.note);
  card.innerHTML = `
    <div class="head">
      <span>#${rec.stepIndex} <span class="tok">${rec.token}</span>${
        rec.proposed && rec.proposed !== rec.token ? ` <span class="warn">（模型原本给 ${rec.proposed}）</span>` : ""
      }</span>
      <span class="meta">${rec.source === "vlm" ? "VLM" : "规则"}</span>
    </div>
    ${rec.image ? `<img alt="这一步模型看到的画面" src="${rec.image}">` : ""}
    <div class="meta">${bits.join(" · ") || "—"}</div>`;
  box.prepend(card);
  while (box.children.length > 12) box.lastChild.remove();
}

/** 场景下拉 + 自定义清单入口。换场景不重启：重建模型、复用渲染器与 ONNX 会话。 */
function setupSceneSelect() {
  const sceneSel = $("scene-select");
  const fill = () => {
    sceneSel.innerHTML = "";
    for (const s of manifest.scenes) {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = `${s.title || s.id}${s.tags?.length ? "（" + s.tags.join("/") + "）" : ""}`;
      sceneSel.appendChild(o);
    }
    sceneSel.value = activeSceneId || manifest.scenes[0].id;
  };
  fill();
  const go = async (id) => {
    if (!id) return;
    $("scene-info").textContent = "正在切换场景…";
    const t0 = performance.now();
    try {
      await activateScene(id);
      refreshTaskSelect();
      log(`[场景] 切换总耗时 ${(performance.now() - t0).toFixed(0)} ms（页面与服务都没有重启）`);
    } catch (e) {
      $("scene-info").textContent = `切换失败：${e}`;
      log(`[错误] 切换场景失败：${e}`);
    }
  };
  sceneSel.onchange = () => go(sceneSel.value);
  $("manifest-load").onclick = async () => {
    const url = $("manifest-url").value.trim();
    if (!url) return;
    try {
      manifest = await loadManifest(url);
      fill();
      await go(manifest.scenes[0].id);
      log(`[清单] 已切到 ${url}（${manifest.scenes.length} 个场景）`);
    } catch (e) {
      log(`[错误] 读清单失败：${e}`);
      $("scene-info").textContent = String(e);
    }
  };
}

/** 任务下拉：直接列场景包里的 task id，避免“输入对不上就没评分”。换场景后要重建。 */
function refreshTaskSelect() {
  const sel = $("task-select");
  const tasks = sceneIndex.listTasks();
  sel.innerHTML = "";
  for (const t of tasks) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = `${t.id} — ${t.instruction_zh || t.instruction_en || ""}`;
    sel.appendChild(opt);
  }
  const free = document.createElement("option");
  free.value = "__free__";
  free.textContent = "（用下面那句话）";
  sel.appendChild(free);

  sel.onchange = () => {
    if (sel.value !== "__free__") {
      const t = sceneIndex.task(sel.value);
      $("task-text").value = "";
      log(`[选择任务] ${sel.value}：${t?.instruction_zh || t?.instruction_en || ""}`);
    }
    applyTask();
  };
  $("task-text").onchange = () => { sel.value = "__free__"; applyTask(); };
  $("manip-mode").onchange = () => {
    if (!agent) return;
    agent.manipulationMode = $("manip-mode").value;
    agent.setTask({ taskId: sel.value === "__free__" ? "" : sel.value, text: $("task-text").value.trim() });
    log(`[操作方式] ${$("manip-mode").selectedOptions[0].textContent} → 本次判定为「${agent.manipMode}」`);
  };
  applyTask();
}

/**
 * 读取界面上的任务选择并交给 agent。放在模块级是因为「运行」按钮也要用它 ——
 * 之前定义在 refreshTaskSelect 里面，拆函数之后按钮就看不见了（运行时报 applyTask is not defined）。
 */
function applyTask() {
  const sel = $("task-select");
  const isFree = sel.value === "__free__";
  const text = isFree ? $("task-text").value.trim() : "";
  const task = agent.setTask({ taskId: isFree ? "" : sel.value, text });
  $("task-info").textContent = `目标 = ${task.target} · 意图 = ${task.intent} · 允许动作 ${task.allowed.length} 个`;
  $("s-target").textContent = task.target;
  $("s-minrange").textContent = "—";
  $("decisions").innerHTML = '<div class="hint">还没有决策。点“开始”。</div>';
  log(`[任务] ${task.taskId || "(自由文本)"} ${task.text} → target=${task.target} intent=${task.intent}`);
  return task;
}

function setupDecisionUi() {
  // ---- 决策模式 + BYO key（只存本机）
  const saved = (() => { try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch { return {}; } })();
  const prov = $("llm-provider");
  for (const [id, p] of Object.entries(PROVIDERS)) {
    const o = document.createElement("option");
    o.value = id; o.textContent = p.label;
    prov.appendChild(o);
  }
  const fillProvider = (id, keepValues = false) => {
    const p = PROVIDERS[id];
    prov.value = id;
    if (!keepValues) {
      $("llm-base").value = saved.baseUrl || p.defaultBaseUrl;
      $("llm-model").value = saved.model || p.defaultModel;
    }
    $("llm-hint").textContent = p.hint;
  };
  fillProvider(saved.provider || "openai", true);
  if (saved.baseUrl) $("llm-base").value = saved.baseUrl;
  if (saved.model) $("llm-model").value = saved.model;
  if (saved.key) $("llm-key").value = saved.key;
  prov.onchange = () => { fillProvider(prov.value); syncAgentConfig(); };

  const syncAgentConfig = () => {
    Object.assign(agent.config, {
      provider: prov.value,
      baseUrl: $("llm-base").value.trim(),
      model: $("llm-model").value.trim(),
      apiKey: $("llm-key").value.trim(),
    });
    localStorage.setItem(LS_KEY, JSON.stringify({
      provider: prov.value, baseUrl: $("llm-base").value.trim(),
      model: $("llm-model").value.trim(), key: $("llm-key").value.trim(),
    }));
  };
  for (const el of [$("llm-base"), $("llm-model"), $("llm-key")]) el.onchange = syncAgentConfig;

  for (const b of document.querySelectorAll("[data-decmode]")) {
    b.onclick = () => {
      for (const o of document.querySelectorAll("[data-decmode]")) o.classList.toggle("on", o === b);
      agent.config.mode = b.dataset.decmode;
      $("llm-box").hidden = b.dataset.decmode !== "llm";
      if (b.dataset.decmode === "llm") syncAgentConfig();
      log(`[模式] ${b.textContent}`);
    };
  }

  // ---- 运行控制
  agent.onRecord = renderRecord;
  $("run").onclick = () => {
    syncAgentConfig();
    const task = applyTask();
    if (agent.config.mode === "llm" && !agent.config.apiKey &&
        !/localhost|127\.0\.0\.1/.test(agent.config.baseUrl)) {
      log("[错误] VLM 模式需要 API key（只存在你本机，直接发给厂商）");
      return;
    }
    driver.agent = true;
    control.running = true;
    $("pause").textContent = "暂停";
    agent.phase = "idle";
    log(`[开始] ${task.taskId || task.text}，${agent.config.mode === "llm" ? agent.config.model : "规则模式"}`);
  };
  $("stop").onclick = () => { driver.agent = false; agent.phase = "finished"; log("[停止] 决策已停止，鸭子原地站住"); };
  $("reset").onclick = () => {
    driver.agent = false;
    duck.reset();
    agent.interpreter.resetHead();
    agent.records.length = 0;
    $("decisions").innerHTML = '<div class="hint">已复位。</div>';
    $("s-minrange").textContent = "—";
    view.frame();
    log("[复位] 鸭子与物体回到初始位姿");
  };

  // ---- 三视角拼图：一眼看到「鸭子眼里的世界 / 它在哪 / 全场」
  $("shot").onclick = async () => {
    const modes = [["workspace", "全景"], ["overhead", "全场俯视"], ["duck", "鸭子眼（VLM 视角）"]];
    const W = 480, H = 360;
    const out = document.createElement("canvas");
    out.width = W * 2 + 12; out.height = H * 2 + 12 + 22;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#0b1118"; ctx.fillRect(0, 0, out.width, out.height);
    ctx.font = "14px 'Microsoft YaHei', system-ui, sans-serif";
    for (let i = 0; i < modes.length; i++) {
      const [m, label] = modes[i];
      view.setMode(m); view.frame();
      await new Promise((r) => requestAnimationFrame(r));
      const x = 4 + (i % 2) * (W + 4), y = 22 + Math.floor(i / 2) * (H + 4);
      ctx.drawImage($("canvas"), 0, 0, $("canvas").width, $("canvas").height, x, y, W, H);
      ctx.fillStyle = "#7fd1ff"; ctx.fillText(label, x + 6, y - 6);
    }
    ctx.fillStyle = "#8b9bad";
    ctx.fillText(`DuckVLM · 任务：${agent.task.text} · 目标：${agent.task.target} · ${agent.config.mode === "llm" ? agent.config.model : "规则模式"}`,
                 8, out.height - 8);
    const a = document.createElement("a");
    a.download = `duckvlm_${Date.now()}.png`;
    a.href = out.toDataURL("image/png");
    a.click();
    log("[截图] 三视角拼图已下载");
  };

  // ---- 配色：默认美化，点一下切回场景包的原始材质（方便对照/排查）
  $("palette").onclick = (e) => {
    const next = view.palette === "duck" ? "raw" : "duck";
    view.applyPalette(next);
    e.target.textContent = next === "duck" ? "原始配色" : "美化配色";
    view.frame();
    log(`[配色] 切到 ${next === "duck" ? "美化配色" : "场景原始材质"}`);
  };
}

function setupUi() {
  setupSceneSelect();
  refreshTaskSelect();
  setupDecisionUi();
}

// ---------------------------------------------------------------- 测试钩子
window.__duckReady = false;
window.__sim = {
  get duck() { return duck; },
  get view() { return view; },
  get agent() { return agent; },
  get scene() { return sceneIndex; },
  get manual() { return manualIt; },
  sceneId: SCENE, policy: POLICY,
  pause() { control.running = false; },
  resume() { control.running = true; },
  /** 切换指令来源：true = 决策层（agent），false = 手动按钮。 */
  setDriver(useAgent) { driver.agent = !!useAgent; },
  /** agent 配置（BYO key、模式等）。 */
  configureAgent(patch) {
    Object.assign(agent.config, patch);
    // 让界面上的模式按钮跟着走，否则用脚本配好 VLM 模式、界面上还亮着「规则模式」
    if (patch.mode) {
      for (const b of document.querySelectorAll("[data-decmode]")) {
        b.classList.toggle("on", b.dataset.decmode === patch.mode);
      }
      $("llm-box").hidden = patch.mode !== "llm";
    }
    return { ...agent.config };
  },
  setTask(opts) { agent.records.length = 0; return agent.setTask(opts); },
  /**
   * 跑一段**完整闭环**：决策 -> 规则 -> 解释器 -> 物理，直到跑够决策数或步数。
   * dt 用控制步长（0.02 s）而不是真实墙钟时间，这样动作时长是仿真时间，
   * 结果与机器快慢无关、可复现。
   */
  async runAgent({ maxDecisions = 12, maxSteps = 1200, dt = 0.02 } = {}) {
    control.running = false;
    driver.agent = true;
    const runLog = [];
    let steps = 0;
    const t0 = performance.now();
    while (steps < maxSteps && agent.phase !== "finished" && agent.phase !== "error" &&
           agent.records.length < maxDecisions) {
      await new Promise((r) => setTimeout(r, 0));   // 让 fetch / 微任务有机会推进
      const out = agent.tick(dt);
      await duck.stepAsync(out.cmd, out.headDelta);
      steps++;
      runLog.push(`${agent.phase}:${out.cmd.map((v) => v.toFixed(2)).join(",")}`);
      // 每 10 步才重绘一次：无头 Chrome 走的是软件光栅化，逐帧重绘会把测试时间
      // 拖成十几分钟（630 步 ≈ 160 s）。相机是按需渲的，抓图和遮挡判断不受影响。
      if (steps % 10 === 0) view.frame();
    }
    view.frame();
    driver.agent = false;
    return {
      steps, decisions: agent.records.length, phase: agent.phase,
      wallMs: performance.now() - t0,
      pose: { ...duck.pose }, upright: duck.upright(),
      distanceToTarget: agent.records[0]?.state?.targetRangeM ?? null,
      minRange: agent.minRange,
      lastNote: agent.lastNote, error: agent.lastError,
      history: agent.history.slice(0, 12),
      tail: runLog.slice(-6),
    };
  },
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
  /** 调试用：把鸭子直接放到某个位置/朝向（自由关节 qpos），并清零速度。 */
  placeDuck({ x = 0, y = 0, yaw = 0, z = 0.125 } = {}) {
    const q = duck.data.qpos;
    q[0] = x; q[1] = y; q[2] = z;
    const h = yaw / 2;
    q[3] = Math.cos(h); q[4] = 0; q[5] = 0; q[6] = Math.sin(h);
    duck.data.qvel.fill(0);
    duck.mujoco.mj_forward(duck.model, duck.data);
    duck.lastAction.fill(0);
    view.frame();
    return { ...duck.pose };
  },
  async decodeDataUrl(url) {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  },
  /**
   * 头摄图的 A/B 差分：正常拍一张，把指定 geom 藏起来再拍一张，
   * 差异像素就是“这些东西在 VLM 眼里的位置”。用来验证状态传感器说的
   * “看得见/看不见”和画面上真实有没有，是同一件事。
   */
  async headCamDiff(geoms, { width = 320, height = 240 } = {}) {
    const a = view.captureHeadCam({ width, height });
    this.setHidden(geoms);
    const b = view.captureHeadCam({ width, height });
    this.setHidden([]);
    const [ia, ib] = await Promise.all([this.decodeDataUrl(a), this.decodeDataUrl(b)]);
    const ca = document.createElement("canvas"); ca.width = width; ca.height = height;
    const cb = document.createElement("canvas"); cb.width = width; cb.height = height;
    ca.getContext("2d").drawImage(ia, 0, 0);
    cb.getContext("2d").drawImage(ib, 0, 0);
    const da = ca.getContext("2d").getImageData(0, 0, width, height).data;
    const db = cb.getContext("2d").getImageData(0, 0, width, height).data;
    let changed = 0, sx = 0, sy = 0;
    for (let i = 0; i < da.length; i += 4) {
      const d = Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2]);
      if (d > 32) { const p = i / 4; changed++; sx += p % width; sy += Math.floor(p / width); }
    }
    return {
      changed, width, height,
      frac: changed / (width * height),
      centroid: changed ? { u: sx / changed / width, v: sy / changed / height } : null,
      image: a,
    };
  },
  /** 直接拿一张头摄图（和 agent 喂给 VLM 的完全同源）。 */
  captureHeadCam(opts) { return view.captureHeadCam(opts); },
  /** 头摄图里“内容整体上下移动了多少像素”——验证低头是否真的改变视线。 */
  async headCamShift(aUrl, bUrl, { width = 320, height = 240, search = 40 } = {}) {
    const [ia, ib] = await Promise.all([this.decodeDataUrl(aUrl), this.decodeDataUrl(bUrl)]);
    const ca = document.createElement("canvas"); ca.width = width; ca.height = height;
    const cb = document.createElement("canvas"); cb.width = width; cb.height = height;
    ca.getContext("2d").drawImage(ia, 0, 0); cb.getContext("2d").drawImage(ib, 0, 0);
    const A = ca.getContext("2d").getImageData(0, 0, width, height).data;
    const B = cb.getContext("2d").getImageData(0, 0, width, height).data;
    const row = (buf, y) => { const s = [0, 0, 0]; for (let x = 0; x < width; x++) { const i = (y * width + x) * 4; s[0] += buf[i]; s[1] += buf[i + 1]; s[2] += buf[i + 2]; } return s.map((v) => v / width); };
    const mid = Math.floor(height / 2);
    const ref = row(A, mid);
    let best = 0, bestErr = Infinity;
    for (let d = -search; d <= search; d++) {
      const y = mid + d;
      if (y < 0 || y >= height) continue;
      const r = row(B, y);
      const e = Math.abs(r[0] - ref[0]) + Math.abs(r[1] - ref[1]) + Math.abs(r[2] - ref[2]);
      if (e < bestErr) { bestErr = e; best = d; }
    }
    return { shiftPx: best, error: bestErr };
  },
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
  /** 把一个自由物体（ball / obj_cube_red / …）摆到指定位置，用于构造测试场景和演示。 */
  placeObject(bodyName, pose) { const ok = duck.placeBody(bodyName, pose); view.frame(); return ok; },
  /** 遮挡判断：目标在 VLM 那张图上到底有没有像素。 */
  maskVisible(geoms) { return view.maskVisible(geoms); },
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
