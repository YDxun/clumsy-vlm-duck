/**
 * 步骤③ 页面验证：真 Chrome 里点按钮，而不是只看 DOM 有没有元素。
 *
 * 用法：node tools/verify_page.mjs [--headed]
 */
import { chromium } from "playwright-core";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ART = path.resolve(HERE, "../artifacts");
const HEADED = process.argv.includes("--headed");
const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
];

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`${ok ? "  PASS" : "  FAIL"}  ${name}${detail ? "   " + detail : ""}`);
}

async function launch() {
  const args = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-sandbox"];
  for (const t of [{ channel: "chrome" }, ...CHROME.map((p) => ({ executablePath: p }))]) {
    try { return await chromium.launch({ ...t, headless: !HEADED, args }); } catch { /* 下一个 */ }
  }
  throw new Error("找不到可用 Chrome/Edge");
}

async function main() {
  await mkdir(ART, { recursive: true });
  const site = await startServer({ port: 0 });
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
  const problems = [];
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" && !/favicon|401/.test(m.text())) problems.push(m.text().slice(0, 200)); });

  console.log("[服务]", site.url);
  await page.goto(site.url, { waitUntil: "load" });
  await page.waitForFunction("window.__duckReady === true", null, { timeout: 120000 });
  await page.evaluate(() => window.__sim.pause());
  console.log("== 加载 ==");
  check("页面完成启动", true, `${await page.title()}`);

  // ---------------------------------------------------------------- 任务下拉
  console.log("\n== 任务下拉：直接列场景包的 task id ==");
  const opts = await page.$$eval("#task-select option", (els) => els.map((e) => ({ v: e.value, t: e.textContent })));
  const sceneTasks = await page.evaluate(() => window.__sim.scene.listTasks().map((t) => t.id));
  check("下拉项数 = 场景任务数 + 自由输入", opts.length === sceneTasks.length + 1,
        `${opts.length} 项（场景 ${sceneTasks.length} 条）`);
  check("每一项都是场景里的真实 task id",
        sceneTasks.every((id) => opts.some((o) => o.v === id)), sceneTasks.join(", "));
  // 选项标签 =「难度标记 · 中文指令」（以前是「id — 指令」）
  const firstSceneTask = await page.evaluate(() => window.__sim.scene.listTasks()[0]);
  check("标签里带人类可读的指令",
        opts[0].t.includes(firstSceneTask.instruction_zh || firstSceneTask.instruction_en) &&
        opts[0].t.length > opts[0].v.length + 3, opts[0].t);

  // ---------------------------------------------------------------- 多场景
  console.log("\n== 多场景：清单驱动 + 热切换（不重启服务）==");
  const sceneOpts = await page.$$eval("#scene-select option", (e) => e.map((x) => ({ v: x.value, t: x.textContent })));
  check("场景下拉列出了清单里的全部场景", sceneOpts.length >= 3, sceneOpts.map((o) => o.t).join(" | "));
  const switched = await page.evaluate(async () => {
    const t0 = performance.now();
    const before = window.__sim.scene.metadata.scene_id;
    document.getElementById("scene-select").value = "duck_home_v1";
    document.getElementById("scene-select").dispatchEvent(new Event("change"));
    for (let i = 0; i < 600; i++) {
      await new Promise((r) => setTimeout(r, 100));
      if (window.__sim.scene.metadata.scene_id === "duck_home_v1") break;
    }
    const s = window.__sim;
    return { before, after: s.scene.metadata.scene_id, ms: performance.now() - t0,
             geoms: s.duck.model.ngeom, meshes: s.view.meshes.length,
             tasks: s.scene.listTasks().map((t) => t.id),
             info: document.getElementById("scene-info").textContent,
             stillReady: window.__duckReady };
  });
  check("换到第二个场景成功（页面没刷新）", switched.after === "duck_home_v1" && switched.stillReady === true,
        `${switched.before} -> ${switched.after}，${(switched.ms / 1000).toFixed(1)} s`);
  check("渲染跟着换成新场景的几何", switched.meshes === switched.geoms && switched.geoms > 0,
        `${switched.meshes} 个 mesh / ${switched.geoms} 个 geom`);
  check("任务表换成了新场景的", switched.tasks.length >= 3 && switched.tasks.includes("search_ball_across_rooms"),
        switched.tasks.join(", "));
  check("场景信息栏显示标题与描述",
        switched.info.includes("家庭") && switched.info.includes(`任务 ${switched.tasks.length} 条`),
        switched.info.slice(0, 70));
  const cachedSwitch = await page.evaluate(async () => {
    const t0 = performance.now();
    document.getElementById("scene-select").value = "duck_workspace_v1";
    document.getElementById("scene-select").dispatchEvent(new Event("change"));
    for (let i = 0; i < 300; i++) {
      await new Promise((r) => setTimeout(r, 50));
      if (window.__sim.scene.metadata.scene_id === "duck_workspace_v1") break;
    }
    return { ms: performance.now() - t0, scene: window.__sim.scene.metadata.scene_id };
  });
  check("换回已加载过的场景走缓存（秒级）", cachedSwitch.ms < 3000,
        `${(cachedSwitch.ms / 1000).toFixed(2)} s`);

  // ---------------------------------------------------------------- 配色
  console.log("\n== 鸭子配色 ==");
  const pal = await page.evaluate(() => {
    const v = window.__sim.view;
    const accents = v.meshes.filter((m) => m.accent);
    const colors = (mode) => {
      v.applyPalette(mode);
      return [...new Set(accents.map((m) => m.mesh.material.color.getHexString()))].sort();
    };
    const duck = colors("duck"), raw = colors("raw");
    v.applyPalette("duck"); v.frame();
    return { count: accents.length, duck, raw, palette: v.palette };
  });
  check("鸭子零件被分配了配色", pal.count > 50, `${pal.count} 个 geom 有专门颜色`);
  check("美化配色不是单一灰（嘴/腿/脚/身各不相同）", pal.duck.length >= 4, pal.duck.join(", "));
  check("能一键切回场景包原始材质", pal.raw.length === 1, pal.raw.join(", "));

  // 挑一个**有目标物体**的精选任务：新加的简单任务里有姿态类的（本来就没有目标），
  // 那些的 targetSource 是 'inferred-no-object'，不该拿来验"来自成功判据"。
  const chosen = "walk_to_ball";      // 此刻是工作台场景；它是有目标物体的精选任务
  await page.selectOption("#task-select", chosen);
  const info = await page.textContent("#task-info");
  const agentTask = await page.evaluate(() => window.__sim.agent.task);
  check("选择任务后 agent 收到同一个 task id", agentTask.taskId === chosen, `${agentTask.taskId} / ${chosen}`);
  // 关键是"目标从哪来"要显示出来：精选任务必须显示"来自成功判据"，
  // 只有自由文本才允许是"从措辞推断"。见 tools/check_task_schema.mjs 的由来。
  check("界面显示目标的来源（精选任务 = 来自成功判据）",
        /目标 = \S+（来自成功判据）/.test(info), info.trim());
  // 换一个"要把东西送到某处"的任务：它才带区域，而且半径应该取**成功判据里的 0.30**，
  // 不是场景元数据里的 0.35（差那 5 cm 会让"自认为成功"和评测不一致）。
  await page.selectOption("#task-select", "kick_ball_to_zone");
  const zoneInfo = await page.textContent("#task-info");
  check("带区域的任务显示区域名与判定半径（判据 0.30，不是元数据 0.35）",
        /区域 = zone_green@0\.30m/.test(zoneInfo), zoneInfo.trim());

  // ---------------------------------------------------------------- 自由文本
  console.log("\n== 自由文本：去红方块 -> obj_cube_red ==");
  await page.selectOption("#task-select", "__free__");
  await page.fill("#task-text", "去红方块");
  await page.dispatchEvent("#task-text", "change");
  const free = await page.evaluate(() => window.__sim.agent.task);
  check("自由文本推出目标", free.target === "obj_cube_red", `${free.target}`);
  check("自由文本推出意图", free.intent === "approach", free.intent);
  check("界面同步显示", (await page.textContent("#s-target")) === "obj_cube_red");

  const kick = await page.evaluate(() => {
    window.__sim.agent.setTask({ text: "把球踢进绿色区域" });
    return window.__sim.agent.task;
  });
  check("操作类句子盯的是球而不是区域", kick.target === "ball" && kick.intent === "manipulate",
        `${kick.target} / ${kick.intent}`);

  // 操作方式（推/踢）：自动按语义判定，也能手动指定
  const modes = await page.$$eval("#manip-mode option", (e) => e.map((x) => x.value));
  check("界面上有操作方式选择", ["auto", "push", "kick"].every((m) => modes.includes(m)), modes.join(","));
  const autoPick = await page.evaluate(() => {
    const a = window.__sim.agent;
    a.manipulationMode = "auto";
    const pushTask = a.setTask({ taskId: "push_red_cube" }).taskId && a.manipMode;
    const kickTask = a.setTask({ taskId: "kick_ball_to_zone" }).taskId && a.manipMode;
    return { pushTask, kickTask };
  });
  check("自动模式按任务语义选推/踢", autoPick.pushTask === "push" && autoPick.kickTask === "kick",
        JSON.stringify(autoPick));
  await page.selectOption("#manip-mode", "kick");
  const forced = await page.evaluate(() => window.__sim.agent.manipMode);
  check("手动指定操作方式生效", forced === "kick", forced);
  await page.selectOption("#manip-mode", "auto");

  // ---------------------------------------------------------------- BYO key
  console.log("\n== BYO key：只存本机，切到 VLM 模式才出现 ==");
  check("key 输入框默认隐藏", await page.isHidden("#llm-box"));
  await page.click('[data-decmode="llm"]');
  check("切到 VLM 模式后出现配置区", await page.isVisible("#llm-box"));
  check("key 用 password 类型（不明文显示）",
        (await page.getAttribute("#llm-key", "type")) === "password");
  await page.fill("#llm-key", "sk-unit-test-not-a-real-key");
  await page.dispatchEvent("#llm-key", "change");
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("duckvlm.config.v1") || "{}"));
  check("key 写进本机 localStorage", stored.key === "sk-unit-test-not-a-real-key", Object.keys(stored).join(","));
  const provOpts = await page.$$eval("#llm-provider option", (e) => e.map((x) => x.value));
  check("三家厂商都列出来了", ["openai", "gemini", "anthropic"].every((p) => provOpts.includes(p)), provOpts.join(", "));
  const baseBefore = await page.inputValue("#llm-base");
  await page.selectOption("#llm-provider", "gemini");
  const baseAfter = await page.inputValue("#llm-base");
  check("换厂商会带出对应默认 base URL", baseBefore !== baseAfter && baseAfter.includes("generativelanguage"),
        baseAfter);

  // 「只填一个 key 就能用」—— 模型是下拉且永远有值，地址自动带出
  console.log("\n== 只填一个 key 就能用（模型/地址自动带出）==");
  await page.selectOption("#llm-provider", "openai");
  const modelOpts = await page.$$eval("#llm-model option", (e) => e.map((x) => x.value));
  check("模型是个下拉，列表里有 qwen3-vl-plus", modelOpts.includes("qwen3-vl-plus"), modelOpts.join(", "));
  const modelNow = await page.inputValue("#llm-model");
  check("选 Qwen 后模型自动带出、不为空（空的会被服务端 400 掉）",
        modelNow === "qwen3-vl-plus", JSON.stringify(modelNow));
  const baseNow = await page.inputValue("#llm-base");
  check("选 Qwen 后地址自动带出新加坡区", baseNow.includes("dashscope-intl"), baseNow);
  check("高级区默认折叠（新手只看得到下拉 + key 两个框）",
        !(await page.evaluate(() => document.querySelector("#llm-base").closest("details").open)));
  check("高级区标题不写多余说明",
        (await page.textContent("#llm-box summary")).trim() === "高级：换模型 / 换地址",
        (await page.textContent("#llm-box summary")).trim());
  check("base URL / 模型名不写进 localStorage（旧值盖掉新默认值的坑）",
        !("baseUrl" in stored) && !("model" in stored), Object.keys(stored).join(","));

  // 测试按钮：真发请求，报错要翻译成人话。指向没人监听的本地端口，不依赖外网。
  const problemsBeforeTest = problems.length;
  await page.evaluate(() => { document.querySelector("#llm-base").closest("details").open = true; });
  await page.fill("#llm-base", "http://127.0.0.1:45999/v1");
  await page.dispatchEvent("#llm-base", "change");
  await page.click("#llm-test");
  await page.waitForFunction(
    () => /❌|✅|先在上面|模型名是空的/.test(document.getElementById("llm-test-result").textContent),
    null, { timeout: 20000 });
  const testMsg = (await page.textContent("#llm-test-result")).trim();
  check("点「测试一下能不能用」会真发请求，且失败信息是人话",
        testMsg.startsWith("❌") && !/Failed to fetch|TypeError|HTTP \d/.test(testMsg), testMsg);
  // 这一步是故意打不通的，浏览器必然记一条「Failed to load resource」；
  // 只豁免这段时间里这一类噪声，其它照旧上报。
  const deliberateNoise = problems.splice(problemsBeforeTest);
  problems.push(...deliberateNoise.filter((p) => !/Failed to load resource/.test(p)));
  await page.selectOption("#llm-provider", "openai");   // 把地址恢复到默认，别影响后面的步骤

  await page.reload({ waitUntil: "load" });
  await page.waitForFunction("window.__duckReady === true", null, { timeout: 120000 });
  await page.evaluate(() => window.__sim.pause());
  check("刷新后 key 还在（本机持久化）",
        (await page.inputValue("#llm-key")) === "sk-unit-test-not-a-real-key");

  // ---------------------------------------------------------------- 跑闭环
  console.log("\n== 点“开始”跑规则模式闭环 ==");
  await page.click('[data-decmode="rule"]');
  await page.selectOption("#task-select", "walk_to_ball");
  await page.click("#run");
  await page.waitForTimeout(9000);
  const runState = await page.evaluate(() => {
    const s = window.__sim, a = s.agent;
    return {
      steps: s.duck.steps, phase: a.phase, decisions: a.records.length,
      minRange: a.minRange, pose: { ...s.duck.pose }, upright: s.duck.upright(),
      cards: document.querySelectorAll("#decisions .card").length,
      cardsWithImage: document.querySelectorAll("#decisions .card img").length,
      hudSteps: document.getElementById("hud-steps").textContent,
      sDecisions: document.getElementById("s-decisions").textContent,
      amp: s.agent.task,
    };
  });
  // 无头环境是软件光栅化，9 秒墙钟跑到多少步取决于机器忙不忙，阈值放宽
  check("开始后物理在推进", runState.steps > 50, `${runState.steps} 个控制步`);
  check("决策层在工作", runState.decisions > 0, `${runState.decisions} 次决策，阶段 ${runState.phase}`);
  check("鸭子没有摔倒", runState.upright > 0.9, `upright=${runState.upright.toFixed(3)}`);
  check("日志出现决策卡片", runState.cards > 0, `${runState.cards} 张`);
  check("每张卡片带模型当时看到的图", runState.cardsWithImage === runState.cards,
        `${runState.cardsWithImage}/${runState.cards}`);
  check("HUD 与状态面板同步刷新",
        runState.hudSteps === String(runState.steps) && runState.sDecisions === String(runState.decisions),
        `hud=${runState.hudSteps} 面板=${runState.sDecisions}`);
  check("规则模式真的在朝球走", runState.minRange !== null && runState.minRange < 1.0,
        `最近 ${runState.minRange?.toFixed(2)} m`);

  // ---------------------------------------------------------------- 换指令再跑
  // 用户报的 bug：跑一次 → 换句指令 → 再点开始，鸭子不动、也不再推理（阶段一直停在
  // 上一次的状态）。根因是 _pending 这道"同一时刻只允许一次决策"的闸门，在请求还在飞
  // 的时候按了停止/复位就再也没人消费它。这里把那条路径整个走一遍。
  console.log("\n== 换指令再点开始（决策闸门不能被上一轮焊死）==");
  const reuse = await page.evaluate(async () => {
    const s = window.__sim, a = s.agent;
    const waitFrame = () => new Promise((r) => requestAnimationFrame(r));
    s.pause();
    document.getElementById("stop").click();

    // 1) 请求正在飞的时候按「停止决策」：闸门必须被清掉
    a._pending = { settled: false, value: null };
    a.phase = "thinking";
    document.getElementById("stop").click();
    const afterStop = { pending: !!a._pending, phase: a.phase };

    // 2) 就算闸门里真的卡了一个迟到的结果，换指令再点开始也得能重新推理
    a._pending = { settled: true, value: { token: "FWD", note: "late", error: null, stepIndex: 0 } };
    a.phase = "finished";
    document.querySelector('[data-decmode="rule"]').click();
    s.setTask({ taskId: "walk_to_ball" });               // 换任务（点「开始」走的也是这条）
    document.getElementById("run").click();
    const before = a.records.length;
    let last = { phase: a.phase, decisions: a.records.length, pending: !!a._pending };
    const t0 = performance.now();
    while (performance.now() - t0 < 8000) {
      await waitFrame();
      last = { phase: a.phase, decisions: a.records.length, pending: !!a._pending };
      if (a.records.length > before) break;
    }
    document.getElementById("stop").click();
    return { afterStop, before, afterRestart: last };
  });
  check("请求在飞时按「停止决策」，决策闸门就被放开了",
        reuse.afterStop.pending === false, JSON.stringify(reuse.afterStop));
  check("闸门里卡着上一轮结果时，换指令再点开始照样能推理",
        reuse.afterRestart.decisions > reuse.before, JSON.stringify(reuse.afterRestart));

  // ---------------------------------------------------------------- 动作序列
  // 用户报的另一种情况：指令是「前进1米，再翻滚一次，最后跳舞」这种**没有目标物体**的
  // 编排。以前会被当成"去找 ball"，鸭子只转圈找球，翻滚跳舞永远轮不上。
  console.log("\n== 动作序列：「前进1米，再翻滚一次，最后跳舞」==");
  const seqRun = await page.evaluate(async () => {
    const s = window.__sim, a = s.agent;
    s.pause(); s.setDriver(false);
    document.getElementById("stop").click();
    document.getElementById("task-select").value = "__free__";
    const inp = document.getElementById("task-text");
    inp.value = "前进1米，再翻滚一次，最后跳舞";
    inp.dispatchEvent(new Event("change"));
    const parsed = { target: a.task.target, intent: a.task.intent, seq: a.task.sequence,
                     steps: a.sequencer.steps.map((x) => x.label) };
    const start = { ...s.duck.pose };
    document.getElementById("run").click();
    const t0 = performance.now();
    let last = null;
    while (performance.now() - t0 < 45000) {
      await new Promise((r) => requestAnimationFrame(r));
      last = { phase: a.phase, progress: a.sequenceProgress, tokens: a.history.slice(0, 6) };
      if (a.phase === "finished") break;
    }
    document.getElementById("stop").click();
    const end = { ...s.duck.pose };
    const out = { parsed, last, moved: Math.hypot(end.x - start.x, end.y - start.y),
                  endUpright: s.duck.upright() };
    s.reset();
    return out;
  });
  check("纯动作指令被识别成序列，不再瞎猜目标",
        seqRun.parsed.intent === "sequence" && seqRun.parsed.target === "",
        JSON.stringify(seqRun.parsed));
  check("拆成 前进 / 翻滚 / 跳舞（已停用）三步",
        seqRun.parsed.steps.length === 3 && seqRun.parsed.steps[0] === "前进 1 m" &&
        seqRun.parsed.steps[1] === "翻滚" && /跳舞/.test(seqRun.parsed.steps[2]),
        JSON.stringify(seqRun.parsed.steps));
  check("序列能跑完（不被当成找球任务卡住）", seqRun.last?.phase === "finished",
        `${seqRun.last?.phase}｜${seqRun.last?.progress}`);
  check("真的走了一米，而不是原地做动作", seqRun.moved > 0.9, `${seqRun.moved.toFixed(2)} m`);
  // 「跳舞」用的 happy_hop 实测会把鸭子放倒，所以现在是"跳过并说明原因"，
  // 而不是真的执行 —— 同时序列收尾要确认鸭子是站着的。
  check("翻滚执行了，跳舞被跳过且说了原因",
        (seqRun.last?.tokens || []).includes("ROLL") &&
        !(seqRun.last?.tokens || []).includes("DANCE") &&
        /停用/.test(seqRun.parsed.steps.join("|")),
        `${(seqRun.last?.tokens || []).join(",")}｜${seqRun.parsed.steps.join(" / ")}`);
  check("序列结束时鸭子是站着的（躺下会用翻滚翻回来）",
        seqRun.endUpright > 0.85, `upright=${seqRun.endUpright}`);

  // ---------------------------------------------------------------- 视角
  console.log("\n== 视角与三视角拼图 ==");
  for (const m of ["workspace", "overhead", "duck", "follow", "free"]) {
    await page.click(`[data-mode="${m}"]`);
    const mode = await page.evaluate(() => window.__sim.view.mode);
    check(`切到「${m}」`, mode === m, mode);
  }
  const orbit = await page.evaluate(() => {
    const v = window.__sim.view;
    if (!v.controls) return { has: false };
    const d0 = v.camera.position.distanceTo(v.controls.target);
    v.camera.position.sub(v.controls.target).multiplyScalar(0.5).add(v.controls.target);  // 等价于滚轮缩放
    v.controls.update(); v.frame();
    const d1 = v.camera.position.distanceTo(v.controls.target);
    v.controls.target.add({ x: 0.3, y: 0, z: 0.2 }); v.controls.update(); v.frame();  // 等价于平移
    // 不比较精确折半：OrbitControls 带阻尼，update() 会按当前状态微调半径
    return { has: true, d0, d1, zoomed: d1 < d0 * 0.9,
             target: [v.controls.target.x, v.controls.target.y, v.controls.target.z],
             enabled: v.controls.enabled };
  });
  check("自由视角装了轨道控制器", orbit.has === true);
  check("滚轮缩放生效（相机到轨道中心的距离变近）", orbit.zoomed === true,
        `${orbit.d0?.toFixed(2)} -> ${orbit.d1?.toFixed(2)}`);
  check("右键平移生效（改轨道中心）", orbit.target?.[0] > 0.25, JSON.stringify(orbit.target));
  const takeover = await page.evaluate(() => {
    const s = window.__sim, v = s.view;
    const c = document.getElementById("canvas");
    const ev = (type, x, y) => c.dispatchEvent(new PointerEvent(type, {
      button: 0, buttons: 1, clientX: x, clientY: y, bubbles: true,
    }));
    window.__sim.setMode("workspace"); s.view.frame();
    // 1) 只是单击（按下就抬起，没有位移）：不该动视角
    ev("pointerdown", 300, 300); ev("pointerup", 300, 300);
    const afterClick = { mode: v.mode, on: [...document.querySelectorAll("#viewbar [data-mode].on")]
      .map((b) => b.dataset.mode).join(",") };
    // 2) 滚轮：也不该动视角（HF Space 里滚动外层页面就会把 wheel 甩到 canvas 上）
    c.dispatchEvent(new WheelEvent("wheel", { deltaY: -120, bubbles: true }));
    const afterWheel = v.mode;
    // 3) 真的拖动：这时才自动接管成自由视角，且按钮高亮要同步过去
    ev("pointerdown", 300, 300); ev("pointermove", 340, 320);
    const afterDrag = { mode: v.mode, controlsEnabled: v.controls.enabled,
      on: [...document.querySelectorAll("#viewbar [data-mode].on")].map((b) => b.dataset.mode).join(",") };
    return { afterClick, afterWheel, afterDrag };
  });
  check("固定视角下单击画布不会切走视角",
        takeover.afterClick.mode === "workspace" && takeover.afterClick.on === "workspace",
        JSON.stringify(takeover.afterClick));
  check("固定视角下滚轮不会切走视角（页面滚动不再偷换视角）",
        takeover.afterWheel === "workspace", takeover.afterWheel);
  check("固定视角下真的拖动才接管，且按钮高亮同步",
        takeover.afterDrag.mode === "free" && takeover.afterDrag.controlsEnabled === true &&
        takeover.afterDrag.on === "free", JSON.stringify(takeover.afterDrag));

  // 决策运行不该动视角：这是用户反馈的 bug，锁进测试里
  const viewDuringRun = await page.evaluate(async () => {
    const s = window.__sim;
    const read = () => ({
      mode: s.view.mode,
      on: [...document.querySelectorAll("#viewbar [data-mode].on")].map((b) => b.dataset.mode).join(","),
    });
    s.setMode("overhead");
    document.getElementById("run").click();          // 走真正的按钮路径
    const before = read();
    let changed = null;
    const t0 = performance.now();
    while (performance.now() - t0 < 6000) {           // 跑几个控制步，看会不会被改
      await new Promise((r) => requestAnimationFrame(r));
      const now = read();
      if (now.mode !== before.mode || now.on !== before.on) { changed = now; break; }
    }
    document.getElementById("stop").click();
    s.pause(); s.reset();                             // 收尾：别把鸭子留在半路影响后面的用例
    return { before, changed: changed || read() };
  });
  check("点「开始」跑决策不会改变当前视角",
        viewDuringRun.before.mode === viewDuringRun.changed.mode &&
        viewDuringRun.before.on === viewDuringRun.changed.on,
        `${JSON.stringify(viewDuringRun.before)} -> ${JSON.stringify(viewDuringRun.changed)}`);

  // ---------------------------------------------------------------- 手动按钮
  console.log("\n== 技能 token ==");
  const menu = await page.evaluate(() => {
    const s = window.__sim;
    return { allowed: s.agent.task.allowed, names: Object.keys(s.duck.policies || {}),
             skillButtons: [...document.querySelectorAll("#skill-bar [data-cmd]")].map((b) => b.dataset.cmd) };
  });
  check("页面把 9 个策略都装进来了", menu.names.length >= 9, menu.names.join(", "));
  // 跳舞用的 happy_hop 会把鸭子放倒 → 已经降级成"未验证"，不再发给模型
  check("实测有效的技能进了 VLM 词表（翻滚在、跳舞已撤下）",
        menu.allowed.includes("ROLL") && !menu.allowed.includes("DANCE"),
        menu.allowed.join(","));
  check("未验证的技能没有发给模型", !menu.allowed.includes("KICK_R") && !menu.allowed.includes("STAND_UP"),
        menu.allowed.filter((t) => ["KICK_L", "KICK_R", "SIT", "STAND_UP"].includes(t)).join(",") || "（一个都没有，正确）");
  check("技能按钮都摆在面板上", ["ROLL", "DANCE", "KICK_R", "SIT", "STAND_UP"].every((t) => menu.skillButtons.includes(t)),
        menu.skillButtons.join(","));

  const roll = await page.evaluate(async () => {
    const s = window.__sim, duck = s.duck;
    s.pause(); s.reset();
    const it = s.agent.interpreter;
    it.policySwitch = s.agent.interpreter.policySwitch;   // 保持页面里那份接线
    const started = it.start("ROLL");
    const trace = [];
    let minUp = 9, switched = null;
    for (let i = 0; i < 200; i++) {
      const out = it.tick(0.02);
      if (!switched && duck.policyName === "roulade") switched = "roulade";
      await duck.stepAsync(out.command, it.headOverride());
      minUp = Math.min(minUp, duck.upright());
      if (i % 20 === 0) trace.push(+duck.upright().toFixed(2));
      if (out.done) break;
    }
    return { started: !started.done, minUp: +minUp.toFixed(2), endUp: +duck.upright().toFixed(2),
             policyAfter: duck.policyName, trace };
  });
  check("ROLL 真的翻过去了（upright 变负）", roll.minUp < -0.5, `最低 upright=${roll.minUp}，轨迹 ${roll.trace.join(" ")}`);
  check("翻完自己站回来", roll.endUp > 0.9, `末态 upright=${roll.endUp}`);
  check("技能结束后策略切回走路", roll.policyAfter === "alpha_walking", roll.policyAfter);

  console.log("\n== 手动按钮：一步就是一步（曾是“一直走到墙上”）==");
  // 手动 token 走的是页面主循环，所以用真实点击 + 等待来测
  await page.evaluate(() => { window.__sim.pause(); window.__sim.reset(); });
  const beforeFwd = await page.evaluate(() => ({ ...window.__sim.duck.pose }));
  await page.click('[data-cmd="FWD"]');
  await page.evaluate(() => window.__sim.resume());
  // 等这一 token 真的跑完，而不是等一个固定墙钟 —— 无头环境里物理循环快慢差别很大
  await page.waitForFunction(() => window.__sim.manual && !window.__sim.manual.busy, null, { timeout: 60000 });
  const afterFwd = await page.evaluate(() => ({ ...window.__sim.duck.pose, phase: window.__sim.agent.phase }));
  const fwdDist = Math.hypot(afterFwd.x - beforeFwd.x, afterFwd.y - beforeFwd.y);
  check("前进 1 步只走一小段（0.03~0.25 m，不是走到墙）", fwdDist > 0.03 && fwdDist < 0.25,
        `${fwdDist.toFixed(3)} m`);

  const beforeTurn = await page.evaluate(() => ({ ...window.__sim.duck.pose }));
  await page.click('[data-cmd="TURN_R"]');
  await page.waitForFunction(() => window.__sim.manual && !window.__sim.manual.busy, null, { timeout: 60000 });
  const afterTurn = await page.evaluate(() => ({ ...window.__sim.duck.pose }));
  let dHead = afterTurn.heading - beforeTurn.heading;
  while (dHead > Math.PI) dHead -= 2 * Math.PI;
  while (dHead < -Math.PI) dHead += 2 * Math.PI;
  check("右转 1 步真的转了（> 15°）", Math.abs(dHead) > 15 * Math.PI / 180,
        `${(dHead * 180 / Math.PI).toFixed(1)}°（以前下发 0.8 rad/s 只有 8% 达成率，几乎不动）`);

  const floor = await page.evaluate(async () => {
    const s = window.__sim, duck = s.duck;
    // 亚地板指令会被抬到地板：0.8 -> 1.15 才有实际效果
    const before = { ...duck.pose };
    s.reset();
    duck.lastTwist = null;
    await duck.stepAsync([0, 0, 0.8]);
    const normalized = duck.lastTwist.slice();
    return { normalized, before };
  });
  check("低于步态地板的指令会被抬到地板", Math.abs(floor.normalized[2] - 1.15) < 1e-6,
        `下发 wz=0.80 -> 实际 ${floor.normalized[2]}`);
  try {
    await page.click('[data-mode="overhead"]');        // 先明确选一个视角，看拍完会不会被留在「鸭子眼」
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 15000 }),
      page.click("#shot"),
    ]);
    const name = download.suggestedFilename();
    const p = path.join(ART, "step3_threeview.png");
    await download.saveAs(p);
    check("三视角拼图能下载", /^duckvlm_.*\.png$/.test(name), `${name} -> ${p}`);
    const afterShot = await page.evaluate(() => ({
      mode: window.__sim.view.mode,
      on: [...document.querySelectorAll("#viewbar [data-mode].on")].map((b) => b.dataset.mode).join(","),
    }));
    check("三视角拼图拍完会回到用户原来选的视角",
          afterShot.mode === "overhead" && afterShot.on === "overhead", JSON.stringify(afterShot));
  } catch (e) {
    check("三视角拼图能下载", false, String(e.message).slice(0, 120));
  }

  await page.screenshot({ path: path.join(ART, "step3_page.png"), fullPage: false });
  console.log(`  ${path.join(ART, "step3_page.png")}   ← 整页截图`);

  console.log("\n== 浏览器报错 ==");
  check("无未处理的页面错误", problems.length === 0, problems.join(" | "));

  const failed = results.filter((r) => !r.ok);
  console.log(`\n结果: ${failed.length ? "FAIL" : "PASS"} —— ${results.length - failed.length}/${results.length} 项通过`);
  if (failed.length) console.log("未通过：" + failed.map((f) => f.name).join("；"));
  await browser.close();
  await site.close();
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => { console.error("异常:", e); process.exit(1); });
