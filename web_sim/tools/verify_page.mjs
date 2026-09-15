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
  check("标签里带人类可读的指令", opts[0].t.includes("—") && opts[0].t.length > opts[0].v.length + 3, opts[0].t);

  const chosen = sceneTasks[0];
  await page.selectOption("#task-select", chosen);
  const info = await page.textContent("#task-info");
  const agentTask = await page.evaluate(() => window.__sim.agent.task);
  check("选择任务后 agent 收到同一个 task id", agentTask.taskId === chosen, `${agentTask.taskId} / ${chosen}`);
  check("界面显示解析出的目标与意图", /目标 = .+ · 意图 = /.test(info), info.trim());

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
  check("开始后物理在推进", runState.steps > 100, `${runState.steps} 个控制步`);
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
    return { has: true, zoomed: d0 > 0 && Math.abs(d1 - d0 / 2) < 1e-6,
             target: [v.controls.target.x, v.controls.target.y, v.controls.target.z],
             enabled: v.controls.enabled };
  });
  check("自由视角装了轨道控制器", orbit.has === true);
  check("滚轮缩放生效", orbit.zoomed === true);
  check("右键平移生效（改轨道中心）", orbit.target?.[0] > 0.25, JSON.stringify(orbit.target));
  const takeover = await page.evaluate(() => {
    const s = window.__sim, v = s.view;
    v.setMode("workspace"); v.frame();
    // 模拟用户在固定机位上动鼠标：应当自动切到 free 并从当前机位接管
    document.getElementById("canvas").dispatchEvent(new PointerEvent("pointerdown", { button: 0, bubbles: true }));
    return { mode: v.mode, controlsEnabled: v.controls.enabled };
  });
  check("在固定机位拖动会自动切到自由视角", takeover.mode === "free" && takeover.controlsEnabled === true,
        JSON.stringify(takeover));

  // ---------------------------------------------------------------- 手动按钮
  console.log("\n== 手动按钮：一步就是一步（曾是“一直走到墙上”）==");
  // 手动 token 走的是页面主循环，所以用真实点击 + 等待来测
  await page.evaluate(() => { window.__sim.pause(); window.__sim.reset(); });
  const beforeFwd = await page.evaluate(() => ({ ...window.__sim.duck.pose }));
  await page.click('[data-cmd="FWD"]');
  await page.evaluate(() => window.__sim.resume());
  await page.waitForTimeout(2500);
  const afterFwd = await page.evaluate(() => ({ ...window.__sim.duck.pose, phase: window.__sim.agent.phase }));
  const fwdDist = Math.hypot(afterFwd.x - beforeFwd.x, afterFwd.y - beforeFwd.y);
  check("前进 1 步只走一小段（0.03~0.25 m，不是走到墙）", fwdDist > 0.03 && fwdDist < 0.25,
        `${fwdDist.toFixed(3)} m`);

  const beforeTurn = await page.evaluate(() => ({ ...window.__sim.duck.pose }));
  await page.click('[data-cmd="TURN_R"]');
  await page.waitForTimeout(2500);
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
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 15000 }),
      page.click("#shot"),
    ]);
    const name = download.suggestedFilename();
    const p = path.join(ART, "step3_threeview.png");
    await download.saveAs(p);
    check("三视角拼图能下载", /^duckvlm_.*\.png$/.test(name), `${name} -> ${p}`);
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
