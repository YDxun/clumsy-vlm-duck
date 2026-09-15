/**
 * 步骤① 渲染器验证：在真正的 Chrome（headless）里开页面，做像素级断言。
 *
 * 为什么不用“截图然后人眼看”：渲染错位（矩阵行序、z-up/y-up、相机朝向）看起来
 * 仍然“像一只鸭子”，只有把**画出来的像素**和**物理状态**对齐才能发现。
 * 所以这里做的是 A/B 差分：
 *   同一帧，隐藏「鸭子 81 个 geom」再渲染一次，两张图的差异像素 = 鸭子实际占据的区域；
 *   这个区域的质心必须落在 `camera.project(geom_xpos)` 的投影位置附近。
 *
 * 用法：node tools/verify_render.mjs [--headed] [--keep]
 * 产物：artifacts/step1_*.png（三视角截图）
 */
import { chromium } from "playwright-core";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const ART = path.join(ROOT, "artifacts");
const HEADED = process.argv.includes("--headed");

const CHROME_CANDIDATES = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
];

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`${ok ? "  PASS" : "  FAIL"}  ${name}${detail ? "   " + detail : ""}`);
}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

async function launch() {
  const args = [
    "--enable-unsafe-swiftshader",   // 新版 Chrome 无 GPU 时软件光栅化需要显式打开
    "--use-angle=swiftshader",
    "--no-sandbox",
  ];
  const tries = [];
  if (!process.env.DUCK_BROWSER_PATH) tries.push({ channel: "chrome" });
  tries.push({ executablePath: process.env.DUCK_BROWSER_PATH }, ...CHROME_CANDIDATES.map((p) => ({ executablePath: p })));
  let lastErr;
  for (const t of tries) {
    if (t.executablePath === undefined && !t.channel) continue;
    try {
      const b = await chromium.launch({ ...t, headless: !HEADED, args });
      console.log(`[浏览器] 使用 ${t.channel || t.executablePath}`);
      return b;
    } catch (e) { lastErr = e; }
  }
  throw new Error(`找不到可用 Chrome/Edge（可用 DUCK_BROWSER_PATH 指定）：${lastErr}`);
}

async function main() {
  await mkdir(ART, { recursive: true });
  const site = await startServer({ port: 0 });
  console.log(`[服务] ${site.url}`);
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });

  const problems = [];
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`console: ${m.text()}`); });
  page.on("requestfailed", (r) => problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => { if (r.status() >= 400) problems.push(`http ${r.status()}: ${r.url()}`); });

  console.log("\n== 加载 ===");
  const t0 = Date.now();
  await page.goto(site.url, { waitUntil: "load", timeout: 60000 });
  let ready = true;
  try {
    await page.waitForFunction("window.__duckReady === true", null, { timeout: 120000 });
  } catch (e) {
    ready = false;
    const err = await page.evaluate(() => window.__duckError || "(无 __duckError)").catch(() => "evaluate 失败");
    console.log(`  __duckError = ${err}`);
  }
  check("页面完成启动（__duckReady）", ready, `${((Date.now() - t0) / 1000).toFixed(1)} s 含 20 MB 网格解析`);
  if (!ready) { console.log(problems.join("\n")); await browser.close(); await site.close(); process.exit(1); }

  const logs = await page.evaluate(() => document.getElementById("log").textContent.trim().split("\n"));
  console.log("  —— 页面日志 ——");
  for (const l of logs) console.log("  " + l);
  await page.waitForTimeout(400);

  // ---------------------------------------------------------------- T1 矩阵
  console.log("\n== T1 物理位姿 -> three.js 矩阵（行序 + z-up/y-up 轴向）==");
  const t1 = await page.evaluate(() => { window.__sim.pause(); window.__sim.render(); return window.__sim.matrixCheck(32); });
  check("每个 geom 的 matrixWorld 与 geom_xpos/xmat 一致", t1.max < 1e-6,
        `最大偏差 ${t1.max.toExponential(2)} (采样 ${t1.samples.length} 个 geom)`);

  // ---------------------------------------------------------------- T2 画面
  console.log("\n== T2 画面不是空白/纯色 ==");
  const t2 = await page.evaluate(() => { window.__sim.setMode("overhead"); return window.__sim.grab("all"); });
  check("渲染出足够多的颜色种类", t2.uniqueColors > 100, `uniqueColors=${t2.uniqueColors}`);
  check("前景占比落在合理区间 (5%~98%)",
        t2.foregroundFrac > 0.05 && t2.foregroundFrac < 0.98,
        `foreground=${(t2.foregroundFrac * 100).toFixed(1)}%`);

  // ------------------------------------------------- T3 画出来的鸭子=物理的鸭子
  console.log("\n== T3 A/B 差分：鸭子的像素质心 vs 投影位置 ==");
  const t3 = await page.evaluate(() => {
    const s = window.__sim, duck = s.duckGeoms();
    s.setHidden([]); s.render(); s.grab("all");
    s.setHidden(duck); s.render(); s.grab("no_duck");
    s.setHidden([]); s.render();
    return { diff: s.diff("all", "no_duck"), geomCount: duck.length, anchor: s.duckAnchor() };
  });
  check("鸭子占住了画面（隐藏鸭子会产生差异像素）", t3.diff.changed > 200,
        `${t3.diff.changed} px = 画面 ${(t3.diff.changedFrac * 100).toFixed(2)}%（${t3.geomCount} 个 geom）`);
  const dDuck = dist(t3.diff.changedCentroid, t3.anchor.screen);
  check("鸭子像素质心 == 物理位置投影", dDuck < 0.04 * t2.width,
        `偏差 ${dDuck.toFixed(1)} px（容差 ${(0.04 * t2.width).toFixed(0)} px）`);

  // --------------------------------------------------------- T4 球也在对的位置
  console.log("\n== T4 A/B 差分：球 ==");
  const t4 = await page.evaluate(() => {
    const s = window.__sim, ball = s.ballGeoms();
    s.setHidden(ball); s.render(); s.grab("no_ball");
    s.setHidden([]); s.render();
    s.grab("with_ball");
    return { diff: s.diff("with_ball", "no_ball"), project: s.ballAnchor().screen,
             info: s.geomInfo(ball[0]), pose: { ...s.duck.ballPose } };
  });
  check("球可见且在投影位置上", t4.diff.changed > 30 && dist(t4.diff.changedCentroid, t4.project) < 0.02 * t2.width,
        `${t4.diff.changed} px，偏差 ${dist(t4.diff.changedCentroid, t4.project).toFixed(1)} px，` +
        `球世界坐标 (${t4.pose.x.toFixed(2)}, ${t4.pose.y.toFixed(2)})`);

  // ------------------------------------------------------------- T5 运动闭环
  console.log("\n== T5 走 3 秒：画面里的位移 == 物理位移 ==");
  const t5 = await page.evaluate(async () => {
    const s = window.__sim;
    s.reset();
    s.render(); s.grab("all0");
    s.setHidden(s.duckGeoms()); s.render(); s.grab("noduck0"); s.setHidden([]);
    const c0 = s.diff("all0", "noduck0"), a0 = s.duckAnchor().screen;
    await s.runSteps(150, [0.3, 0, 0]);
    s.render(); s.grab("all1");
    s.setHidden(s.duckGeoms()); s.render(); s.grab("noduck1"); s.setHidden([]); s.render();
    const c1 = s.diff("all1", "noduck1"), a1 = s.duckAnchor().screen;
    return { pose: { ...s.duck.pose }, upright: s.duck.upright(), steps: s.duck.steps,
             c0: { x: c0.changedCentroid.x, y: c0.changedCentroid.y },
             c1: { x: c1.changedCentroid.x, y: c1.changedCentroid.y },
             p0: a0, p1: a1, frameDiff: s.diff("all0", "all1") };
  });
  check("策略没摔（upright > 0.9）", t5.upright > 0.9, `upright=${t5.upright.toFixed(3)}`);
  check("物理上确实前进了", t5.pose.x > 0.3,
        `x: 0 -> ${t5.pose.x.toFixed(3)} m（${t5.steps} 步 @50 Hz）`);
  check("画面整帧发生变化", t5.frameDiff.changedFrac > 0.0003,
        `changed=${(t5.frameDiff.changedFrac * 100).toFixed(3)}%（上帝视角下鸭子只占画面的零点几个百分点）`);
  const shiftPix = t5.c1.x - t5.c0.x, shiftProj = t5.p1.x - t5.p0.x;
  check("画面里鸭子的位移方向与物理一致（向右）", shiftPix > 20,
        `像素位移 ${shiftPix.toFixed(1)} px，投影位移 ${shiftProj.toFixed(1)} px`);
  check("像素位移与投影位移相符（±8 px）", Math.abs(shiftPix - shiftProj) < 8,
        `差 ${(shiftPix - shiftProj).toFixed(1)} px`);

  // ------------------------------------------------------- T6 能不能实时跑
  console.log("\n== T6 控制步吞吐（③ 要交互，必须比 50 Hz 快不少）==");
  const t6 = await page.evaluate(async () => {
    const s = window.__sim;
    s.reset();
    const t0 = performance.now();
    await s.runSteps(200, [0.3, 0, 0]);
    return { ms: performance.now() - t0, steps: 200 };
  });
  const hz = (t6.steps / t6.ms) * 1000;
  check("控制步吞吐 > 100 步/秒（实时闭环的 2 倍余量）", hz > 100,
        `${hz.toFixed(0)} 步/秒 = ${(hz / 50).toFixed(1)}× 实时（${(t6.ms / t6.steps).toFixed(2)} ms/步，含 ONNX 推理）`);

  // ---------------------------------------------------------------- 截图
  console.log("\n== 截图 ==");
  const shots = [
    ["workspace", "全景（场景自带 cam_workspace）"],
    ["overhead", "上帝视角（cam_overhead，正上方全场）"],
    ["follow", "跟随视角（第三人称）"],
    ["duck", "鸭子第一人称（将来喂给 VLM 的就是这个）"],
  ];
  for (const [mode, label] of shots) {
    await page.evaluate((m) => { window.__sim.setMode(m); window.__sim.render(); }, mode);
    await page.waitForTimeout(120);
    const file = path.join(ART, `step1_render_${mode}.png`);
    await page.screenshot({ path: file, clip: await page.evaluate(() => {
      const c = document.getElementById("canvas").getBoundingClientRect();
      return { x: c.x, y: c.y, width: c.width, height: c.height };
    }) });
    console.log(`  ${file}   ← ${label}`);
  }

  console.log("\n== 浏览器报错 ==");
  const real = problems.filter((p) => !/favicon/i.test(p));
  check("无未处理的页面错误", real.length === 0, real.length ? "\n    " + real.join("\n    ") : "");

  const failed = results.filter((r) => !r.ok);
  console.log(`\n结果: ${failed.length ? "FAIL" : "PASS"} —— ${results.length - failed.length}/${results.length} 项通过`);
  if (failed.length) console.log("未通过：" + failed.map((f) => f.name).join("；"));

  await browser.close();
  await site.close();
  process.exit(failed.length ? 1 : 0);
}

main().catch(async (e) => { console.error("异常:", e); process.exit(1); });
