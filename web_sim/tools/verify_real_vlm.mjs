/**
 * 真模型端到端：让 Qwen-VL（或任何 OpenAI 兼容的 VL 模型）在**浏览器里**
 * 直接控制鸭子，走完整闭环。
 *
 * 这一步和 verify_agent.mjs 的 T4 的区别：T4 用本地假端点验"链路对不对"，
 * 这里用真厂商验"模型到底会不会玩"——提示词好不好、token 选得对不对、
 * 解析会不会失败，只有真模型能回答。
 *
 * key 从环境变量读，不落盘、不进仓库：
 *   $env:DUCK_VLM_KEY="sk-..."
 *   node tools/verify_real_vlm.mjs ["任务文本"] [决策数]
 *
 * 没设 key 就打印 SKIP 并以 0 退出，方便放进 npm run verify 里。
 */
import { chromium } from "playwright-core";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ART = path.resolve(HERE, "../artifacts");
const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
];

const KEY = process.env.DUCK_VLM_KEY || process.env.DASHSCOPE_KEY || "";
const BASE = process.env.DUCK_VLM_BASE_URL || "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
const MODEL = process.env.DUCK_VLM_MODEL || "qwen3-vl-plus";
const TASK = process.argv[2] || "go to the orange ball and stop next to it";
const DECISIONS = Number(process.argv[3] || 8);

if (!KEY) {
  console.log("SKIP —— 没有 DUCK_VLM_KEY / DASHSCOPE_KEY 环境变量，跳过真模型验证。");
  console.log('用法：$env:DUCK_VLM_KEY="sk-..."; node tools/verify_real_vlm.mjs');
  process.exit(0);
}

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`${ok ? "  PASS" : "  FAIL"}  ${name}${detail ? "   " + detail : ""}`);
}

async function launch() {
  const args = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-sandbox"];
  for (const t of [{ channel: "chrome" }, ...CHROME.map((p) => ({ executablePath: p }))]) {
    try { return await chromium.launch({ ...t, headless: true, args }); } catch { /* 下一个 */ }
  }
  throw new Error("找不到可用 Chrome/Edge");
}

async function main() {
  await mkdir(ART, { recursive: true });
  console.log(`[配置] ${BASE} / ${MODEL}`);
  console.log(`[任务] ${TASK}（最多 ${DECISIONS} 次决策）\n`);
  const site = await startServer({ port: 0 });
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  page.on("pageerror", (e) => console.log("  [页面异常]", e.message));

  await page.goto(site.url, { waitUntil: "load" });
  await page.waitForFunction("window.__duckReady === true", null, { timeout: 120000 });
  await page.evaluate(() => window.__sim.pause());

  const setup = await page.evaluate(({ key, base, model, task }) => {
    const s = window.__sim;
    s.reset();
    s.configureAgent({
      mode: "llm", provider: "openai", baseUrl: base, model, apiKey: key,
      temperature: 0, maxTokens: 48, reasoningEffort: "low", decisionEveryS: 0.1, maxSteps: 40,
    });
    const t = s.setTask({ text: task });     // 自由文本，不走精选任务 id
    return { task: t, config: s.agent.config };
  }, { key: KEY, base: BASE, model: MODEL, task: TASK });
  console.log(`[解析] 目标=${setup.task.target} 意图=${setup.task.intent}\n`);

  console.log("== 让模型自己飞 ==");
  const t0 = Date.now();
  const run = await page.evaluate(async (n) => {
    const s = window.__sim;
    return s.runAgent({ maxDecisions: n, maxSteps: 4000 });
  }, DECISIONS);
  const wall = (Date.now() - t0) / 1000;

  const transcript = await page.evaluate(() => window.__sim.agent.records.slice().reverse().map((r) => ({
    i: r.stepIndex, token: r.token, proposed: r.proposed, raw: (r.raw || "").trim().slice(0, 60),
    source: r.source, latency: r.latencyMs, rules: r.rules, note: r.note,
    visible: r.state?.targetVisible, range: r.state?.targetRangeM, bearing: r.state?.targetBearingRad,
    stateText: r.state ? null : null,
  })));

  console.log("步  选择        模型原文                             延迟     目标状态");
  for (const r of transcript) {
    const st = r.visible ? `${r.range?.toFixed(2)}m / ${(r.bearing ?? 0).toFixed(2)}rad` : "看不见";
    const flag = r.proposed !== r.token ? ` ← 规则改成 ${r.token}` : "";
    console.log(`${String(r.i).padStart(2)}  ${r.token.padEnd(11)} ${JSON.stringify(r.raw).padEnd(36)} ` +
                `${String(Math.round(r.latency || 0)).padStart(5)}ms  ${st}${flag}`);
  }

  const vlmRecords = transcript.filter((r) => r.source === "vlm");
  const parseFails = transcript.filter((r) => r.note === "parse-failed");
  const tokens = new Set(transcript.map((r) => r.token));
  const moved = run.pose.x !== 0 || run.pose.y !== 0;
  const ranges = transcript.map((r) => r.range).filter((v) => v != null);
  const firstRange = ranges[0] ?? null;
  const lastRange = ranges[ranges.length - 1] ?? null;
  // 门槛要按模型自己学到的停点来定：它在 0.32~0.33 m 处收工（见 README 的几局记录）。
  // 之前写 0.45 m 就要求它停手是不合理的 —— 0.39 m 对一颗 3.5 cm 的球来说还有 11 个球径远，
  // 继续前进是对的。只有真的贴近了（<0.35 m）却还在猛走，才是问题。
  const arrived = (run.minRange ?? 99) < 0.35;
  const tailTokens = transcript.slice(-2).map((r) => r.token);

  console.log(`\n[结果] ${run.decisions} 次决策 / ${run.steps} 控制步 / 墙钟 ${wall.toFixed(1)} s`);
  console.log(`[结果] 最终位置 (${run.pose.x.toFixed(2)}, ${run.pose.y.toFixed(2)}) 直立度 ${run.upright.toFixed(3)}`);
  console.log(`[结果] 最近距离 ${run.minRange?.toFixed(2) ?? "没看见过"} m  阶段 ${run.phase}`);
  console.log(`[结果] 用到的动作 token：${[...tokens].join(", ")}\n`);

  console.log("== 断言 ==");
  check("真模型给出过决策", vlmRecords.length >= 1, `${vlmRecords.length} 条来自 VLM`);
  check("延迟是真网络量级（>100 ms）", vlmRecords.every((r) => (r.latency || 0) > 100),
        `中位 ${vlmRecords.map((r) => Math.round(r.latency)).sort((a, b) => a - b)[Math.floor(vlmRecords.length / 2)]} ms`);
  check("模型输出基本都能解析成 token", parseFails.length <= Math.ceil(transcript.length * 0.3),
        `${parseFails.length}/${transcript.length} 条解析失败`);
  check("鸭子真的动了", moved, `(${run.pose.x.toFixed(2)}, ${run.pose.y.toFixed(2)})`);
  check("没有摔倒", run.upright > 0.85, `upright=${run.upright.toFixed(3)}`);
  check("距离在缩小（模型在真的靠近）", firstRange != null && lastRange != null && lastRange < firstRange - 0.1,
        `${firstRange?.toFixed(2)} m -> ${lastRange?.toFixed(2)} m`);
  check("靠近过程中没有把球撞飞（最近距离没小于球半径太多）",
        (run.minRange ?? 99) > 0.10, `最近 ${run.minRange?.toFixed(2)} m`);
  if (arrived) {
    check("到了跟前就不再猛走（最后两步出现停/结束类动作）",
          tailTokens.some((t) => ["STOP", "DONE", "LOOK_DOWN", "HEAD_CENTER"].includes(t)),
          `最近 ${run.minRange.toFixed(2)} m，末尾 ${tailTokens.join(" -> ")}`);
  } else {
    console.log(`  （跳过“到点停住”检查：${DECISIONS} 次决策内没走到 0.45 m，最近 ${run.minRange?.toFixed(2)} m）`);
  }
  console.log(`  [记录] 用到的动作 token：${[...tokens].join(", ")}`);

  await page.evaluate(() => {
    const s = window.__sim;
    s.setMode("workspace"); s.render();
  });
  await page.waitForTimeout(200);
  await page.screenshot({ path: path.join(ART, "real_vlm_page.png") });
  const img = await page.evaluate(() => window.__sim.captureHeadCam());
  await writeFile(path.join(ART, "real_vlm_eye.jpg"), Buffer.from(img.split(",")[1], "base64"));
  const log = transcript.map((r) => `#${r.i} ${r.token} raw=${JSON.stringify(r.raw)} ${Math.round(r.latency)}ms ` +
    `visible=${r.visible} range=${r.range} bearing=${r.bearing} rules=${(r.rules || []).join("+")}`).join("\n");
  await writeFile(path.join(ART, "real_vlm_transcript.txt"), `任务: ${TASK}\n模型: ${MODEL}\n\n${log}\n`, "utf8");
  console.log(`\n产物：${path.join(ART, "real_vlm_page.png")}`);
  console.log(`      ${path.join(ART, "real_vlm_transcript.txt")}`);

  const failed = results.filter((r) => !r.ok);
  console.log(`\n结果: ${failed.length ? "FAIL" : "PASS"} —— ${results.length - failed.length}/${results.length} 项通过`);
  await browser.close();
  await site.close();
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => { console.error("异常:", e); process.exit(1); });
