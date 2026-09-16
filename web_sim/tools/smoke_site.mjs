/**
 * 发布产物冒烟测试 —— 验证 `_site/` 那个**没有 node_modules** 的版本能跑。
 *
 * 为什么必须单独测：开发树里一切从 node_modules 来，发布产物里 three / MuJoCo WASM /
 * onnxruntime-web 全走 CDN。两者的失败模式完全不同（CDN 路径错、wasm 取不到、
 * 跨域被拒……），"本地能跑"完全不能证明"线上能跑"。
 *
 * 用法：
 *   node tools/build_site.mjs --out _site && node tools/smoke_site.mjs _site
 *   node tools/smoke_site.mjs https://huggingface.co/spaces/XenderYang/duck-vlm-simulator   # 直接测线上
 */
import { chromium } from "playwright-core";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const SITE = path.resolve(ROOT, process.argv[2] || "_site");
/** 传 http(s):// 就测线上地址，否则把参数当成本地目录起一个静态服务。 */
const LIVE_URL = /^https?:\/\//.test(process.argv[2] || "") ? process.argv[2] : null;
const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
];

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "  PASS" : "  FAIL"}  ${name}${detail ? "   " + detail : ""}`);
};

async function launch() {
  const args = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-sandbox"];
  for (const t of [{ channel: "chrome" }, ...CHROME.map((p) => ({ executablePath: p }))]) {
    try { return await chromium.launch({ ...t, headless: true, args }); } catch { /* 下一个 */ }
  }
  throw new Error("找不到 Chrome/Edge");
}

async function main() {
  const site = LIVE_URL ? { url: LIVE_URL, close: async () => {} }
                        : await startServer({ port: 0, root: SITE });
  console.log(`[冒烟] ${LIVE_URL ? "线上地址" : "本地站点根目录"} ${LIVE_URL || SITE}`);
  console.log(`[冒烟] ${site.url}`);
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  const problems = [];
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" && !/401|favicon/.test(m.text())) problems.push(m.text().slice(0, 200)); });

  const t0 = Date.now();
  await page.goto(site.url, { waitUntil: "load", timeout: 60000 });
  let ready = true;
  try {
    await page.waitForFunction("window.__duckReady === true", null, { timeout: 240000 });
  } catch {
    ready = false;
    const err = await page.evaluate(() => window.__duckError || "(无 __duckError)").catch(() => "evaluate 失败");
    console.log("  __duckError =", err);
  }
  check("页面在无 node_modules 的产物里启动成功", ready, `${((Date.now() - t0) / 1000).toFixed(1)} s（含 CDN 取 three/MuJoCo/ORT）`);
  if (!ready) {
    console.log(problems.join("\n"));
    await browser.close(); await site.close(); process.exit(1);
  }

  await page.evaluate(() => window.__sim.pause());
  const state = await page.evaluate(() => {
    const s = window.__sim;
    return {
      scenes: s.scene ? 1 : 0,
      sceneId: s.scene?.metadata?.scene_id,
      meshes: s.view?.meshes?.length,
      policies: Object.keys(s.duck?.policies || {}).length,
      allowed: s.agent?.task?.allowed || [],
    };
  });
  check("场景加载完成", state.sceneId === "duck_workspace_v1" && state.meshes > 90,
        `${state.sceneId}，${state.meshes} 个 mesh`);
  check("策略从配置的坐标取到了", state.policies >= 1, `${state.policies} 个`);
  check("动作词表可用", state.allowed.includes("FWD") && state.allowed.includes("LOOK_DOWN"), state.allowed.join(","));

  // 标签页图标：以前是 `data:,` 占位（浏览器会用默认图标），现在应当是站点的 SVG 图标。
  // 顺手验 MIME —— Chrome 对 SVG 图标的 MIME 很挑，给错就当没有。
  const icon = await page.evaluate(async () => {
    const link = document.querySelector('link[rel="icon"]');
    const href = link ? link.getAttribute("href") : null;
    const r = href ? await fetch(href) : null;
    return { href, ok: !!r && r.ok, type: r ? r.headers.get("content-type") : null,
             bytes: r ? (await r.text()).length : 0 };
  });
  check("页面图标已就位（SVG + 正确 MIME）",
        icon.href && icon.href.endsWith("favicon.svg") && icon.ok && /svg/.test(icon.type || "") && icon.bytes > 500,
        `${icon.href} · ${icon.type} · ${icon.bytes} 字节`);

  const run = await page.evaluate(async () => {
    const s = window.__sim;
    s.setTask({ taskId: "walk_to_ball" });
    const task = { ...s.agent.task };
    s.agent.config.mode = "rule"; s.agent.config.maxSteps = 6;
    const out = await s.runAgent({ maxDecisions: 3, maxSteps: 400 });
    return { task, steps: out.steps, decisions: out.decisions, pose: out.pose, upright: out.upright };
  });
  check("精选任务目标来自成功判据（不是推断）", run.task.targetSource === "declared",
        `${run.task.target} / ${run.task.targetSource}`);
  check("闭环能跑（决策 + 物理）", run.decisions > 0 && run.steps > 0,
        `${run.decisions} 次决策 / ${run.steps} 步 / 直立度 ${run.upright.toFixed(2)}`);
  check("产物里没有未处理的报错", problems.length === 0, problems.slice(0, 2).join(" | "));

  await page.screenshot({ path: path.join(ROOT, "artifacts", "site_smoke.png") });
  const failed = results.filter((r) => !r.ok);
  console.log(`\n结果: ${failed.length ? "FAIL" : "PASS"} —— ${results.length - failed.length}/${results.length} 项通过`);
  await browser.close(); await site.close();
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => { console.error("异常:", e); process.exit(1); });
