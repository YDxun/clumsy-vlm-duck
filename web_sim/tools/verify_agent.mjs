/**
 * 步骤② 决策层验证：在真 Chrome 里跑完整闭环。
 *
 * 最重要的两条：
 *   - 「看不见就不给距离」不是嘴上说说：把目标挡住/转开，状态里就必须是 not_found；
 *   - 近距死锁真的解开了：球近到掉出画面下方 → LOOK_DOWN → 球重新回到画面里。
 *
 * 用法：node tools/verify_agent.mjs [--headed]
 */
import { chromium } from "playwright-core";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";
import { startMockLlm } from "./mock_llm.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const ART = path.join(ROOT, "artifacts");
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
    try { return await chromium.launch({ ...t, headless: !HEADED, args }); } catch { /* 换下一个 */ }
  }
  throw new Error("找不到可用 Chrome/Edge");
}

async function main() {
  await mkdir(ART, { recursive: true });
  const site = await startServer({ port: 0 });
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 760 }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.log("  [页面异常]", e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/favicon/i.test(m.text())) console.log("  [console]", m.text().slice(0, 200)); });

  console.log("[服务]", site.url);
  await page.goto(site.url, { waitUntil: "load" });
  await page.waitForFunction("window.__duckReady === true", null, { timeout: 120000 });
  await page.evaluate(() => window.__sim.pause());

  // ============================================================ T0 头部符号
  console.log("\n== T0 头部朝向：低头就是低头（回归测试：曾把旋转矩阵的行当列用）==");
  const t0 = await page.evaluate(async () => {
    const s = window.__sim, duck = s.duck;
    const probe = async (delta) => {
      s.placeDuck({ x: 0, y: 0, yaw: 0 });
      for (let i = 0; i < 70; i++) await duck.stepAsync([0, 0, 0], delta);
      s.render();
      const b = duck.headCamBasis();
      return { depressionDeg: Math.asin(-b.forward[2]) * 180 / Math.PI, pos: b.pos };
    };
    const nominal = await probe(null);
    const down = await probe({ head_pitch: 0.6, neck_pitch: 0, head_yaw: 0, head_roll: 0 });
    const up = await probe({ head_pitch: -0.6, neck_pitch: 0, head_yaw: 0, head_roll: 0 });
    const neck = await probe({ neck_pitch: 0.6, head_pitch: 0, head_yaw: 0, head_roll: 0 });
    return { nominal, down, up, neck };
  });
  check("默认姿态是略微俯视（不是抬头看天）", t0.nominal.depressionDeg > 5 && t0.nominal.depressionDeg < 40,
        `俯角 ${t0.nominal.depressionDeg.toFixed(1)}°`);
  check("head_pitch +0.60 = 低头（俯角变大）", t0.down.depressionDeg > t0.nominal.depressionDeg + 20,
        `${t0.nominal.depressionDeg.toFixed(1)}° -> ${t0.down.depressionDeg.toFixed(1)}°`);
  check("head_pitch -0.60 = 抬头（俯角变小）", t0.up.depressionDeg < t0.nominal.depressionDeg - 20,
        `${t0.nominal.depressionDeg.toFixed(1)}° -> ${t0.up.depressionDeg.toFixed(1)}°`);
  check("neck_pitch +0.60 = 抬头（与 Python 端注释一致）", t0.neck.depressionDeg < t0.nominal.depressionDeg,
        `${t0.nominal.depressionDeg.toFixed(1)}° -> ${t0.neck.depressionDeg.toFixed(1)}°`);

  // ============================================================ T1 状态 vs 画面
  console.log("\n== T1 「看不见就不给距离」：状态传感器与头摄画面必须一致 ==");
  const t1 = await page.evaluate(async () => {
    const s = window.__sim, agent = s.agent;
    const out = {};
    // 先把上一节的头部姿态清掉：placeDuck 只动自由关节，不动头/腿
    s.reset();
    agent.interpreter.resetHead();

    // (a) 正对球：状态说看得见，画面差分也必须能找到球，而且位置要对得上
    s.placeDuck({ x: 0, y: 0, yaw: 0 });
    let snap = agent.sensor.snapshot("ball", 0);
    const diff = await s.headCamDiff(s.ballGeoms());
    out.facing = {
      visible: snap.targetVisible, range: snap.targetRangeM, uv: snap.targetUv,
      diffChanged: diff.changed, diffUv: diff.centroid, image: diff.image,
    };

    // (b) 背对球：状态必须变成 not_found，而且画面里差分不出球
    s.placeDuck({ x: 0, y: 0, yaw: Math.PI });
    snap = agent.sensor.snapshot("ball", 0);
    const diff2 = await s.headCamDiff(s.ballGeoms());
    out.away = { visible: snap.targetVisible, range: snap.targetRangeM, diffChanged: diff2.changed };
    return out;
  });
  check("面向球时状态可见", t1.facing.visible === true,
        `range=${t1.facing.range?.toFixed(2)} m, uv=(${t1.facing.uv?.map((v) => v.toFixed(2)).join(", ")})`);
  check("面向球时画面差分也找得到球", t1.facing.diffChanged > 20,
        `${t1.facing.diffChanged} px（320x240 中的 ${(t1.facing.diffChanged / 768).toFixed(1)}%）`);
  const uvErr = t1.facing.diffUv && t1.facing.uv
    ? Math.hypot(t1.facing.diffUv.u - t1.facing.uv[0], t1.facing.diffUv.v - t1.facing.uv[1]) : 9;
  check("状态给的图像坐标就是球在画面里的位置", uvErr < 0.05,
        `偏差 ${(uvErr * 100).toFixed(1)}%（归一化图像坐标）`);
  check("背对球时状态变成 not_found", t1.away.visible === false && t1.away.range === null,
        `visible=${t1.away.visible}, range=${t1.away.range}`);
  check("背对球时画面里也差分不出球（真看不见，不是猜的）", t1.away.diffChanged === 0,
        `${t1.away.diffChanged} px`);

  // ============================================================ T1b 遮挡
  console.log("\n== T1b 遮挡：几何上在视野里、但被物体挡住 ==");
  const t1b = await page.evaluate(async () => {
    const s = window.__sim, agent = s.agent;
    s.reset();
    // obstacle_2 是位于 (1.10, 1.05) 的圆柱，半径 0.10、顶面 z=0.24。
    // 把球放在它后面 (1.10, 0.75)，鸭子站在另一边 (1.10, 1.35) 朝 -y 看过去：
    // 视线正好穿过圆柱轴心 → 球被完全挡住，但仍然落在相机视锥内。
    s.placeObject("ball", { x: 1.10, y: 0.75, z: 0.035 });
    s.placeDuck({ x: 1.10, y: 1.35, yaw: -Math.PI / 2 });
    const snap = agent.sensor.snapshot("ball", 0);
    const frustum = s.duck.projectToHeadCam([s.duck.ballPose.x, s.duck.ballPose.y, s.duck.ballPose.z]);
    const mask = s.maskVisible(s.ballGeoms());
    const diff = await s.headCamDiff(s.ballGeoms());
    return {
      visible: snap.targetVisible, range: snap.targetRangeM,
      inFrustum: !!frustum, frustumUv: frustum ? [frustum.u, frustum.v] : null,
      maskCount: mask.count, diffPixels: diff.changed,
    };
  });
  check("视线穿过圆柱轴心时球被挡住", t1b.visible === false && t1b.range === null,
        `visible=${t1b.visible}`);
  check("但它确实还在相机视锥里（所以不是靠出画蒙对的）", t1b.inFrustum === true,
        `视锥内 uv=(${t1b.frustumUv?.map((v) => v.toFixed(2)).join(", ")})`);
  check("渲染遮挡判断：目标像素数为 0", t1b.maskCount === 0, `${t1b.maskCount} px`);
  check("独立复核：隐藏球的像素差分也是 0", t1b.diffPixels === 0, `${t1b.diffPixels} px（旧 mj_ray 方案在这里会误判为可见）`);

  // ============================================================ T2 近距死锁
  console.log("\n== T2 球掉出画面下方 -> LOOK_DOWN 把它找回来 ==");
  const t2 = await page.evaluate(async () => {
    const s = window.__sim, agent = s.agent;
    s.reset();                                  // 先复位，再读球的位置（T1b 挪过球）
    const ball = { ...s.duck.ballPose };
    // 先在 0.55 m 处“看见”球（真实追球过程里就是这样先看到再跟丢的）
    s.placeDuck({ x: ball.x - 0.55, y: ball.y, yaw: 0 });
    agent.buildObservation();
    // 再贴到 0.18 m：球会掉到画面下沿之外
    s.placeDuck({ x: ball.x - 0.18, y: ball.y, yaw: 0 });
    const before = agent.sensor.snapshot("ball", 0);
    const imgBefore = s.captureHeadCam();
    const ctrlIdx = s.duck.headCtrl.head_pitch;

    s.pause();
    agent.config.mode = "rule";
    agent.config.decisionEveryS = 0;
    s.setTask({ text: "go to the ball" });
    // 0.55 m 先记录一次“看得见”，否则 lastSeen 为空、低头规则不会触发
    s.placeDuck({ x: ball.x - 0.55, y: ball.y, yaw: 0 });
    agent.buildObservation();
    s.placeDuck({ x: ball.x - 0.18, y: ball.y, yaw: 0 });
    // 跑到第 2 次决策：第 1 次会下发 LOOK_DOWN 并跑完 1.2 s
    const out = await s.runAgent({ maxDecisions: 2, maxSteps: 260 });
    const after = agent.sensor.snapshot("ball", 0);
    const imgAfter = s.captureHeadCam();
    const shift = await s.headCamShift(imgBefore, imgAfter);
    return {
      before: { visible: before.targetVisible, range: before.targetRangeM, uv: before.targetUv },
      after: { visible: after.targetVisible, range: after.targetRangeM, uv: after.targetUv },
      headPitchCtrl: s.duck.data.ctrl[ctrlIdx],
      nominal: 0.3491, shift, run: out,
      imgBefore, imgAfter,
    };
  });
  check("贴到跟前时球确实看不见了（原始死锁条件）", t2.before.visible === false,
        `range=${t2.before.range}, uv=${JSON.stringify(t2.before.uv)}`);
  check("agent 选了 LOOK_DOWN 而不是继续乱走", /LOOK_DOWN/.test(JSON.stringify(t2.run.history)),
        `history=${JSON.stringify(t2.run.history)}`);
  check("头部关节真的压下去了", t2.headPitchCtrl > t2.nominal + 0.5,
        `head_pitch ctrl=${t2.headPitchCtrl.toFixed(3)}（标称 ${t2.nominal} + 0.60）`);
  check("低头之后球重新进入视野", t2.after.visible === true,
        `uv=${t2.after.uv?.map((v) => v.toFixed(2)).join(", ")}`);
  check("低头确实改变了画面（不是数值上骗人）", Math.abs(t2.shift.shiftPx) > 8,
        `画面内容整体位移 ${t2.shift.shiftPx} px`);

  // ============================================================ T3 规则模式闭环
  console.log("\n== T3 无 key（规则模式）也能跑通闭环 ==");
  const t3 = await page.evaluate(async () => {
    const s = window.__sim, agent = s.agent;
    s.reset();                       // 关键帧复位：鸭子和球都回到场景初始位姿
    agent.interpreter.resetHead();
    agent.setTask({ text: "walk to the orange ball" });
    agent.config.mode = "rule";
    agent.config.decisionEveryS = 0;
    agent.config.maxSteps = 40;
    const run = await s.runAgent({ maxDecisions: 30, maxSteps: 900 });
    const snap = agent.sensor.snapshot(agent.task.target, 0);
    const done = agent.history.includes("DONE");
    return { run, snap: { visible: snap.targetVisible, range: snap.targetRangeM }, task: agent.task,
             done, minRange: agent.minRange,
             records: agent.records.slice(0, 8).map((r) => `${r.token}${r.rules.length ? "(" + r.rules.join("+") + ")" : ""}`) };
  });
  check("自由文本任务推出正确目标", t3.task.target === "ball", JSON.stringify(t3.task));
  check("规则模式跑出了动作序列", t3.run.decisions > 0, `${t3.run.decisions} 次决策 / ${t3.run.steps} 个控制步`);
  check("鸭子没有摔倒", t3.run.upright > 0.9, `upright=${t3.run.upright.toFixed(3)}`);
  check("走到了球跟前（历史最近距离 < 0.40 m）", t3.minRange !== null && t3.minRange < 0.40,
        `最近 ${t3.minRange?.toFixed(2)} m，最终 ${t3.snap.range?.toFixed(2) ?? "看不见"} m`);
  check("到达后宣告 DONE", t3.done === true, `history=${JSON.stringify(t3.run.history)}`);
  check("决策历史里有前进", t3.records.some((r) => r.startsWith("FWD")), t3.records.join(" -> "));

  // ============================================================ T4 真 HTTP 路径
  console.log("\n== T4 BYO key 的 HTTP 链路（本地假端点，证明请求体/图片/跨域都通）==");
  const mock = await startMockLlm({ reply: ["TURN_L", "FWD", "DONE"] });
  console.log("  [mock]", mock.url);
  const t4 = await page.evaluate(async ({ baseUrl }) => {
    const s = window.__sim, agent = s.agent;
    s.reset();
    s.setTask({ taskId: "", text: "go to the ball" });   // setTask 会顺手清空 decisions 记录
    agent.config.mode = "llm";
    agent.config.provider = "openai";
    agent.config.baseUrl = baseUrl;
    agent.config.apiKey = "sk-local-test";
    agent.config.model = "mock-vl";
    agent.config.decisionEveryS = 0;
    agent.config.maxSteps = 6;
    const run = await s.runAgent({ maxDecisions: 3, maxSteps: 400 });
    return { run, records: agent.records.map((r) => ({ token: r.token, source: r.source, model: r.model,
             latency: r.latencyMs, raw: r.raw, err: r.error })) };
  }, { baseUrl: mock.url });
  const req = mock.requests[0];
  check("假端点收到请求", mock.requests.length >= 1, `${mock.requests.length} 次`);
  check("走了 OpenAI 兼容路径", String(req?.url || "").includes("/v1/chat/completions"), req?.url);
  check("带上了 Authorization", /^Bearer sk-local-test$/.test(req?.headers?.authorization || ""), req?.headers?.authorization);
  check("请求体带模型名", req?.body?.model === "mock-vl", req?.body?.model);
  const content = req?.body?.messages?.[0]?.content || [];
  const imgUrl = content.find((c) => c.type === "image_url")?.image_url?.url || "";
  const textPart = content.find((c) => c.type === "text")?.text || "";
  const imgBytes = imgUrl.startsWith("data:image/jpeg;base64,")
    ? Buffer.from(imgUrl.split(",")[1], "base64") : Buffer.alloc(0);
  check("图片是一张真的 JPEG（data URL，SOI 头正确）",
        imgUrl.startsWith("data:image/jpeg;base64,") && imgBytes.length > 1000 &&
        imgBytes[0] === 0xff && imgBytes[1] === 0xd8 && imgBytes[2] === 0xff,
        `${imgBytes.length} 字节 ≈ ${(imgBytes.length / 1024).toFixed(1)} KB`);
  // 词表现在是 8 个基础动作 + 通过验证的技能（翻滚/跳舞）；未验证的踢球类不该出现
  check("提示词带动作菜单，且不含未验证的技能", textPart.includes("AVAILABLE TOKENS") && textPart.includes("LOOK_DOWN") &&
        !textPart.includes("KICK_L"));
  check("提示词是鸭子真实状态（不是模板）", /body=\(-?\d/.test(textPart) && /TASK: go to the ball/.test(textPart));
  // records 是「新在前」，所以第 0 条其实是最后一次决策
  const order = t4.records.map((r) => r.token).reverse();
  check("三条回复按顺序被解析并执行", JSON.stringify(order) === JSON.stringify(["TURN_L", "FWD", "DONE"]),
        order.join(" -> "));
  check("三条都标记来源为 vlm", t4.records.every((r) => r.source === "vlm"), JSON.stringify(t4.records.map((r) => r.source)));
  check("记录了推理延迟", t4.records.every((r) => typeof r.latency === "number" && r.latency >= 0),
        `${t4.records.map((r) => r.latency?.toFixed(1)).join(", ")} ms`);
  check("鸭子执行了模型给的动作", t4.run.steps > 0, `${t4.run.steps} 步`);

  // ============================================================ T5 厂商 CORS
  console.log("\n== T5 浏览器能否直连真实厂商（DashScope，用假 key 只验 CORS）==");
  const t5 = await page.evaluate(async () => {
    try {
      const res = await fetch("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer sk-invalid-cors-probe" },
        body: JSON.stringify({ model: "qwen3-vl-plus", messages: [{ role: "user", content: "ping" }], max_tokens: 1 }),
      });
      const text = (await res.text()).slice(0, 200);
      return { reachable: true, status: res.status, body: text, acao: res.headers.get("access-control-allow-origin") };
    } catch (e) {
      return { reachable: false, error: String(e.message || e) };
    }
  });
  check("浏览器直连 DashScope 不被 CORS 拦（拿到可读的 HTTP 响应）", t5.reachable === true,
        t5.reachable ? `HTTP ${t5.status}（假 key 应当被拒，但响应可读 = 跨域放行）`
                     : `被拦：${t5.error}`);
  check("响应里带 access-control-allow-origin", !!t5.acao, String(t5.acao));

  // ============================================================ 截图
  console.log("\n== 截图 ==");
  await page.evaluate(() => {
    const s = window.__sim;
    s.placeDuck({ x: 0, y: 0, yaw: 0 });
    s.setMode("workspace"); s.render();
  });
  await page.waitForTimeout(200);
  const clip = await page.evaluate(() => {
    const c = document.getElementById("canvas").getBoundingClientRect();
    return { x: c.x, y: c.y, width: c.width, height: c.height };
  });
  await page.screenshot({ path: path.join(ART, "step2_workspace.png"), clip });
  // VLM 实际看到的 320x240（把 data URL 落盘）
  const img = await page.evaluate(() => window.__sim.captureHeadCam());
  await page.evaluate(async (dataUrl) => {
    const a = document.createElement("a"); a.href = dataUrl; a.download = "x";
  }, img);
  const b64 = img.split(",")[1];
  const { writeFile } = await import("node:fs/promises");
  await writeFile(path.join(ART, "step2_vlm_eye.jpg"), Buffer.from(b64, "base64"));
  console.log(`  ${path.join(ART, "step2_workspace.png")}   ← 全景`);
  console.log(`  ${path.join(ART, "step2_vlm_eye.jpg")}      ← VLM 实际收到的 320x240 头摄`);

  const failed = results.filter((r) => !r.ok);
  console.log(`\n结果: ${failed.length ? "FAIL" : "PASS"} —— ${results.length - failed.length}/${results.length} 项通过`);
  if (failed.length) console.log("未通过：" + failed.map((f) => f.name).join("；"));

  await mock.close();
  await browser.close();
  await site.close();
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => { console.error("异常:", e); process.exit(1); });
