/**
 * 决策层的单元测试（纯 Node，不开浏览器、不发网络请求）。
 *
 * 覆盖：动作词表与解析、解释器时序、提示词渲染、场景与意图解析、规则层、
 *      LLM 端点拼接、agent 的任务推断与进度 verbalization。
 *
 * 用法：node tools/test_agent.mjs
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ACTION_SPECS, TOKENS, SKILL_TOKENS, availableTokens, parseToken, tokenMenu, tokenSemantics } from "../src/actions.js";
import { ActionInterpreter } from "../src/interpreter.js";
import { renderPrompt } from "../src/prompt.js";
import { SceneIndex } from "../src/scene.js";
import { resolveIntent, pickTarget, isBlockingIntent } from "../src/intent.js";
import { applyRules, reflexToken, DEFAULT_RULES } from "../src/rules.js";
import { resolveEndpoint, splitDataUrl } from "../src/llm.js";
import { DuckAgent } from "../src/agent.js";
import { DuckStateSensor } from "../src/state.js";
import { parseSequence, parseNumber, describeSequence } from "../src/sequence.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCENE_DIR = path.resolve(HERE, "../assets/duck_workspace_v1");

let passed = 0, failed = 0;
function eq(name, got, want) {
  if (JSON.stringify(got) === JSON.stringify(want)) { passed++; console.log(`  PASS  ${name}`); }
  else {
    failed++;
    console.log(`  FAIL  ${name}\n        期望 ${JSON.stringify(want)}\n        实际 ${JSON.stringify(got)}`);
  }
}
function ok(name, cond, detail = "") {
  if (cond) { passed++; console.log(`  PASS  ${name}`); }
  else { failed++; console.log(`  FAIL  ${name}${detail ? "   " + detail : ""}`); }
}

console.log("\n== 1. 动作词表与解析 ==");
const BASE8 = ["FWD", "BACK", "TURN_L", "TURN_R", "STOP", "LOOK_DOWN", "HEAD_CENTER", "DONE"];
const NO_POLICY = availableTokens({});
eq("基础 token 是 8 个", TOKENS.filter((t) => !SKILL_TOKENS.includes(t)), BASE8);
eq("技能 token 6 个", SKILL_TOKENS.length, 6);
eq("没加载策略时，发给模型的只有 8 个基础动作", NO_POLICY, BASE8);
eq("加载翻滚+跳舞后词表多两个", availableTokens({ roulade: {}, happy_hop: {} }), [...BASE8, "ROLL", "DANCE"]);
eq("未验证的技能（踢球）不进词表", availableTokens({ ball_kick_right: {} }), BASE8);
eq("显式要求时才带未验证技能", availableTokens({ ball_kick_right: {} }, { includeUnverified: true }), [...BASE8, "KICK_R"]);
eq("直接输出", parseToken("FWD", TOKENS), "FWD");
eq("代码围栏", parseToken("```text\nTURN_L\n```", TOKENS), "TURN_L");
eq("JSON 包裹", parseToken('{"action": "turn right"}', TOKENS), "TURN_R");
eq("大写别名", parseToken("FORWARD", TOKENS), "FWD");
eq("中文别名（低头）", parseToken("低头", TOKENS), "LOOK_DOWN");
eq("中文别名（抬头）", parseToken("抬头", TOKENS), "HEAD_CENTER");
eq("句子里的唯一 token", parseToken("I would go backward now", TOKENS), "BACK");
eq("纯噪声", parseToken("banana", TOKENS), null);
eq("空回复", parseToken("   ", TOKENS), null);
eq("不在允许集里的 token", parseToken("KICK_L", NO_POLICY), null);
ok("词表菜单列出 8 个", tokenMenu(NO_POLICY).split(",").length === 8);
ok("语义说明逐行列出", tokenSemantics(NO_POLICY).split("\n").length === 8);
ok("LOOK_DOWN 语义提到地面", /ground/i.test(ACTION_SPECS.LOOK_DOWN.description));

console.log("\n== 2. 解释器：token -> 运动指令 ==");
{
  const it = new ActionInterpreter();
  eq("FWD 指令", it.start("FWD").command, [0.35, 0, 0]);
  eq("FWD 时间未到不结束", it.tick(0.54).done, false);
  eq("FWD 时间到就结束", it.tick(0.02).done, true);
  eq("结束后处于空闲", it.busy, false);

  eq("TURN_R 指令", new ActionInterpreter().start("TURN_R").command, [0, 0, -1.5]);
  eq("STOP 指令", new ActionInterpreter().start("STOP").command, [0, 0, 0]);

  const head = new ActionInterpreter();
  head.start("LOOK_DOWN");
  eq("LOOK_DOWN 保持站立", head.tick(0.5).command, [0, 0, 0]);
  eq("LOOK_DOWN 低头 0.60 rad", head.headOverride().head_pitch, 0.6);
  eq("LOOK_DOWN 计时结束", head.tick(0.8).done, true);
  eq("结束后头部偏移仍然锁着（不会自动回正）", head.headOverride().head_pitch, 0.6);
  head.start("HEAD_CENTER");
  eq("HEAD_CENTER 才回正", head.headOverride().head_pitch, 0);

  const done = new ActionInterpreter();
  eq("DONE 立即结束", done.start("DONE").done, true);
  eq("DONE 不占着解释器", done.busy, false);
  const bad = new ActionInterpreter().start("MOONWALK");
  ok("未知 token 会被拒绝", bad.done && /unknown/.test(bad.note), JSON.stringify(bad));
}

console.log("\n== 3. 提示词渲染 ==");
{
  const obs = {
    task: "去红方块", taskId: "go_to_red_cube", target: "obj_cube_red",
    stateText: "body=(0.00,0.00,0.13) | heading=0.00rad | target(obj_cube_red)=range 0.86m, bearing 0.40rad, visible=yes",
    subgoal: "align: obj_cube_red is 0.86 m away and 23 deg to the left",
    recentActions: ["TURN_L", "FWD"],
    affordanceText: "obj_cube_red: visible at image (u=0.30, v=0.55)",
    proprioText: "proprio: head=forward", recoveryHint: "",
  };
  const p = renderPrompt(obs, NO_POLICY);      // 真正会发给模型的就是这 8 个
  ok("含 TASK 段", p.includes("TASK: 去红方块"));
  ok("含目标", p.includes("TARGET: obj_cube_red"));
  ok("含本体状态", p.includes(obs.stateText));
  ok("含进度（米）", p.includes("0.86 m"));
  ok("含最近动作", p.includes("TURN_L, FWD"));
  ok("含 token 菜单", p.includes("AVAILABLE TOKENS:") && p.includes("LOOK_DOWN"));
  ok("含控制规则", p.includes("CONTROL RULES:"));
  ok("不提示没有的技能", !p.includes("KICK_L") && !p.includes("STAND_UP") && !p.includes("ROLL"));
  ok("没有漏填的占位符", !/\{[a-z_]+\}/.test(p));
}

console.log("\n== 4. 自然语言 -> body / 意图 ==");
{
  const metadata = JSON.parse(await readFile(path.join(SCENE_DIR, "metadata.json"), "utf8"));
  const tasks = JSON.parse(await readFile(path.join(SCENE_DIR, "tasks.json"), "utf8"));
  const scene = new SceneIndex(metadata, tasks);

  eq("去红方块", scene.resolveBody("去红方块"), "obj_cube_red");
  eq("绿色区域", scene.resolveBody("绿色区域"), "zone_green");
  eq("橙色球", scene.resolveBody("橙色球"), "ball");
  eq("蓝色方块", scene.resolveBody("蓝色方块"), "obj_cube_blue");
  eq("黄色信标", scene.resolveBody("黄色信标"), "beacon_target");

  const hits = scene.scanText("把红方块推到蓝色区域");
  const bodies = hits.map((h) => h[1]);
  ok("句子扫出红方块和蓝色区域", bodies.includes("obj_cube_red") && bodies.includes("zone_blue"), JSON.stringify(bodies));
  ok("不会误匹配“蓝色方块”", !bodies.includes("obj_cube_blue"), JSON.stringify(bodies));
  eq("操作类意图优先选方块（不是区域）", pickTarget(hits, "把红方块推到蓝色区域").body, "obj_cube_red");
  eq("“把球踢进绿色区域”盯的是球", pickTarget(scene.scanText("把球踢进绿色区域"), "把球踢进绿色区域").body, "ball");
  eq("走过去 = approach", resolveIntent("", "走过去"), "approach");
  eq("抬腿 = 未知", resolveIntent("", "抬腿"), "unknown");
  eq("原地右转 = pose", resolveIntent("", "原地右转"), "pose");
  eq("别靠近蓝方块 = avoid", resolveIntent("", "别靠近蓝方块"), "avoid");
  eq("avoid 时不去追那个东西", pickTarget(scene.scanText("别靠近蓝方块"), "别靠近蓝方块").body, "ball");
  ok("pose 会挡住寻路", isBlockingIntent("", "原地右转") === true);
  ok("approach 不挡寻路", isBlockingIntent("", "走过去") === false);

  const curated = scene.task("push_red_cube");
  ok("场景任务表里有 push_red_cube", !!curated);
  ok("任务表至少 6 条", scene.listTasks().length >= 6, `实际 ${scene.listTasks().length}`);
}

console.log("\n== 5. 规则层（可消融）==");
{
  const state = { fallen: false, targetVisible: true, targetRangeM: 0.5, targetBearingRad: 0.0 };
  const base = { recentActions: [], state, lastSeen: null, stepsSinceLastSeen: 999, distanceMovedInWindow: 1 };
  eq("什么都不触发时原样通过", applyRules({ ...base, proposed: "FWD" }).token, "FWD");

  const fallen = applyRules({ ...base, proposed: "FWD", state: { ...state, fallen: true } });
  eq("摔倒时否决前进", fallen.token, "STOP");
  eq("记录触发的规则名", fallen.applied, ["fallen_stop"]);

  const stuck = applyRules({ ...base, proposed: "FWD", recentActions: ["FWD", "FWD"], distanceMovedInWindow: 0.005 });
  eq("顶住墙 -> 转向脱困", stuck.token, "TURN_L");
  eq("触发反卡死", stuck.applied, ["anti_stuck"]);

  const spin = applyRules({ ...base, proposed: "TURN_L", recentActions: ["TURN_R", "TURN_L"] });
  eq("原地互摆 -> 改前进", spin.token, "FWD");
  eq("触发反空转", spin.applied, ["anti_spin"]);

  const lost = applyRules({ ...base, proposed: "TURN_L", state: { ...state, targetVisible: false },
                            lastSeen: { rangeM: 0.4 }, stepsSinceLastSeen: 2 });
  eq("近处丢目标 -> 低头找", lost.token, "LOOK_DOWN");

  const off = { ...DEFAULT_RULES, fallen_stop: { enabled: false, description: "x" } };
  eq("关掉规则后不再否决", applyRules({ ...base, proposed: "FWD", state: { ...state, fallen: true } }, off).token, "FWD");

  eq("reflex: 看不见就扫视", reflexToken({ fallen: false, targetVisible: false }), "TURN_L");
  eq("reflex: 目标在左边就左转", reflexToken({ fallen: false, targetVisible: true, targetBearingRad: 0.5, targetRangeM: 1 }), "TURN_L");
  eq("reflex: 对准了就走", reflexToken({ fallen: false, targetVisible: true, targetBearingRad: 0.1, targetRangeM: 1 }), "FWD");
  eq("reflex: 到了就结束", reflexToken({ fallen: false, targetVisible: true, targetBearingRad: 0.1, targetRangeM: 0.2 }), "DONE");
}

console.log("\n== 6. LLM 端点与图片编码 ==");
eq("base 只有域名", resolveEndpoint("https://api.example.com"), "https://api.example.com/v1/chat/completions");
eq("base 已带 /v1", resolveEndpoint("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
   "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions");
eq("base 已是完整端点", resolveEndpoint("http://127.0.0.1:8000/v1/chat/completions"),
   "http://127.0.0.1:8000/v1/chat/completions");
{
  const { mime, base64 } = splitDataUrl("data:image/jpeg;base64,QUJD");
  eq("data URL 的 mime", mime, "image/jpeg");
  eq("data URL 的 base64", base64, "QUJD");
}

console.log("\n== 7. agent：任务推断与进度 verbalization ==");
{
  const metadata = JSON.parse(await readFile(path.join(SCENE_DIR, "metadata.json"), "utf8"));
  const tasks = JSON.parse(await readFile(path.join(SCENE_DIR, "tasks.json"), "utf8"));
  const scene = new SceneIndex(metadata, tasks);
  const fakeDuck = {
    steps: 100, pose: { x: 0, y: 0 }, upright: () => 1,
    mujoco: { mjtObj: { mjOBJ_BODY: { value: 1 } } }, model: {}, data: {},
  };
  const agent = new DuckAgent({ duck: fakeDuck, view: null, scene });
  const fakeState = {
    simTime: 2, x: 0, y: 0, z: 0.13, headingRad: 0, linearSpeedMps: 0, angularSpeedRps: 0,
    upright: 1, fallen: false, targetName: "obj_cube_red", targetVisible: true,
    targetRangeM: 0.86, targetBearingRad: 0.40, targetElevationRad: 0.0,
    targetUv: [0.3, 0.55], targetWorldXyz: [0.8, 0.35, 0.03],
  };
  agent.sensor = { snapshot: () => fakeState };

  const t = agent.setTask({ text: "去红方块" });
  eq("自由文本推出目标", t.target, "obj_cube_red");
  eq("自由文本推出意图", t.intent, "approach");

  // 精选任务的靶子必须**从成功判据推导**出来，而不是靠措辞推断。
  // 旧断言只检查"target 是非空字符串"，而推断出来的 "ball" 也满足它 —— 测试太弱，
  // 掩盖了"14 个任务全在靠推断"这个事实（见 tools/check_task_schema.mjs）。
  const EXPECT = [
    ["walk_to_ball", "ball"],
    ["kick_ball_to_zone", "ball"],
    ["push_red_cube", "obj_cube_red"],
    ["choose_red_not_blue", "obj_cube_red"],      // 别被"距离蓝方块≥0.45"那条判据抢走
    ["avoid_obstacle_reach_beacon", "beacon_target"],
  ];
  for (const [id, want] of EXPECT) {
    const t = agent.setTask({ taskId: id });
    eq(`${id} 的目标来自成功判据`, t.target, want);
    eq(`${id} 标记为 declared`, t.targetSource, "declared");
  }
  const kickZone = agent.setTask({ taskId: "kick_ball_to_zone" });
  eq("踢球任务的区域半径用判据里的 0.30（不是元数据的 0.35）",
     agent.stationZone?.successRadius, 0.3);
  ok("踢球任务带上了区域名", agent.stationZone?.name === "zone_green", kickZone.text);

  const freeTask = agent.setTask({ text: "去红方块" });
  eq("自由文本标记为 inferred", freeTask.targetSource, "inferred");
  eq("自由文本仍能推出目标", freeTask.target, "obj_cube_red");

  const posed = agent.setTask({ taskId: "walk_turn_stop" });
  ok("姿态类任务被标成 no-object（没有目标物体）",
     posed.targetSource === "inferred-no-object", `${posed.targetSource}/${posed.target}`);

  agent.setTask({ text: "去红方块" });
  const obs = agent.buildObservation();
  ok("ROBOT STATE 带米数", obs.stateText.includes("range 0.86m"), obs.stateText);
  ok("ROBOT STATE 带航向误差", obs.stateText.includes("bearing 0.40rad"), obs.stateText);
  ok("SUBTASK 带角度与米数", /0\.86 m/.test(obs.subgoal) && /deg/.test(obs.subgoal), obs.subgoal);
  ok("AFFORDANCE 给的是图像坐标", obs.affordanceText.includes("u=0.30"), obs.affordanceText);
  ok("本体状态文本与 Python 同格式", DuckStateSensor.promptText(fakeState).includes("upright=1.00"));

  const blind = { ...fakeState, targetVisible: false, targetRangeM: null, targetBearingRad: null, targetUv: null };
  agent.sensor = { snapshot: () => blind };
  const obs2 = agent.buildObservation();
  ok("看不见目标时进度变成搜索", obs2.subgoal.startsWith("search:"), obs2.subgoal);
  ok("看不见目标时状态里没有距离", obs2.stateText.includes("not_found"), obs2.stateText);
  ok("看不见目标时 affordance 说明未检出", /not detected/.test(obs2.affordanceText), obs2.affordanceText);

  // —— 决策闸门：一次决策还在飞的时候被打断，之后必须还能再发起决策。
  //
  // 用户报的 bug：VLM 模式跑一次 → 换句指令 → 再点开始，鸭子不动、也不再推理。
  // 根因是 _pending 这个"同一时刻只允许一次决策"的闸门**只有 tick() 的 thinking
  // 分支会消费**；在请求在飞时按「停止决策」/「复位」把 phase 改掉之后，那个结果
  // 就永远没人取走，闸门从此关着。
  agent.sensor = { snapshot: () => fakeState };
  agent.setTask({ text: "去红方块" });
  let lateResolve = null;
  agent._decide = () => new Promise((r) => { lateResolve = r; });   // 假装请求还挂在网上
  agent._idleS = agent.config.decisionEveryS;
  agent.tick(0.02);
  ok("够钟后进入 thinking 并挂上闸门", agent.phase === "thinking" && !!agent._pending,
     `${agent.phase} / pending=${!!agent._pending}`);

  agent.abortPending();                       // == 用户点「停止决策」
  agent.phase = "finished";
  lateResolve({ token: "FWD", note: "迟到的结果", error: null });   // 请求这时候才落地
  ok("「停止决策」把闸门清掉，迟到的结果不会再堵住下一次",
     agent._pending === null && agent.phase === "finished");

  agent.setTask({ text: "去绿区" });           // == 用户换一句指令
  ok("换任务时闸门重置、计数归零",
     agent._pending === null && agent._idleS === 0 && agent.records.length === 0);
  agent._idleS = agent.config.decisionEveryS;
  const resumed = agent.tick(0.02);
  ok("换指令后再点开始能重新发起决策", agent.phase === "thinking" && resumed.phase === "thinking",
     `${agent.phase} / ${resumed.phase}`);
  agent.abortPending();
}

console.log("\n== 9. 动作序列解析（「前进1米，再翻滚一次，最后跳舞」）==");
{
  eq("中文数字：二十五", parseNumber("二十五"), 25);
  eq("中文数字：十二", parseNumber("十二"), 12);
  eq("中文数字：半", parseNumber("半"), 0.5);
  eq("阿拉伯数字：1.5", parseNumber("1.5"), 1.5);

  const seq = parseSequence("前进1米，再翻滚一次，最后跳舞");
  eq("用户原句拆成 3 步", seq.map((s) => s.kind), ["move", "skill", "skill"]);
  eq("前进的米数被读出来", seq[0].meters, 1);
  eq("翻滚在前、跳舞在后", [seq[1].token, seq[2].token], ["ROLL", "DANCE"]);
  eq("能读成一句话", describeSequence(seq), "前进 1 m → 翻滚 → 跳舞");

  eq("左转 + 前进", parseSequence("原地左转90度，然后前进0.5米").map((s) => [s.kind, s.deg ?? s.meters]),
     [["turn", 90], ["move", 0.5]]);
  eq("右转是负角度", parseSequence("右转45度")[0].deg, -45);
  eq("没说角度默认 90 度", parseSequence("左转")[0].deg, 90);
  eq("后退半米", parseSequence("后退半米")[0].meters, -0.5);
  eq("翻滚三次展开成三步", parseSequence("翻滚三次，然后跳舞").map((s) => s.token),
     ["ROLL", "ROLL", "ROLL", "DANCE"]);

  // 关键：带目标的导航指令不能被误判成动作序列（否则会退化成"原地做动作"）
  eq("「去红方块旁边停下」不是序列", parseSequence("去红方块旁边停下"), []);
  eq("「把球踢进绿色区域」不是序列", parseSequence("把球踢进绿色区域"), []);
  eq("「找到绿色区域，走过去」不是序列", parseSequence("找到绿色区域，走过去"), []);
  eq("看不懂的句子不猜", parseSequence("随便说点什么"), []);
}

console.log(`\n结果: ${failed ? "FAIL" : "PASS"} —— ${passed}/${passed + failed} 项通过`);
process.exit(failed ? 1 : 0);
