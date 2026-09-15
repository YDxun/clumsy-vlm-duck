/**
 * 「感知 — 推理 — 动作」闭环。
 *
 * 一个决策周期的真实流程（每一步都用真实传感器数据，不许偷偷用仿真真值）：
 *
 *   1. 抓一张 320×240 的头摄图（view.captureHeadCam，构图与 Python 端一致）
 *   2. 采本体状态 + 目标可见性（state.js；目标没看见就不给距离和方位）
 *   3. 拼提示词（prompt.js，字段与 Python 端同一套）
 *   4. 问模型一个 token（llm.js，浏览器直连厂商；无 key 时走规则兜底）
 *   5. 规则层可以否决/替换（rules.js，每条可单独关掉做消融）
 *   6. 解释器把 token 跑完（interpreter.js），期间鸭子持续维持姿态
 *   7. 记一条带图的决策记录，回到 1
 *
 * 为什么把「问模型」做成异步状态机而不是顺序调用：模型一次要 1~3 秒，
 * 这段时间物理不能停（停了画面就僵住），鸭子也不该保持上一个速度乱走。
 * 所以 thinking 阶段一律下发 [0,0,0] 让它原地站稳，头部姿态保持不动。
 */
import { ActionInterpreter } from "./interpreter.js";
import { DuckStateSensor } from "./state.js";
import { renderPrompt } from "./prompt.js";
import { applyRules, reflexToken, DEFAULT_RULES } from "./rules.js";
import { askVlm, PROVIDERS } from "./llm.js";
import { parseToken, TOKENS, availableTokens } from "./actions.js";
import { resolveIntent, pickTarget } from "./intent.js";
import { StationKicker, planStation } from "./station.js";

const DEG = 180 / Math.PI;

export class DuckAgent {
  constructor({ duck, view, scene = null, rules = DEFAULT_RULES, maxRecords = 80 }) {
    this.duck = duck;
    this.view = view;
    this.scene = scene;
    this.sensor = new DuckStateSensor({ duck, scene, view });
    this.interpreter = new ActionInterpreter();
    this.rules = rules;
    this.maxRecords = maxRecords;

    this.config = {
      mode: "rule",                 // rule | llm
      provider: "openai",
      model: PROVIDERS.openai.defaultModel,
      baseUrl: PROVIDERS.openai.defaultBaseUrl,
      apiKey: "",
      temperature: 0,
      maxTokens: 48,
      imageDetail: "high",
      reasoningEffort: "low",
      timeoutMs: 40000,
      maxSteps: 60,
      decisionEveryS: 0.2,          // 一个动作跑完后至少停这么久再决策（避免抢拍）
    };

    this.task = { text: "(no task set)", taskId: "", target: "ball", allowed: [...TOKENS] };
    this.policies = {};             // 策略名 -> session，决定哪些技能 token 能发给模型
    // 踢球/推球站位：见到球就算一次站位，之后由规则闭环把最后 20 cm 走完
    this.station = new StationKicker();
    this.stationEnabled = true;
    this.stationZone = null;
    this.phase = "idle";            // idle | thinking | acting | finished | error
    this.records = [];
    this.history = [];              // 已执行的 token（新在前）
    this.stepIndex = 0;
    this.lastNote = "";
    this.lastError = null;
    this.lastSeen = null;           // {rangeM, step}
    this.stepsSinceLastSeen = 999;
    this.minRange = null;           // 本局最近一次离目标多近
    this._pending = null;
    this._idleS = 0;
    this._lastMoveStep = 0;
    this._lastMoveXyz = null;
    this.onRecord = null;
  }

  /** 设定任务：可以给 task id（场景包里的精选任务），也可以给一句自由文本。 */
  setTask({ text = "", taskId = "", target = null } = {}) {
    const curated = taskId && this.scene ? this.scene.task(taskId) : null;
    const taskText = text || curated?.instruction_zh || curated?.instruction_en || taskId || "(no task set)";
    let resolvedTarget = target;
    if (!resolvedTarget) {
      const hits = this.scene ? this.scene.scanText(taskText) : [];
      resolvedTarget = (curated && curated.target) || pickTarget(hits, taskText).body;
    }
    this.task = {
      text: taskText,
      taskId: taskId || curated?.id || "",
      target: resolvedTarget || "ball",
      intent: resolveIntent(taskId, taskText),
      // 只把「策略在、且本地实测有效」的技能发给模型（见 actions.availableTokens）
      allowed: availableTokens(this.policies || {}),
    };
    this.history = [];
    this.stepIndex = 0;
    this.phase = "idle";
    this.lastSeen = null;
    this.stepsSinceLastSeen = 999;
    this.minRange = null;
    this.interpreter.cancel();
    this.interpreter.resetHead();
    this.stationZone = (this.scene && this.scene.zoneFor(taskText)) || null;
    this.station.reset();
    return this.task;
  }

  /** 站位阶段该不该由规则接管：有目标区域 + 是操作类意图 + 开了开关。 */
  stationApplies() {
    return !!(this.stationEnabled && this.stationZone &&
              ["manipulate", "approach"].includes(this.task.intent));
  }

  /**
   * 走一步站位闭环。
   * 球的位置**只在真的看见时**才喂进去；看不见就用锁存的计划继续走（航位推算）。
   */
  stationTick() {
    // 摔倒了就先别走：躺着继续下方位移指令只会在地上蹭
    if (this.duck.upright() < 0.55) {
      this.station.note = "鸭子倒了，先停住（起身策略还没接入）";
      return { cmd: [0, 0, 0], kick: false, phase: "fallen", note: this.station.note };
    }
    const needObs = !this.station.latched ||
                     (this.duck.steps - (this._stationObsStep ?? -999)) > 60;
    let ballXy = null;
    if (needObs) {
      this._stationObsStep = this.duck.steps;
      const s = this.sensor.snapshot(this.task.target, this.duck.steps * 0.02);
      if (s.targetVisible && s.targetWorldXyz) {
        ballXy = { x: s.targetWorldXyz[0], y: s.targetWorldXyz[1] };
        if (this.station.lastKick) {          // 刚踢过：用这一脚的真实落点标定偏角
          const bias = this.station.calibrate(ballXy);
          if (bias !== null) {
            this.lastNote = `标定踢球偏角：${(bias * 180 / Math.PI).toFixed(0)}°（原 ${(-33).toFixed(0)}°）`;
            this._stationBias = bias;
          }
        }
      }
    }
    return this.station.tick({ duckPose: this.duck.pose, ballXy, zone: this.stationZone });
  }

  /** 任务进度 verbalization：用真实单位报出还差多远、航向差多少。 */
  subgoalText(state) {
    if (state.fallen) return "recover: the duck is on the ground; stabilise before moving";
    if (!state.targetVisible || state.targetRangeM === null) {
      return `search: ${state.targetName} is not in the head camera (missing for ${this.stepsSinceLastSeen} decisions); ` +
             `turn to sweep, and after a few sweeps move forward to a new vantage point`;
    }
    const range = state.targetRangeM;
    const err = state.targetBearingRad;
    if (Math.abs(err) > 0.30) {
      return `align: ${state.targetName} is ${range.toFixed(2)} m away and ${Math.abs(err * DEG).toFixed(0)} deg ` +
             `to the ${err > 0 ? "left" : "right"}; turn until the heading error is under 17 deg`;
    }
    if (range > 0.35) {
      return `approach: heading error ${(err * DEG).toFixed(0)} deg, ${range.toFixed(2)} m to go; ` +
             `walk forward one step at a time and re-check`;
    }
    return `arrive: ${state.targetName} is ${range.toFixed(2)} m ahead (heading error ${(err * DEG).toFixed(0)} deg); ` +
           `finish the task or stop if nothing more is required`;
  }

  affordanceText(state) {
    const label = this.task.target;
    if (!state.targetVisible) {
      return `${label}: not detected in the head camera` +
             (this.lastSeen ? ` (last seen ${this.lastSeen.rangeM.toFixed(2)} m away, ${this.stepsSinceLastSeen} decisions ago)` : "");
    }
    const uv = state.targetUv;
    return `${label}: visible at image (u=${uv[0].toFixed(2)}, v=${uv[1].toFixed(2)}) ` +
           `[0,0 = top-left of the duck-camera], bearing ${state.targetBearingRad.toFixed(2)} rad, ` +
           `range ${state.targetRangeM.toFixed(2)} m, elevation ${state.targetElevationRad.toFixed(2)} rad`;
  }

  proprioText() {
    const hd = this.interpreter.headOverride();
    const looking = hd && hd.head_pitch > 0.2 ? `down (head_pitch +${hd.head_pitch.toFixed(2)} rad, still latched)` : "forward";
    const up = this.duck.upright();
    return `proprio: head=${looking} | upright=${up.toFixed(2)} | ` +
           `current action=${this.interpreter.token || "none"}`;
  }

  /** 组装这一轮要发给模型的所有东西（也方便测试直接检查，不用真的发请求）。 */
  buildObservation() {
    const simTime = this.duck.steps * 0.02;
    const state = this.sensor.snapshot(this.task.target, simTime);
    if (state.targetVisible) {
      this.lastSeen = { rangeM: state.targetRangeM, step: this.stepIndex };
      this.stepsSinceLastSeen = 0;
      // 最近一次离目标多近（给复跑/评分看进度，规则层不依赖它）
      this.minRange = Math.min(this.minRange ?? Infinity, state.targetRangeM);
    } else {
      this.stepsSinceLastSeen += 1;
    }
    const recentActions = this.history.slice(0, 5);
    const ctx = {
      proposed: "STOP",
      recentActions,
      state,
      lastSeen: this.lastSeen,
      stepsSinceLastSeen: this.stepsSinceLastSeen,
      distanceMovedInWindow: this.distanceMoved(3),
    };
    const obs = {
      task: this.task.text,
      taskId: this.task.taskId,
      target: this.task.target,
      state,
      stateText: DuckStateSensor.promptText(state),
      subgoal: this.subgoalText(state),
      recentActions,
      affordanceText: this.affordanceText(state),
      proprioText: this.proprioText(),
      recoveryHint: this.recoveryHint(),
      ruleCtx: ctx,
    };
    obs.prompt = renderPrompt(obs, this.task.allowed);
    return obs;
  }

  recoveryHint() {
    const r = this.records[0];
    if (!r) return "";
    if (r.rules && r.rules.length) {
      return `NOTE: the safety layer replaced your "${r.proposed}" with "${r.token}" (${r.rules.join(", ")}). ` +
             `Do not repeat an action that just got overridden.`;
    }
    if (r.source === "rule" && r.note === "parse-failed") {
      return `NOTE: your last answer could not be read as a token. Reply with exactly one token from AVAILABLE TOKENS.`;
    }
    return "";
  }

  /** 最近 window 个决策步里鸭子实际移动了多远（用来判断是不是顶住墙了）。 */
  distanceMoved(window = 3) {
    const trail = this._trail || [];
    if (trail.length < 2) return 0;
    const now = [this.duck.pose.x, this.duck.pose.y];
    const prev = trail[Math.max(0, trail.length - 1 - window)];
    return Math.hypot(now[0] - prev[0], now[1] - prev[1]);
  }

  _pushTrail() {
    this._trail = this._trail || [];
    this._trail.push([this.duck.pose.x, this.duck.pose.y]);
    if (this._trail.length > 12) this._trail.shift();
  }

  /**
   * 推进一个控制步。返回这一帧该下发的 { cmd, headDelta, phase }。
   * @param {number} dt 真实经过的时间（秒），用来给解释器计时
   */
  tick(dt) {
    if (this.phase === "finished" || this.phase === "error") {
      return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: this.phase };
    }
    if (this.phase === "thinking") {
      // 等模型的时候原地站稳，但头部姿态继续锁着（LOOK_DOWN 不会被走路策略顶回去）
      if (this._pending && this._pending.settled) {
        const rec = this._pending.value;
        this._pending = null;
        if (rec.error) {
          this.lastError = rec.error;
          this.phase = "error";
          return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "error" };
        }
        this.records.unshift(rec);
        if (this.records.length > this.maxRecords) this.records.pop();
        this.lastNote = rec.note;
        const res = this.interpreter.start(rec.token);
        this._pushTrail();
        if (res.done && rec.token === "DONE") {
          this.phase = "finished";
          this.history.unshift("DONE");
          if (this.onRecord) this.onRecord(rec);
          return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "finished" };
        }
        this.phase = "acting";
        this.currentRecord = rec;
        return { cmd: res.command, headDelta: this.interpreter.headOverride(), phase: "acting" };
      }
      return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "thinking" };
    }
    if (this.phase === "acting") {
      const out = this.interpreter.tick(dt);
      if (!out.done) {
        return { cmd: out.command, headDelta: this.interpreter.headOverride(), phase: "acting" };
      }
      this.history.unshift(this.currentRecord?.token || out.token);
      if (this.history.length > 20) this.history.pop();
      this.stepIndex += 1;
      if (this.currentRecord) {
        this.currentRecord.completedAt = this.duck.steps;
        if (this.onRecord) this.onRecord(this.currentRecord);
      }
      this.currentRecord = null;
      // 踢完一脚后重开站位：球被踢到哪了要用新观测重算，没进区域就再来一脚
      if (this.history[0] === "KICK_R" || this.history[0] === "KICK_L") {
        const attempts = this.stationKicks || 0;
        this.station.reset();
        this._stationObsStep = -999;
        this.lastNote = `第 ${attempts} 脚踢完，重新站位`;
      }
      if (this.stepIndex >= this.config.maxSteps) {
        this.phase = "finished";
        this.lastNote = `达到 max_steps=${this.config.maxSteps}`;
        return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "finished" };
      }
      this.phase = "idle";
      this._idleS = 0;
      return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "idle" };
    }

    // idle：停够 decisionEveryS 之后再开新一轮决策
    this._idleS += dt;

    // 站位优先：操作类任务的最后 20 cm 交给规则闭环，别让 VLM 用 0.07 m 的步子去对 1.5 cm 的容差
    if (this.stationApplies()) {
      const st = this.stationTick();
      this.lastNote = st.note;
      if (st.done) {
        this.phase = "finished";
        this.lastNote = `任务完成：${st.note}`;
        return { cmd: [0, 0, 0], headDelta: null, phase: "finished" };
      }
      if (st.kick) {
        this.stationKicks = (this.stationKicks || 0) + 1;
        const res = this.interpreter.start(this.station.foot === "right" ? "KICK_R" : "KICK_L");
        if (!res.done) {
          this.phase = "acting";
          this.currentRecord = {
            stepIndex: this.stepIndex, simTime: this.duck.steps * 0.02, task: this.task.text,
            taskId: this.task.taskId, target: this.task.target, state: null, prompt: "",
            image: "", proposed: res.token, token: res.token, source: "station",
            raw: st.note, latencyMs: null, rules: ["station_kick"], note: st.note, error: null,
          };
        }
        return { cmd: res.command, headDelta: null, phase: "acting" };
      }
      if (st.phase !== "no-plan") {
        return { cmd: st.cmd, headDelta: null, phase: "station:" + st.phase };
      }
      // no-plan：还没看见球 → 掉回正常决策（模型/规则去找球）
    }

    if (this._idleS >= this.config.decisionEveryS && !this._pending) {
      this._idleS = 0;
      this.phase = "thinking";
      this._pending = { settled: false, value: null };
      const pending = this._pending;
      this._decide().then((rec) => { pending.value = rec; pending.settled = true; },
                          (err) => { pending.value = { error: String(err?.message || err) }; pending.settled = true; });
    }
    return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: this.phase };
  }

  /** 一轮决策：抓图 → 组提示词 → 问模型（或规则）→ 过规则层 → 返回记录。 */
  async _decide() {
    const obs = this.buildObservation();
    const image = this.view.captureHeadCam();
    const rec = {
      stepIndex: this.stepIndex,
      simTime: obs.state.simTime,
      task: obs.task, taskId: obs.taskId, target: obs.target,
      state: obs.state, prompt: obs.prompt, image,
      proposed: "", token: "", source: "", raw: "", latencyMs: null,
      rules: [], note: "", error: null, provider: "", model: "",
    };

    let proposed, raw = "", latencyMs = null;
    if (this.config.mode === "llm") {
      if (!this.config.apiKey && !/localhost|127\.0\.0\.1/.test(this.config.baseUrl || "")) {
        return { ...rec, error: "llm 模式需要 API key（BYO key，存在浏览器本地，不会上传到任何服务器）" };
      }
      const out = await askVlm(this.config, { prompt: obs.prompt, imageDataUrl: image });
      raw = out.text;
      latencyMs = out.latencyMs;
      proposed = parseToken(raw, this.task.allowed);
      rec.source = "vlm";
      rec.provider = this.config.provider;
      rec.model = this.config.model;
      if (!proposed) {
        proposed = reflexToken(obs.state);
        rec.note = "parse-failed";
      }
    } else {
      proposed = reflexToken(obs.state);
      rec.source = "rule";
    }

    const ctx = { ...obs.ruleCtx, proposed };
    const ruled = applyRules(ctx, this.rules);
    rec.proposed = proposed;
    rec.token = ruled.token;
    rec.rules = ruled.applied;
    rec.raw = raw;
    rec.latencyMs = latencyMs;
    if (ruled.notes.length) rec.note = (rec.note ? rec.note + "; " : "") + ruled.notes.join("; ");
    else if (!rec.note) rec.note = "ok";
    return rec;
  }

  /** 供测试/脚本用：同步跑完一整轮决策（不经过 tick 的节流）。 */
  async decideOnce() {
    const rec = await this._decide();
    this.records.unshift(rec);
    if (this.records.length > this.maxRecords) this.records.pop();
    return rec;
  }

  status() {
    return {
      phase: this.phase, mode: this.config.mode, task: this.task,
      decisions: this.records.length, stepIndex: this.stepIndex,
      lastNote: this.lastNote, lastError: this.lastError,
      interpreter: this.interpreter.status(),
      lastRecord: this.records[0] || null,
    };
  }
}
