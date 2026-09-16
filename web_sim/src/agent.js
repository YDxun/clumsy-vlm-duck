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
import { StationKicker, Pusher, planStation } from "./station.js";
import { GoalCheck } from "./goal.js";
import { parseSequence, describeSequence, SequenceRunner } from "./sequence.js";

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
    this.pusher = new Pusher();
    // 动作序列（「前进1米，再翻滚一次，最后跳舞」这类没有目标物体的编排）
    this.sequencer = new SequenceRunner({ duck, interpreter: this.interpreter });
    this.sequenceProgress = "";
    this.manipulationMode = "auto";   // auto | push | kick
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
    /**
     * 纯动作序列（「前进1米，再翻滚一次，最后跳舞」）：这类指令**没有目标物体**，
     * 以前会被硬猜成"去找 ball" —— 于是鸭子只会转圈找球，翻滚跳舞永远轮不上。
     * 现在把它交给序列执行器，逐步按世界状态验收（走了几米 / 转了几度 / 技能跑完没）。
     */
    const seqSteps = curated ? [] : parseSequence(taskText);
    // 精选任务的"要操作什么、送到哪"**从它的成功判据里推导**（场景包声明的权威来源）。
    // 以前这里是读顶层 curated.target —— 那个字段在 tasks.yaml 的 schema 里根本不存在，
    // 所以 14 个任务全都在靠措辞推断，只是碰巧推对了。
    const derived = curated && this.scene ? this.scene.taskTarget(curated) : null;
    let resolvedTarget = target;
    let targetSource = target ? "explicit" : "declared";
    let zoneName = derived?.zone || null;
    let zoneRadius = derived?.radius ?? null;
    if (!resolvedTarget && !seqSteps.length) {
      if (derived?.target) {
        resolvedTarget = derived.target;
      } else {
        // 自由文本，或者本来就是"没有目标物体"的任务（走到某点/起身）→ 只能靠措辞
        const hits = this.scene ? this.scene.scanText(taskText) : [];
        resolvedTarget = pickTarget(hits, taskText).body;
        targetSource = derived ? "inferred-no-object" : "inferred";
        const z = this.scene ? this.scene.zoneFor(taskText) : null;
        zoneName = z?.name || zoneName;
      }
    }
    this.task = {
      text: taskText,
      taskId: taskId || curated?.id || "",
      target: seqSteps.length ? "" : (resolvedTarget || "ball"),
      targetSource: seqSteps.length ? "sequence" : targetSource,
      targetReason: derived?.source || null,
      intent: seqSteps.length ? "sequence" : resolveIntent(taskId, taskText),
      sequence: seqSteps.length ? describeSequence(seqSteps) : null,
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
    // 换任务 = 一张白纸：
    //   · records 清空 —— 界面上的日志卡片换任务时清了，次数也得跟着归零，否则两者对不上；
    //   · _pending 作废 —— 上一次"还在飞"的那次决策必须丢掉，不然决策闸门永远关着
    //     （详见 abortPending 的注释）。
    this.records = [];
    this._pending = null;
    this._idleS = 0;
    /**
     * 场景判据执行器：精选任务自带 success 判据，**由世界状态决定成没成**，
     * 不再等模型喊 DONE，也不靠 max_steps 收场。自由文本任务没有判据，
     * goal 为 null，行为与以前一致（靠模型 DONE）。
     */
    this.goal = curated ? new GoalCheck(curated, { duck: this.duck, scene: this.scene }) : null;
    this.goalProgress = "";
    this.sequencer.load(seqSteps);
    this.sequenceProgress = this.sequencer.progress;
    // 站位/判成功用的区域：优先用成功判据里那个（半径以任务为准，比如踢球是 0.30 而不是 0.35）
    const zone = zoneName && this.scene ? this.scene.zoneByName(zoneName) : null;
    this.stationZone = zone ? { ...zone, successRadius: zoneRadius } : null;
    this.station.reset();
    this.pusher.reset();
    // 推还是踢：按任务语义自动选（"推/push/retrieve" → 推；"踢/kick" → 踢），也能手动指定
    const wantsPush = /推|push|retrieve|带来|带走/i.test(taskText + " " + taskId);
    const wantsKick = /踢|kick/i.test(taskText + " " + taskId);
    this.manipMode = this.manipulationMode === "auto"
      ? (wantsKick && !wantsPush ? "kick" : "push")
      : this.manipulationMode;
    // 滚动物体要短促得多：实测球被"推 0.25 m"之后自己滚了 1.9 m（放大 7.6 倍），
    // 所以球用 25 拍的短脉冲，方块这类不滚的用 90 拍。
    const objMeta = (this.scene?.metadata?.objects || []).find((o) => o.body === this.task.target);
    this.manipObjectKind = objMeta?.kind || "unknown";
    return this.task;
  }

  /**
   * 作废「正在飞的那次决策」—— 点「停止决策」「复位」时调用。
   *
   * `_pending` 是"同一时刻只允许一次决策"的闸门，而**只有 tick() 的 thinking 分支
   * 会消费它**。所以外部一旦把 phase 改掉（停止/复位/换任务），那个已经落地或者还在
   * 飞的结果就再也没人取走，闸门从此关着：再点开始，鸭子站着不动、一次推理都不发，
   * 阶段停在最后那个状态 —— 用户实际踩到的就是这个（VLM 模式下一个请求要 1~3 s，
   * 很容易在它飞的时候按停止）。
   */
  abortPending() {
    this._pending = null;
    this._idleS = 0;
    if (this.phase === "thinking") this.phase = "idle";
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
    const fallen = this.duck.upright() < 0.55;
    const needObs = !this.station.latched ||
                     (this.duck.steps - (this._stationObsStep ?? -999)) > 60 ||
                     // 摔倒了也要继续偶尔看一眼：踢完球往往前扑，这时候球可能正好滚进区域，
                     // 不看了就等于"事情做成了但没认账"（实测就是这样）。
                     (fallen && (this.duck.steps - (this._stationObsStep ?? -999)) > 50);
    let ballXy = null;
    if (needObs) {
      this._stationObsStep = this.duck.steps;
      const s = this.sensor.snapshot(this.task.target, this.duck.steps * 0.02);
      if (s.targetVisible && s.targetWorldXyz) {
        ballXy = { x: s.targetWorldXyz[0], y: s.targetWorldXyz[1] };
        // 球已经进区 → 任务完成。**这一步必须在摔倒判断之前**：
        // 鸭子踢完往前扑倒是常事，不能因为摔了就假装任务没成。
        if (StationKicker.ballInZone(ballXy, this.stationZone)) {
          this.station.note = `球已进区 (${ballXy.x.toFixed(2)},${ballXy.y.toFixed(2)})`;
          return { cmd: [0, 0, 0], kick: false, phase: "done", note: this.station.note, done: true };
        }
        if (this.station.lastKick) {          // 刚踢过：用这一脚的真实落点标定偏角
          const k = this.station.lastKick.ball;
          const moved = Math.hypot(ballXy.x - k.x, ballXy.y - k.y);
          if (moved < 0.05) {
            // 这一脚没碰到球 → 换一个站位再来，直到踢到为止
            this.stationDither = (this.stationDither || 0) + 1;
            this.station.ditherIndex = this.stationDither;
            this.station.lastKick = null;
            this.lastNote = `这一脚没碰到球（位移 ${moved.toFixed(3)} m），换站位重试（第 ${this.stationDither} 次）`;
            if (this.stationDither > 6) { this.lastNote += " —— 抖动表用完了，仍然踢不到"; }
          } else {
            const bias = this.station.calibrate(ballXy);
            this.stationDither = 0;
            this.station.ditherIndex = 0;
            if (bias !== null) {
              this.lastNote = `踢到了（球走了 ${moved.toFixed(2)} m），标定出手偏角 ${(bias * 180 / Math.PI).toFixed(0)}°`;
            }
          }
        }
      }
    }
    // 摔倒了就先别走：躺着继续下方位移指令只会在地上蹭
    if (fallen) {
      this.station.note = "鸭子倒了，先停住（起身策略还没接入）";
      return { cmd: [0, 0, 0], kick: false, phase: "fallen", note: this.station.note };
    }
    const q = this.duck.data.qvel;
    return this.station.tick({
      duckPose: this.duck.pose, ballXy, zone: this.stationZone,
      motion: { speed: Math.hypot(q[0], q[1]), gyro: this.duck.angularSpeed() },
    });
  }

  /** 推东西：见物体才喂观测，其余靠锁存的计划推进（棘轮式：推几轮退回来重看一眼）。 */
  pusherTick() {
    // 滚动物体（球）用短脉冲，不滚的（方块/杯子）用长脉冲
    this.pusher.burstTicks = this.manipObjectKind === "sphere" ? 25 : 90;
    if (this.duck.upright() < 0.55) {
      this.pusher.note = "鸭子倒了，先停住";
      return { cmd: [0, 0, 0], phase: "fallen", note: this.pusher.note, done: false };
    }
    const needObs = !this.pusher.latched || this.pusher.needObserve;
    let objXy = null;
    if (needObs && (this.duck.steps - (this._pushObsStep ?? -999)) > 25) {
      this._pushObsStep = this.duck.steps;
      const s = this.sensor.snapshot(this.task.target, this.duck.steps * 0.02);
      if (s.targetVisible && s.targetWorldXyz) {
        objXy = { x: s.targetWorldXyz[0], y: s.targetWorldXyz[1] };
      }
    }
    return this.pusher.tick({ duckPose: this.duck.pose, objXy, zone: this.stationZone });
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
      // 已经收工了就把闸门放开，别把"还在飞/已落地但没人消费"的那次决策一直挂着
      this._pending = null;
      return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: this.phase };
    }
    // 场景判据说了算：世界状态满足 + 保持够 success_hold_s → 当场收工。
    // 不等模型喊 DONE（它可能正低头看着地板，也可能已经摔了但球其实进区了）。
    if (this.goal?.enabled) {
      const g = this.goal.step(dt);
      this.goalProgress = g.progress;
      if (g.done) {
        this.phase = "finished";
        this.lastNote = `任务完成（场景判据）：${this.task.taskId || this.task.text}`;
        this.interpreter.cancel();
        return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "finished" };
      }
      // 位置已经到位、只差"停稳"（判据里有速度条件）→ 主动刹住。
      // 不刹的话鸭子会自己走出区域，"进了区域"这一条反而永远满足不了。
      if (g.positional) {
        this.interpreter.cancel();
        this.lastNote = "已到位，停稳中…";
        return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: this.phase };
      }
    }
    /**
     * 动作序列：不需要视觉、也不需要模型逐步推理 —— 第几步做没做完由**世界状态**验收
     * （走了几米、转了几度、技能跑完没有）。所以这一段完全不走决策层（也就不用花钱调模型）。
     */
    if (this.sequencer.active) {
      const s = this.sequencer.tick(dt);
      this.sequenceProgress = s.note;
      this.lastNote = s.note;
      if (s.advanced) {                    // 让"最近动作"里能看到序列真的做到哪了
        this.history.unshift(s.advanced);
        if (this.history.length > 20) this.history.pop();
      }
      if (s.done) {
        this.phase = "finished";
        return { cmd: s.command, headDelta: s.headDelta, phase: "finished" };
      }
      return { cmd: s.command, headDelta: s.headDelta, phase: "sequence" };
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
          // 场景判据优先：世界状态还没达成就不认这个 DONE（模型/规则层可能提前喊），
          // 当作"再走一步"，继续决策 —— 免得出现"界面说完成、任务判据却没过"。
          if (this.goal?.enabled && !this.goal.lastOk) {
            this.phase = "idle";
            this._idleS = 0;
            this.lastNote = `说完成但判据未满足：${this.goalProgress}`;
            if (this.onRecord) this.onRecord(rec);
            return { cmd: [0, 0, 0], headDelta: this.interpreter.headOverride(), phase: "idle" };
          }
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
        // 注意别调 station.reset()：那会把 lastKick 清掉，而"这一脚有没有碰到球"
        // 正要靠 lastKick 和下一次观测去判断（踢空就换站位重试）。
        this.station.kicked = false;
        this.station.latched = null;
        this.station.wpIndex = 0;
        this.station.needObserve = false;
        this.station.settleLeft = this.station.settleTicks;
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
      // —— 推东西：宽松得多，先走这条
      if (this.manipMode === "push") {
        const ps = this.pusherTick();
        this.lastNote = ps.note;
        if (ps.done) {
          this.phase = "finished";
          this.lastNote = `任务完成：${ps.note}`;
          return { cmd: [0, 0, 0], headDelta: null, phase: "finished" };
        }
        if (ps.phase !== "no-plan") {
          return { cmd: ps.cmd, headDelta: null, phase: "push:" + ps.phase };
        }
        // 还没看见物体 → 掉回正常决策去找
      } else {
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
        proposed = reflexToken(obs.state, { stopRangeM: this.goal?.arriveRangeM ?? 0.35 });
        rec.note = "parse-failed";
      }
    } else {
      proposed = reflexToken(obs.state, { stopRangeM: this.goal?.arriveRangeM ?? 0.35 });
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
      goal: this.goal?.enabled ? { progress: this.goalProgress, holdS: this.goal.hold } : null,
      sequence: this.sequencer.active ? { steps: this.sequencer.steps.length,
                                          progress: this.sequenceProgress } : null,
    };
  }
}
