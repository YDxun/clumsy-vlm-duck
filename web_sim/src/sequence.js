/**
 * 动作序列（编排）—— 把「前进1米，再翻滚一次，最后跳舞」这种**没有目标物体**的
 * 指令翻译成一串可执行、可验收的步骤，自己一步步跑完。
 *
 * 为什么需要它：以前这类指令会被当成"去找某个物体"（目标默认退化成 ball），
 * 于是鸭子要么一直转圈找球、要么走到球边就喊完成，翻滚和跳舞**永远不会被执行**。
 * 这类指令跟"看见目标 → 走过去"不是一回事：它的正确性由**做了多少动作**决定，
 * 不需要视觉，也不需要模型逐步推理。
 *
 * 每一步都由世界状态验收（走了多少米 / 转了多少度 / 技能有没有跑完），
 * 所以"前进 1 米"是真的量出来的 1 米，而不是"跑 10 个 FWD 就宣布完事"。
 */

const CN_NUM = { 零: 0, 一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };

/** 中文/阿拉伯数字 → 数值。支持 "1.5 / 二十五 / 十二 / 半"。 */
export function parseNumber(raw) {
  const s = String(raw ?? "").trim();
  if (!s) return null;
  if (/^\d+(\.\d+)?$/.test(s)) return Number(s);
  if (s === "半") return 0.5;
  if (/^十[一二三四五六七八九]?$/.test(s)) return 10 + (CN_NUM[s[1]] ?? 0);
  if (/^[一二两三四五六七八九]十[一二三四五六七八九]?$/.test(s)) {
    return CN_NUM[s[0]] * 10 + (s[2] ? CN_NUM[s[2]] : 0);
  }
  if (s.length === 1 && CN_NUM[s] != null) return CN_NUM[s];
  return null;
}

const NUM = "([0-9]+(?:\\.[0-9]+)?|[一二两三四五六七八九十]+|半)";
const SKILL_WORDS = [
  [/(翻滚|翻个跟头|打个滚|翻一下|roll)/i, "ROLL", "翻滚"],
  [/(坐下|坐一下|sit)/i, "SIT", "坐下"],
  [/(起身|站起来|起立|stand ?up)/i, "STAND_UP", "起身"],
  [/(低头|看地面|look ?down)/i, "LOOK_DOWN", "低头"],
  [/(抬头|回正|head ?center)/i, "HEAD_CENTER", "抬头"],
];

/**
 * 认识、但**目前不能用**的动作。
 *
 * 「跳舞」曾经映射到 `happy_hop`，实测它会把鸭子一点点放倒到侧躺、策略结束后
 * 彻底摔平且起不来 —— 用户看到的就是"跳舞怎么变成侧滚翻了，后面还站不起来"。
 * 与其糊弄，不如明确告诉用户这一步被跳过（序列其余部分照跑）。
 */
const UNSUPPORTED_WORDS = [
  [/(跳舞|跳个舞|舞一支|dance)/i, "跳舞（happy_hop 会把鸭子放倒，已停用）"],
];

/**
 * 把一句自然语言拆成动作步骤。
 * @returns {Array<{kind:string}>} 空数组 = 这不是一段纯动作序列
 */
export function parseSequence(text) {
  const src = String(text || "").trim();
  if (!src) return [];
  // 带"物体/位置目标"的指令不算序列：那是导航任务，交给决策层
  const hasObjectTarget = /(去|到|找|拿|推|踢|搬|送到|放进|过去|go to|reach|find|push|kick)/i.test(src);
  if (hasObjectTarget && !/(前进|后退|左转|右转|翻滚|跳舞)/.test(src)) return [];

  // 先按"然后 / 再 / 最后 / 逗号"切开
  const parts = src.split(/[，,、;；]|然后|接着|再|之后|最后|跟着|\+\s*/)
    .map((s) => s.trim()).filter(Boolean);
  const steps = [];
  for (const part of parts) {
    const times = (() => {
      const m = new RegExp(NUM + "\\s*(?:次|遍|回|下)").exec(part);
      const n = m ? parseNumber(m[1]) : null;
      return n && n > 0 ? Math.min(10, Math.round(n)) : 1;
    })();
    // 1) 平移
    let m = new RegExp("(前进|向前走|往前走|往前|向前|直走|go forward|forward)\\s*" + NUM + "\\s*(米|m)?", "i").exec(part);
    if (m) {
      const d = parseNumber(m[2]);
      if (d > 0) { steps.push({ kind: "move", meters: d, label: `前进 ${d} m` }); continue; }
    }
    m = new RegExp("(后退|往后退|向后走|back)\\s*" + NUM + "\\s*(米|m)?", "i").exec(part);
    if (m) {
      const d = parseNumber(m[2]);
      if (d > 0) { steps.push({ kind: "move", meters: -d, label: `后退 ${d} m` }); continue; }
    }
    // 2) 转向（没写角度就按 90 度）
    m = new RegExp("(左转|向左转|左拐|turn left|left)\\s*" + NUM + "?\\s*(度|°|degrees?)?", "i").exec(part);
    if (m) {
      const deg = parseNumber(m[2]) ?? 90;
      steps.push({ kind: "turn", deg, label: `左转 ${deg}°` });
      continue;
    }
    m = new RegExp("(右转|向右转|右拐|turn right|right)\\s*" + NUM + "?\\s*(度|°|degrees?)?", "i").exec(part);
    if (m) {
      const deg = parseNumber(m[2]) ?? 90;
      steps.push({ kind: "turn", deg: -deg, label: `右转 ${deg}°` });
      continue;
    }
    // 3) 技能
    const skill = SKILL_WORDS.find(([re]) => re.test(part));
    if (skill) {
      for (let i = 0; i < times; i++) steps.push({ kind: "skill", token: skill[1], label: skill[2] });
      continue;
    }
    // 3.5) 认识但暂时用不了的动作：记成一步，跑到它时跳过并说明原因
    const unsupported = UNSUPPORTED_WORDS.find(([re]) => re.test(part));
    if (unsupported) {
      steps.push({ kind: "unsupported", label: unsupported[1] });
      continue;
    }
    // 4) 停住
    if (/(停住|停下|站住|别动|stop)/i.test(part)) { steps.push({ kind: "stop", label: "停住" }); continue; }
    // 不认识的片段：整句放弃，别猜
    return [];
  }
  return steps;
}

/** 一句话描述一串步骤（界面/日志用）。 */
export function describeSequence(steps) {
  return (steps || []).map((s) => s.label).join(" → ");
}

/**
 * 逐步执行。每步都用**世界状态**验收：走了几米、转了几度、技能跑没跑完。
 */
export class SequenceRunner {
  constructor({ duck, interpreter }) {
    this.duck = duck;
    this.interpreter = interpreter;
    this.steps = [];
    this.reset();
  }

  load(steps) {
    this.steps = steps || [];
    this.reset();
  }

  reset() {
    this.index = 0;
    this.pulses = 0;
    this.settled = false;          // 序列跑完 + 确认站着，才算真的完成
    this.recoverTries = 0;
    this.startPose = null;
    this.startHeading = null;
    this.stepStarted = false;
    this.stepStartedRun = false;
    this.lastNote = this.steps.length ? `准备执行：${describeSequence(this.steps)}` : "";
  }

  get active() { return this.steps.length > 0; }
  get done() { return this.active && this.index >= this.steps.length && this.settled; }
  get current() { return this.active ? this.steps[this.index] || null : null; }

  get progress() {
    if (!this.active) return "";
    if (this.settled) return `全部完成：${describeSequence(this.steps)}`;
    if (this.index >= this.steps.length) return "正在确认站姿…";
    return `第 ${this.index + 1}/${this.steps.length} 步：${this.steps[this.index].label}`;
  }

  /** 一个控制步。返回 {command, headDelta, done, phase, note}；没在跑序列时返回 null。 */
  tick(dt) {
    if (!this.active) return null;
    if (this.settled) return this._finish("");
    if (this.index >= this.steps.length) return this._settleTick(dt);
    const step = this.steps[this.index];
    if (!this.stepStarted) this._begin();

    if (step.kind === "move") {
      if (!this.interpreter.busy && this._travelled() < Math.abs(step.meters) - 0.03 && this.pulses < 40) {
        this.interpreter.start(step.meters > 0 ? "FWD" : "BACK");
        this.pulses += 1;
      }
      if (this._travelled() >= Math.abs(step.meters) - 0.03 || this.pulses >= 40) {
        this.interpreter.cancel();
        return this._advance(`${step.label}（实测 ${this._travelled().toFixed(2)} m / ${this.pulses} 步）`,
                             step.meters > 0 ? "FWD" : "BACK");
      }
    } else if (step.kind === "turn") {
      if (!this.interpreter.busy && this._turned() < Math.abs(step.deg) - 12 && this.pulses < 20) {
        this.interpreter.start(step.deg > 0 ? "TURN_L" : "TURN_R");
        this.pulses += 1;
      }
      if (this._turned() >= Math.abs(step.deg) - 12 || this.pulses >= 20) {
        this.interpreter.cancel();
        return this._advance(`${step.label}（实测 ${this._turned().toFixed(0)}°）`,
                             step.deg > 0 ? "TURN_L" : "TURN_R");
      }
    } else if (step.kind === "skill") {
      if (!this.stepStartedRun) { this.interpreter.start(step.token); this.stepStartedRun = true; }
      if (!this.interpreter.busy) return this._advance(step.label, step.token);
    } else if (step.kind === "stop") {
      this.interpreter.cancel();
      return this._advance("停住", "STOP");
    } else if (step.kind === "unsupported") {
      // 认识、但现在不能用的动作：跳过，并把原因写清楚（其余步骤照跑）
      this.interpreter.cancel();
      return this._advance(`跳过 ${step.label}`, "");
    }
    // 关键：把解释器这一步的**速度指令**真的下发出去。
    // 以前这里写死 [0,0,0]，于是解释器永远推进不了、鸭子原地不动，
    // 界面还一直显示"第 1/3 步：前进 1 m"。
    const out = this.interpreter.tick(dt);
    return { command: out.command, headDelta: this.interpreter.headOverride(),
             done: false, phase: "sequence", note: this.progress };
  }

  /**
   * 收尾：确认鸭子是**站着**的再宣布完成。
   *
   * 实测 `happy_hop`（原本的"跳舞"）会把鸭子一点点放倒，`alpha_standup` 起不来；
   * 唯一稳定有效的是 roulade —— 3/3 次、约 4~5 s 翻回站立（末态 upright 0.95）。
   * 所以序列结束时只要鸭子是躺着的，就用翻滚策略翻回来再收工，
   * 免得用户看到"任务完成"四个字，鸭子却躺在地上。
   */
  _settleTick(dt) {
    const up = this.duck.upright();
    if (up >= 0.6) return this._finish("");
    if (this.recoverTries >= 6) {
      return this._finish(`，但没能自己站起来（upright ${up.toFixed(2)}，可再点一次「翻滚」）`);
    }
    if (!this.interpreter.busy) {
      this.interpreter.start("ROLL");
      this.recoverTries += 1;
    }
    this.interpreter.tick(dt);
    return { command: [0, 0, 0], headDelta: this.interpreter.headOverride(), done: false,
             phase: "sequence",
             note: `摔倒恢复中：翻滚第 ${this.recoverTries} 次（upright ${up.toFixed(2)}）` };
  }

  _finish(extra) {
    this.settled = true;
    return { command: [0, 0, 0], headDelta: this.interpreter.headOverride(), done: true,
             phase: "finished",
             note: `动作序列完成：${describeSequence(this.steps)}${extra}` };
  }

  _begin() {
    this.stepStarted = true;
    this.stepStartedRun = false;
    this.pulses = 0;
    const p = this.duck.pose;
    this.startPose = { x: p.x, y: p.y };
    this.startHeading = p.heading;
  }

  _advance(note, token = "") {
    this.index += 1;
    this.stepStarted = false;
    this.startPose = null;
    this.startHeading = null;
    const tail = this.index >= this.steps.length ? "正在确认站姿…"
                                                 : `下一步：${this.steps[this.index].label}`;
    return { command: [0, 0, 0], headDelta: this.interpreter.headOverride(),
             done: false, phase: "sequence", advanced: token, note: `${note}｜${tail}` };
  }

  /** 这一步已经走了多远（直线距离）。 */
  _travelled() {
    if (!this.startPose) return 0;
    const p = this.duck.pose;
    return Math.hypot(p.x - this.startPose.x, p.y - this.startPose.y);
  }

  /** 这一步已经转了多少度（绝对值，处理 ±180° 环绕）。 */
  _turned() {
    if (this.startHeading == null) return 0;
    const d = (this.duck.pose.heading - this.startHeading) * 180 / Math.PI;
    return Math.abs(((d + 180) % 360 + 360) % 360 - 180);
  }
}
