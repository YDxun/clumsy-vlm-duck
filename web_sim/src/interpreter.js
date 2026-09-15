/**
 * 把动作 token 翻译成具体运动 —— `duck_vlm/interpreter.py` 的浏览器版（去掉技能态机）。
 *
 * 一次只跑一个 token：motion 类按固定时长发同一组速度指令然后结束；
 * head 类保持站立、把头部关节顶到固定偏移，**结束后仍然锁住**（latch）——
 * 这样鸭子低头之后不会因为走路策略的头部目标而自动抬回来，
 * VLM 必须显式发 HEAD_CENTER 才会回正。
 */
import { ACTION_SPECS, TOKENS } from "./actions.js";

export class ActionInterpreter {
  constructor() {
    this.token = null;
    this.headDelta = null;     // 锁存的头部关节偏移；null = 跟随走路策略
    this.elapsed = 0;
    this.duration = 0;
    this.command = [0, 0, 0];
    this.doneCount = 0;
    /** 技能 token 需要切 ONNX 策略，由外部注入（浏览器里是 DuckSim.setPolicy）。 */
    this.policySwitch = null;
  }

  get busy() { return this.token !== null; }

  /** 丢弃当前动作。headDelta 是刻意保留的，见类注释。 */
  cancel() {
    const was = this.token;
    this.token = null;
    return was;
  }

  /** 让头部回到中立位（换场景 / 重置时用）。 */
  resetHead() { this.headDelta = null; }

  /**
   * 开始执行一个 token。
   * @returns {{command:number[], done:boolean, note:string, token:string}}
   */
  start(rawToken) {
    const token = String(rawToken || "").toUpperCase().trim();
    const spec = ACTION_SPECS[token];
    if (!spec) return { command: [0, 0, 0], done: true, note: `unknown token ${token}`, token };

    this.token = token;
    this.elapsed = 0;
    if (spec.terminal) {
      this.token = null;
      this.doneCount += 1;
      return { command: [0, 0, 0], done: true, note: "task marked done", token };
    }
    if (spec.kind === "skill") {
      this.duration = Math.max(0.10, spec.durationS);
      this.command = [0, 0, 0];
      this.skill = token;
      // 技能策略自己会动，别让头部姿态覆盖它（走路策略的头部约束不适用）
      this.headDelta = null;
      if (this.policySwitch) this.policySwitch(spec.policy, token);
      return { command: [0, 0, 0], done: false, note: "skill started", token };
    }
    if (spec.kind === "head") {
      this.duration = Math.max(0.10, spec.durationS);
      this.headDelta = Object.fromEntries(spec.headDelta || []);
      this.command = [0, 0, 0];
      return { command: [0, 0, 0], done: false, note: "head pose requested", token };
    }
    this.duration = Math.max(0.10, spec.durationS);
    this.command = [...spec.command];
    return { command: [...this.command], done: false, note: "motion started", token };
  }

  /** 推进一步；返回当前该发的速度指令与是否跑完。 */
  tick(dt) {
    if (this.token === null) return { command: [0, 0, 0], done: true, note: "idle", token: "" };
    const token = this.token;
    this.elapsed += Math.max(0, Number(dt) || 0);
    if (this.elapsed >= this.duration - 1e-9) {
      this.token = null;
      const kind = ACTION_SPECS[token].kind;
      this.doneCount += 1;
      if (kind === "skill") {
        this.skill = null;
        // 技能跑完一定要切回走路，否则鸭子会一直用它、站着不动
        if (this.policySwitch) this.policySwitch("alpha_walking", null);
        return { command: [0, 0, 0], done: true, note: "skill completed", token };
      }
      return { command: [0, 0, 0], done: true,
               note: kind === "head" ? "head pose completed" : "motion completed", token };
    }
    return { command: [...this.command], done: false,
             note: ACTION_SPECS[token].kind === "head" ? "head pose running" : "motion running",
             token };
  }

  /** 当前该叠加到 ctrl 上的头部关节偏移（null = 不干预）。 */
  headOverride() { return this.headDelta; }

  availableTokens() { return [...TOKENS]; }

  status() {
    return { token: this.token, head_delta: this.headDelta, elapsed: this.elapsed,
             duration: this.duration, command: this.command, done_count: this.doneCount };
  }
}
