/**
 * 场景包的**成功判据** → 「这一局到底成没成」，由世界状态说了算。
 *
 * 以前网页端只用 success 判据来**推导目标**（scene.js 的 taskTarget），判完不判成：
 * 任务什么时候结束全看模型什么时候喊 DONE，或者撞上 max_steps。用户看到的
 * 「明明进了红色区域还要等半天才 finish」「球其实已经进区了却不算完成」都是这么来的。
 *
 * 这里把 `duck_scenes/validation_core.py` 的原子判据语义搬到浏览器端，用**同一套**判据：
 *   distance_xy_le/ge/range · body_in_zone · upright · speed_xy_le · robot_pose_region ·
 *   ordered_zone_sequence · ever_fallen · hold_condition_s · all/any 嵌套
 * 满足后再持续 `success_hold_s` 秒（默认 0.5 s）才判成 —— 和 Python 端验证器一致，
 * 免得"路过一下"就算过。
 */

const DEG = 180 / Math.PI;

const angleDiffDeg = (a, b) => Math.abs(((a - b + 180) % 360 + 360) % 360 - 180);

export class GoalCheck {
  /**
   * @param {object} task  场景包里的精选任务（带 success / success_hold_s）
   * @param {{duck:object, scene:object}} deps
   */
  constructor(task, { duck, scene } = {}) {
    this.duck = duck;
    this.scene = scene;
    this.robotBody = scene?.metadata?.robot?.body_name || scene?.metadata?.robot?.root_body || "trunk_base";
    this.success = task?.success || null;
    this.holdS = Number(task?.success_hold_s ?? 0.5);
    this.arriveRangeM = this._arrival();
    this.reset();
  }

  /**
   * 判据里的"到达半径"（米）：规则层的 DONE 阈值要用它，不能再用写死的 0.35。
   * 取所有跟机器人自己有关的 distance_xy_le / body_in_zone / robot_pose_region 里
   * 最小的那个半径 —— 那就是"到了"的真实标准。
   */
  _arrival() {
    const vals = [];
    for (const p of this.success?.all || []) {
      if (p?.type === "body_in_zone" && this._isRobot(p.body)) vals.push(Number(p.radius_m));
      else if (p?.type === "distance_xy_le" && this._isRobot(p.body)) vals.push(Number(p.distance_m));
      else if (p?.type === "robot_pose_region" && this._isRobot(p.body)) vals.push(Number(p.radius_m));
    }
    const ok = vals.filter((v) => Number.isFinite(v) && v > 0);
    return ok.length ? Math.min(...ok) : null;
  }

  reset() {
    this.timers = new Map();      // 判据路径 → 已连续满足的秒数
    this.latched = new Set();     // ever_fallen 这类一次性闩锁
    this.sequenceIndex = 0;
    this.hold = 0;
    this.lastOk = false;
  }

  get enabled() { return !!this.success; }

  /** 一个控制步调用一次；返回 {done, ok, holdS, progress}。 */
  step(dt) {
    if (!this.success) return { done: false, ok: false, positional: false, holdS: 0, progress: "" };
    const kids = this.success.all || null;
    let ok, positional;
    if (kids) {
      const states = kids.map((c, i) => this._eval(c, `root.${i}`, dt));
      ok = states.every(Boolean);
      /**
       * positional = 「除了速度条件以外都满足」。
       *
       * 为什么需要它：很多判据是 `位姿 + speed_xy_le` 的组合（"走到 X 并**停住**"）。
       * 如果只按 ok 判，鸭子在区域里还在走 → 速度条件不成立 → 判据永远不满足 →
       * 它就一直往前走穿出区域，反而永远完成不了。有了 positional，决策层就能
       * 在"位置到了、只差停稳"的时候主动刹住（agent.tick 里那段）。
       */
      positional = kids.every((c, i) => (c?.type === "speed_xy_le" ? true : states[i]));
    } else {
      ok = this._eval(this.success, "root", dt);
      positional = ok;
    }
    this.hold = ok ? this.hold + dt : 0;
    this.lastOk = ok;
    return {
      done: ok && this.hold >= this.holdS - 1e-9,
      ok, positional, holdS: this.hold,
      progress: this._progress(),
    };
  }

  /**
   * 人话版进度：给界面和提示词用。
   * 只描述**还没满足**的那几条，全满足时说"已达成，保持中"。
   */
  _progress() {
    if (!this.success) return "";
    const items = this.success.all || this.success.any || [];
    const pending = items.filter((p) => !this._eval(p, "peek", 0));
    const show = (pending.length ? pending : items).slice(0, 2).map((p) => this._describe(p));
    const head = pending.length ? "未达成" : "已达成";
    return `${head}：${show.join("；")}`;
  }

  _describe(p) {
    const f2 = (v) => (v == null ? "?" : v.toFixed(2));
    switch (p.type) {
      case "body_in_zone": {
        const a = this._pos(p.body), b = this._pos(p.zone);
        if (!a || !b) return `${p.body}→${p.zone}（找不到对象）`;
        return `${p.body} 距 ${p.zone} ${f2(Math.hypot(a.x - b.x, a.y - b.y))}/${p.radius_m} m`;
      }
      case "distance_xy_le":
      case "distance_xy_ge": {
        const a = this._pos(p.body), b = this._pos(p.target);
        if (!a || !b) return `${p.body}→${p.target}（找不到对象）`;
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        const want = p.type === "distance_xy_le" ? `≤ ${p.distance_m}` : `≥ ${p.distance_m}`;
        return `${p.body} 距 ${p.target} ${f2(d)} m（需 ${want}）`;
      }
      case "distance_xy_range": {
        const a = this._pos(p.body), b = this._pos(p.target);
        if (!a || !b) return `${p.body}→${p.target}（找不到对象）`;
        return `${p.body} 距 ${p.target} ${f2(Math.hypot(a.x - b.x, a.y - b.y))} m（需 ${p.min_m}~${p.max_m}）`;
      }
      case "robot_pose_region": {
        const a = this._pos(p.body);
        if (!a) return "找不到机器人";
        const d = Math.hypot(a.x - p.center_xy[0], a.y - p.center_xy[1]);
        const de = angleDiffDeg(this._yaw(p.body) * DEG, p.yaw_deg);
        return `距停靠点 ${f2(d)}/${p.radius_m} m，朝向差 ${de.toFixed(0)}°（容差 ${p.yaw_tolerance_deg}°）`;
      }
      case "speed_xy_le":
        return `速度 ${f2(this._speed(p.body))} ≤ ${p.speed_mps} m/s`;
      case "upright":
        return `直立度 ${f2(this._upright(p.body))} ≥ ${p.min_upright}`;
      case "ordered_zone_sequence":
        return `按顺序访区域：还差 ${Math.max(0, (p.zones || []).length - this.sequenceIndex)} 个`;
      case "ever_fallen":
        return this.latched.has("ever_fallen") ? "已经摔过了" : "还没摔倒（这个任务要求先摔一下）";
      case "hold_condition_s":
        return `保持 ${f2(this.timers.get("hold") || 0)}/${p.seconds} s`;
      case "all":
      case "any":
        return `${p.type === "all" ? "全部" : "任一"}条件`;
      default:
        return p.type;
    }
  }

  // ------------------------------------------------------------ 求值

  _eval(node, key, dt) {
    if (!node) return false;
    if (Array.isArray(node)) return node.every((n, i) => this._eval(n, `${key}.${i}`, dt));
    if (node.all) {
      const kids = node.all;
      // 和 Python 一样：hold_condition_s 只负责计时，不参与"此刻是否成立"的与运算
      const now = kids.every((c, i) => (c?.type === "hold_condition_s" ? true : this._eval(c, `${key}.${i}`, dt)));
      let holdsOk = true;
      kids.forEach((c, i) => {
        if (c?.type !== "hold_condition_s") return;
        const k = `${key}.${i}`;
        // 与 Python 的 timer() 一致：满足时累加，不满足时**保持不动**（不是清零）
        this.timers.set(k, (this.timers.get(k) || 0) + (now ? dt : 0));
        if (this.timers.get(k) < Number(c.seconds) - 1e-9) holdsOk = false;
      });
      return now && holdsOk;
    }
    if (node.any) return node.any.some((c, i) => this._eval(c, `${key}.${i}`, dt));
    return this._atomic(node, key, dt);
  }

  _atomic(n, key, dt) {
    switch (n.type) {
      case "distance_xy_le": return this._dist(n.body, n.target) <= Number(n.distance_m);
      case "distance_xy_ge": return this._dist(n.body, n.target) >= Number(n.distance_m);
      case "distance_xy_range": {
        const d = this._dist(n.body, n.target);
        return d >= Number(n.min_m) && d <= Number(n.max_m);
      }
      case "body_in_zone": return this._dist(n.body, n.zone) <= Number(n.radius_m);
      case "upright": return (this._upright(n.body) ?? -1) >= Number(n.min_upright);
      case "speed_xy_le": return (this._speed(n.body) ?? 1e9) <= Number(n.speed_mps);
      case "robot_pose_region": {
        const p = this._pos(n.body);
        if (!p) return false;
        const d = Math.hypot(p.x - n.center_xy[0], p.y - n.center_xy[1]);
        if (d > Number(n.radius_m)) return false;
        return angleDiffDeg(this._yaw(n.body) * DEG, Number(n.yaw_deg)) <= Number(n.yaw_tolerance_deg);
      }
      case "ever_fallen": {
        const up = this._upright(n.body);
        if (up != null && up < Number(n.max_upright ?? 0.5)) this.latched.add("ever_fallen");
        return this.latched.has("ever_fallen");
      }
      case "ordered_zone_sequence": {
        const zones = n.zones || [];
        const p = this._pos(n.body);
        if (!p) return false;
        if (this.sequenceIndex < zones.length) {
          const z = this._pos(zones[this.sequenceIndex]);
          if (z && Math.hypot(p.x - z.x, p.y - z.y) <= Number(n.radius_m)) this.sequenceIndex += 1;
        }
        return this.sequenceIndex >= zones.length;
      }
      case "hold_condition_s": {
        // 单独出现时（不在 all 里）也要能计时
        const t = this.timers.get(key) || 0;
        this.timers.set(key, t + dt);
        return this.timers.get(key) >= Number(n.seconds) - 1e-9;
      }
      default:
        // 不认识的判据不要假装成功：宁可判不成，也不能骗用户说任务完成了
        return false;
    }
  }

  // ------------------------------------------------------------ 取世界状态

  _isRobot(body) {
    return body === this.robotBody || body === this.scene?.metadata?.robot?.root_body ||
           body === "trunk_base" || body === "robot";
  }

  _pos(name) {
    if (!name) return null;
    if (this._isRobot(name)) {
      const p = this.duck?.pose;
      return p ? { x: p.x, y: p.y, z: p.z } : null;
    }
    const p = this.duck?.bodyPose?.(name);
    if (p) return p;
    const z = this.scene?.zoneByName?.(name);      // 区域是静态 body，直接用元数据里的圆心
    return z ? { x: z.x, y: z.y, z: 0 } : null;
  }

  _dist(a, b) {
    const p = this._pos(a), q = this._pos(b);
    if (!p || !q) return Infinity;
    return Math.hypot(p.x - q.x, p.y - q.y);
  }

  _yaw(name) {
    if (this._isRobot(name)) return this.duck?.pose?.heading ?? 0;
    return 0;
  }

  _upright(name) {
    if (this._isRobot(name)) return this.duck?.upright?.() ?? null;
    return null;
  }

  _speed(name) {
    return this.duck?.bodySpeedXY?.(name) ?? null;
  }
}
