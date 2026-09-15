/**
 * 踢球站位状态机 —— 把 `_dev_cockpit/local_kick.py` 那套「站稳 → 站位 → 进动作 → 恢复」
 * 搬过来，并按实测的踢球特性重写。
 *
 * 为什么必须有它：训练代码里写得很清楚（`microduck_ball_kick_env_cfg.py`）——
 *   > The policy is BLIND to the ball … the operator aims the robot at the ball.
 *   > 球出生在踢球脚前面：机器人偏航系 (x=0.09, |y|=0.042)，随机化只有 ±0.015 m
 * 踢球策略自己不知道球在哪，全靠外部把它摆到那个位置。容差 1.5 cm，
 * 交给 VLM 用 0.07 m 一步的 token 去对是做不到的 —— 这一段必须用规则闭环。
 *
 * 实测（3 次完全一致）：
 *   · 站位精确时球飞 1.14 m，方向**始终是鸭头朝向 -33°**（右脚）
 *   · 所以瞄准式是：鸭头朝向 = 球→目标方向 + 33°
 */

const DEG = Math.PI / 180;
const wrap = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };

/** 训练代码里的球心名义偏移（机器人偏航系）。右脚 -y，左脚 +y。 */
export const BALL_OFFSET_X = 0.09;
export const BALL_OFFSET_ABS_Y = 0.042;
/** 球飞出去的方向相对鸭头朝向的偏角（右脚实测 -33°）。 */
export const KICK_BIAS_RIGHT_RAD = -33 * DEG;
/** 预备点距离：从球的背向目标一侧 0.45 m 处先绕过去，避免正面撞球。 */
export const PRE_STANDOFF_M = 0.45;
/** 单次指令的速度。必须**高于**实测步态地板 vx=0.22，否则会被抬到地板、失去"慢"的意义。 */
export const DRIVE_SPEED = 0.24;

/** 算站位：球心 + 目标区域 + 哪只脚 → 机器人该站哪、朝哪。 */
export function planStation(ballXy, zoneXy, foot = "right", biasRad = null) {
  const side = foot === "right" ? -1 : 1;
  const goal = Math.atan2(zoneXy.y - ballXy.y, zoneXy.x - ballXy.x);
  const bias = biasRad ?? (foot === "right" ? KICK_BIAS_RIGHT_RAD : -KICK_BIAS_RIGHT_RAD);
  const heading = wrap(goal - bias);
  const ox = BALL_OFFSET_X, oy = side * BALL_OFFSET_ABS_Y;
  const c = Math.cos(heading), s = Math.sin(heading);
  return {
    foot, goalHeading: goal, heading,
    stand: { x: ballXy.x - (ox * c - oy * s), y: ballXy.y - (ox * s + oy * c) },
    pre: { x: ballXy.x - Math.cos(heading) * PRE_STANDOFF_M,
           y: ballXy.y - Math.sin(heading) * PRE_STANDOFF_M },
    ball: { ...ballXy }, zone: { ...zoneXy },
    ballToZone: Math.hypot(zoneXy.x - ballXy.x, zoneXy.y - ballXy.y),
  };
}

/**
 * 站位 → 对准 → 起脚 的连续控制器。
 * 用连续速度指令而不是离散 token：token 一步走 0.07 m，而这里要的是 ±0.02 m 的精度。
 */
export class StationKicker {
  constructor({ standTolM = 0.025, headTolRad = 0.08, turnRate = 1.15, foot = "right",
                settleTicks = 20 } = {}) {
    Object.assign(this, { standTolM, headTolRad, turnRate, foot, settleTicks });
    this.reset();
  }

  reset() {
    this.phase = "idle";
    this.kicked = false;
    this.note = "";
    this.plan = null;
    this.latched = null;
    this.blindSteps = 0;
    this.wpIndex = 0;      // 航点只能往前走，不能回头 —— 否则会在 pre/stand 之间反复横跳
    this.settleLeft = 0;
    // 踢球偏角。标称 -33° 是静止起脚的实测值，但走完一整套动作后偏角会变，
    // 所以每踢一脚就用球的真实飞行方向标定一次（见 calibrate）。
    this.biasRad = this.foot === "right" ? KICK_BIAS_RIGHT_RAD : -KICK_BIAS_RIGHT_RAD;
    this.lastKick = null;
    this.lastDistance = null;
  }

  /**
   * 用"上一脚把球踢到哪了"反推真实偏角。
   * 球在滚动摩擦下基本沿直线走，所以位移方向 ≈ 出脚方向，足够当标定用。
   * @returns {number|null} 实测偏角（弧度）
   */
  calibrate(ballXy) {
    if (!this.lastKick) return null;
    const dx = ballXy.x - this.lastKick.ball.x, dy = ballXy.y - this.lastKick.ball.y;
    const moved = Math.hypot(dx, dy);
    if (moved < 0.12) return null;                       // 球没怎么动，量不出方向
    const measured = wrap(Math.atan2(dy, dx) - this.lastKick.heading);
    this.biasRad = measured;
    this.lastKick = null;
    return measured;
  }

  static ballInZone(ballXy, zone) {
    return Math.hypot(ballXy.x - zone.x, ballXy.y - zone.y) <= (zone.radius ?? 0.3);
  }

  /**
   * 推进一步。
   *
   * 关键设计：**站位一旦算出来就锁存，之后不再依赖看得见球**。
   * 物理原因是站位点上球只在相机前方 9 cm，早就掉出画面下沿了（实测 <0.14 m 就出画）。
   * 如果非要"看见才算"，最需要精度的最后 20 cm 反而在瞎走 —— 这大概就是之前踢不准的原因。
   * 锁存是合法的：计划是用真实观测到的球位算的，最后那段靠航位推算，没读任何隐藏真值。
   */
  tick({ duckPose, ballXy, zone }) {
    // 球已经在区域里了：直接收工（这也是多脚连踢的终止条件）
    if (ballXy && StationKicker.ballInZone(ballXy, zone)) {
      this.phase = "done";
      this.note = `球已在区域内 (${ballXy.x.toFixed(2)},${ballXy.y.toFixed(2)})`;
      return { cmd: [0, 0, 0], kick: false, phase: this.phase, note: this.note, done: true };
    }
    if (ballXy && (!this.latched ||
        Math.hypot(ballXy.x - this.latched.ball.x, ballXy.y - this.latched.ball.y) > 0.05)) {
      this.latched = planStation(ballXy, zone, this.foot, this.biasRad);
      this.blindSteps = 0;
      this.wpIndex = 0;
      this.settleLeft = this.settleTicks;
    }
    if (!this.latched) {
      this.phase = "no-plan";
      return { cmd: [0, 0, 0], kick: false, phase: this.phase, note: "还没看到球，交给探索/模型去找" };
    }
    if (!ballXy) this.blindSteps += 1;
    const p = this.plan = this.latched;

    if (this.kicked) {
      this.phase = "recover";
      return { cmd: [0, 0, 0], kick: false, phase: this.phase, note: this.note };
    }

    // 1) 先绕到预备点（从背向目标那一侧靠近），2) 再顺着踢球朝向进站位点
    const waypoints = [
      { name: "pre", xy: p.pre, tol: 0.07, drive: 3, coast: 1 },
      { name: "stand", xy: p.stand, tol: this.standTolM, drive: 1, coast: 1 },
    ];
    if (this.wpIndex < waypoints.length) {
      const wp = waypoints[this.wpIndex];
      const dx = wp.xy.x - duckPose.x, dy = wp.xy.y - duckPose.y;
      const dist = Math.hypot(dx, dy);
      this.lastDistance = dist;
      if (dist <= wp.tol) { this.wpIndex += 1; }         // 这个航点过了，永不回头
      else {
        const err = wrap(Math.atan2(dy, dx) - duckPose.heading);
        this.phase = wp.name;
        if (Math.abs(err) > 0.25) {
          this.note = `${wp.name}: 转向（差 ${(Math.abs(err) / DEG).toFixed(0)}°）`;
          return { cmd: [0, 0, Math.sign(err) * this.turnRate], kick: false, phase: this.phase, note: this.note };
        }
        // 脉冲式驱动：单次指令给足 DRIVE_SPEED（**必须高于步态地板 0.22**，
        // 否则会被 normalizeTwist 抬到地板、根本不慢），靠通断比压平均速度。
        // 站位段每 4 拍才走 1 拍，等效 ~0.06 m/s，精度靠这个拿。
        this.pulse = (this.pulse || 0) + 1;
        const cycle = wp.drive + wp.coast;
        const armed = (this.pulse % cycle) < wp.drive;
        this.note = `${wp.name}: ${armed ? "迈步" : "站定"}，还差 ${dist.toFixed(2)} m${this.blindSteps > 10 ? "（球已出画，靠推算）" : ""}`;
        return { cmd: [armed ? DRIVE_SPEED : 0, 0,
                       Math.sign(err) * Math.min(0.35, Math.abs(err))], kick: false,
                 phase: this.phase, note: this.note };
      }
    }

    // 3) 到位，转到起脚朝向
    const headErr = wrap(p.heading - duckPose.heading);
    if (Math.abs(headErr) > this.headTolRad) {
      this.phase = "align";
      this.note = `站位就绪，对准（差 ${(Math.abs(headErr) / DEG).toFixed(0)}°）`;
      return { cmd: [0, 0, Math.sign(headErr) * this.turnRate], kick: false, phase: this.phase, note: this.note };
    }

    // 4) 起脚
    if (this.settleLeft > 0) {       // 先站稳：训练时也是从稳定的站立姿态起步的
      this.settleLeft -= 1;
      this.phase = "settle";
      this.note = `站定中（还剩 ${this.settleLeft} 拍）`;
      return { cmd: [0, 0, 0], kick: false, phase: this.phase, note: this.note };
    }
    this.phase = "kick";
    this.kicked = true;
    this.lastKick = { heading: duckPose.heading, ball: { ...p.ball } };
    this.note = `站位 (${duckPose.x.toFixed(2)},${duckPose.y.toFixed(2)}) 朝向 ${(duckPose.heading / DEG).toFixed(0)}°，起脚`;
    return { cmd: [0, 0, 0], kick: true, phase: this.phase, note: this.note };
  }
}
