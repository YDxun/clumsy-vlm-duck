/**
 * 规则层 —— 「其余走规则」里那个“其余”。
 *
 * 挂在「推理 → 动作」之间：模型给出候选 token，规则可以否决或替换它。
 * 每条规则都有名字和开关，可以在页面上单独关掉做消融 —— 这正是这套系统
 * 要证明的事：哪些能力真来自 VLM，哪些其实是规则在兜底。
 *
 * 这里只放“不依赖视觉理解”的确定性规则；需要看懂画面的判断一律留给模型。
 */

export const MOTION_TOKENS = ["FWD", "BACK", "TURN_L", "TURN_R"];

export const DEFAULT_RULES = {
  fallen_stop: {
    enabled: true,
    description: "摔倒时拒绝执行位移动作（鸭子站不稳，走了只会更糟）",
  },
  anti_stuck: {
    enabled: true,
    description: "连续前进却几乎没位移 → 改成左转脱困（顶住墙/卡在障碍物上）",
  },
  anti_spin: {
    enabled: true,
    description: "连续左右互摆地原地转 → 改成前进，打破“原地扫描”死循环",
  },
  look_down_when_lost: {
    enabled: true,
    description: "近距离把目标跟丢了 → 强制低头找（球出画的经典死锁）",
  },
};

/**
 * @param {object} ctx
 * @param {string} ctx.proposed            模型给的 token
 * @param {string[]} ctx.recentActions     最近执行过的 token（新在前）
 * @param {object} ctx.state               DuckStateSensor 的快照
 * @param {{rangeM:number}|null} ctx.lastSeen 上一次看见目标时的距离
 * @param {number} ctx.stepsSinceLastSeen  距离上次看见过了几个决策步
 * @param {number} ctx.distanceMovedInWindow 最近窗口内的总位移（米）
 * @returns {{token:string, applied:string[], notes:string[]}}
 */
export function applyRules(ctx, rules = DEFAULT_RULES) {
  const applied = [];
  const notes = [];
  let token = ctx.proposed;
  const on = (name) => rules[name]?.enabled;
  const recent = ctx.recentActions || [];

  // 1) 摔倒保护
  if (on("fallen_stop") && ctx.state?.fallen && MOTION_TOKENS.includes(token)) {
    applied.push("fallen_stop");
    notes.push(`fallen=true，否决 ${token}，原地停住`);
    token = "STOP";
  }

  // 2) 顶住东西走不动：连续 FWD 但位移极小
  if (on("anti_stuck") && token === "FWD" && recent.length >= 2 &&
      recent[0] === "FWD" && recent[1] === "FWD" &&
      ctx.distanceMovedInWindow < 0.02) {
    applied.push("anti_stuck");
    notes.push(`连续 FWD 只移动 ${ctx.distanceMovedInWindow.toFixed(3)} m，改成 TURN_L 脱困`);
    token = "TURN_L";
  }

  // 3) 原地左右互摆：三轮交替转身 = 白转 1.65 s，直接往前挪个位置再找
  if (on("anti_spin") && ["TURN_L", "TURN_R"].includes(token) && recent.length >= 2 &&
      ["TURN_L", "TURN_R"].includes(recent[0]) && ["TURN_L", "TURN_R"].includes(recent[1]) &&
      recent[0] !== recent[1]) {
    applied.push("anti_spin");
    notes.push(`最近在 ${recent[1]} / ${recent[0]} 之间反复，改成 FWD 换视角`);
    token = "FWD";
  }

  // 4) 近距离丢目标：目标大概率就在脚下或正前方地面，只有低头才看得见
  if (on("look_down_when_lost") && !ctx.state?.targetVisible &&
      ctx.lastSeen && ctx.lastSeen.rangeM !== null && ctx.lastSeen.rangeM < 0.6 &&
      ctx.stepsSinceLastSeen <= 40 && token !== "LOOK_DOWN" && token !== "HEAD_CENTER") {
    applied.push("look_down_when_lost");
    notes.push(`目标 ${ctx.lastSeen.rangeM.toFixed(2)} m 处刚丢失，低头找`);
    token = "LOOK_DOWN";
  }

  return { token, applied, notes };
}

/**
 * 无 key 时的兜底策略（Python 端 dry_run 的对应物）。
 * 不做任何视觉理解，只用本体感知：看不见就转圈扫，看得见就对准了走。
 * UI 上的“规则模式”就是它；也用来在没有 API key 的环境里回归整条闭环。
 */
export function reflexToken(state) {
  if (state.fallen) return "STOP";
  if (!state.targetVisible || state.targetRangeM === null) return "TURN_L";
  const bearing = state.targetBearingRad;
  if (Math.abs(bearing) > 0.30) return bearing > 0 ? "TURN_L" : "TURN_R";
  if (state.targetRangeM > 0.35) return "FWD";
  return "DONE";
}
