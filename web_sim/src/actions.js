/**
 * 离散动作词表 —— 浏览器端**最小集**（8 个 token）。
 *
 * 语义与 `duck_vlm/actions.py` 的 ACTION_SPECS 一致，只是砍到最小可用集合：
 * 视觉语言模型永远不输出速度或关节目标，只输出这里的**一个词**，
 * 由 interpreter.js 翻译成具体的运动指令或头部姿态。
 *
 * 有意没放进最小集的（等页面跑通再加，词表越大 zero-shot 越容易选错）：
 *   STRAFE_L / STRAFE_R（侧移）、STAND（长站）、LOOK_DOWN_MORE（更深的低头）、
 *   KICK_L / KICK_R / ROLL / SIT / STAND_UP / DANCE（技能，需要各自的 ONNX 状态机）。
 */

export const ACTION_SPECS = {
  FWD: {
    token: "FWD", description: "Walk forward one short step",
    kind: "motion", durationS: 0.55, command: [0.35, 0, 0],
  },
  BACK: {
    token: "BACK", description: "Walk backward one short step",
    kind: "motion", durationS: 0.55, command: [-0.35, 0, 0],
  },
  TURN_L: {
    token: "TURN_L", description: "Turn left in place",
    kind: "motion", durationS: 0.55, command: [0, 0, 1.5],
  },
  TURN_R: {
    token: "TURN_R", description: "Turn right in place",
    kind: "motion", durationS: 0.55, command: [0, 0, -1.5],
  },
  STOP: {
    token: "STOP", description: "Stop and stand still for one short interval",
    kind: "control", durationS: 0.30, command: [0, 0, 0],
  },
  // 回头看地面：鸭子保持站立，只把头部关节压下去，让头摄能看到脚前的地面。
  // 当初加这两个 token 就是为了解开“球在画面外就再也找不到”的死锁；
  // head_pitch 增大 = 低头（已用正运动学验证过）。
  LOOK_DOWN: {
    token: "LOOK_DOWN",
    description: "Keep standing and pitch the head down to see the ground just in front of the duck",
    kind: "head", durationS: 1.20,
    headDelta: [["neck_pitch", 0.0], ["head_pitch", 0.60], ["head_yaw", 0.0], ["head_roll", 0.0]],
  },
  HEAD_CENTER: {
    token: "HEAD_CENTER",
    description: "Return the head to the neutral forward pose so the duck looks ahead again",
    kind: "head", durationS: 0.90,
    headDelta: [["neck_pitch", 0.0], ["head_pitch", 0.0], ["head_yaw", 0.0], ["head_roll", 0.0]],
  },
  DONE: {
    token: "DONE", description: "Declare the task complete and stop",
    kind: "terminal", durationS: 0, terminal: true,
  },

  // ---------------- 技能 token ----------------
  // 每个技能 = 切到自己的 ONNX 策略、跑固定时长、再切回走路。
  // `verified` 表示「在本仓库的浏览器仿真里实测有效」。只有 verified 的才会进
  // VLM 的候选词表 —— 没验过的动作喂给模型，只会让它学会输出一个没反应的词。
  ROLL: {
    token: "ROLL", description: "Run the forward-roll policy, then get back up",
    kind: "skill", policy: "roulade", durationS: 2.8, verified: true,
  },
  DANCE: {
    token: "DANCE", description: "Run the happy-hop dance policy",
    kind: "skill", policy: "happy_hop", durationS: 2.5, verified: true,
  },
  // 下面四个策略文件都在仓库里、也都能被 onnxruntime-web 加载，但**从站立姿态 + 零速度指令**
  // 跑起来看不到可见动作：Python 端靠 LocalKick 的站位/状态机（先站稳、再进动作、再恢复）才踢得中。
  // 这里如实标成未验证：手动面板可以试，但不会进 VLM 词表。
  KICK_L: {
    token: "KICK_L", description: "Run the left-foot ball-kick policy (needs the duck to be lined up on the ball)",
    kind: "skill", policy: "ball_kick_left", durationS: 1.2, verified: false,
  },
  KICK_R: {
    token: "KICK_R", description: "Run the right-foot ball-kick policy (needs the duck to be lined up on the ball)",
    kind: "skill", policy: "ball_kick_right", durationS: 1.2, verified: false,
  },
  SIT: {
    token: "SIT", description: "Run the sit-stand policy, then hold",
    kind: "skill", policy: "alpha_sitstand", durationS: 1.5, verified: false,
  },
  STAND_UP: {
    token: "STAND_UP", description: "Get back up onto the feet after a fall",
    kind: "skill", policy: "alpha_standup", durationS: 2.0, verified: false,
  },
};

export const TOKENS = Object.keys(ACTION_SPECS);
export const MOTION_TOKENS = TOKENS.filter((t) => ACTION_SPECS[t].kind === "motion");
export const SKILL_TOKENS = TOKENS.filter((t) => ACTION_SPECS[t].kind === "skill");

/**
 * 该发给模型哪些 token。
 * 规则：非技能 token 全给；技能 token 必须**策略已加载**且**本地实测有效**。
 * 这就是 Python `available_tokens()` 的同一条原则：不许给模型一个它做不到的动作。
 */
export function availableTokens(policies = {}, { includeUnverified = false } = {}) {
  return TOKENS.filter((t) => {
    const spec = ACTION_SPECS[t];
    if (spec.kind !== "skill") return true;
    if (!policies[spec.policy]) return false;
    return includeUnverified || spec.verified === true;
  });
}

/** 托管 VLM 爱加前缀/换写法，这里把常见同义词收回来（中英都有）。 */
export const ALIASES = {
  FORWARD: "FWD", MOVE_FORWARD: "FWD", MOVE_FWD: "FWD", 前进: "FWD", 向前: "FWD",
  BACKWARD: "BACK", MOVE_BACK: "BACK", 后退: "BACK", 向后: "BACK",
  TURN_LEFT: "TURN_L", LEFT_TURN: "TURN_L", 左转: "TURN_L",
  TURN_RIGHT: "TURN_R", RIGHT_TURN: "TURN_R", 右转: "TURN_R",
  HALT: "STOP", WAIT: "STOP", STAND: "STOP", 停止: "STOP", 站住: "STOP",
  LOW_HEAD: "LOOK_DOWN", 低头: "LOOK_DOWN", 看地面: "LOOK_DOWN",
  HEAD_UP: "HEAD_CENTER", RAISE_HEAD: "HEAD_CENTER", 抬头: "HEAD_CENTER", 头回正: "HEAD_CENTER",
  FINISH: "DONE", FINISHED: "DONE", COMPLETE: "DONE", 完成: "DONE",
};

export function tokenMenu(tokens = TOKENS) {
  return tokens.join(", ");
}

export function tokenSemantics(tokens = TOKENS) {
  return tokens.map((t) => `- ${t}: ${ACTION_SPECS[t].description}`).join("\n");
}

/** 清洗模型输出：去代码围栏、去包裹符号、统一大小写与分隔符（与 actions.py 同步）。 */
export function normalise(raw) {
  let v = String(raw ?? "").trim();
  v = v.replace(/^```(?:text)?\s*/i, "").replace(/\s*```$/, "").trim();
  v = v.replace(/^[\[\]{}()<>"']+/, "").replace(/[\[\]{}()<>"']+$/, "").trim();
  return v.toUpperCase().replace(/-/g, "_").replace(/ /g, "_");
}

/**
 * 从一段模型回复里恢复出一个**允许的** token。
 * 对训练够严，对会加话的托管 VLM 够宽容：直接命中 → JSON 字段 → 别名 →
 * 词边界匹配 → 别名包含 → 唯一包含。
 *
 * @returns {string|null} 解析失败返回 null（调用方决定用哪条规则兜底）
 */
export function parseToken(raw, allowed = TOKENS) {
  const allowedSet = new Set(allowed);
  const text = String(raw ?? "").trim();
  if (!text) return null;

  let direct = normalise(text);
  direct = ALIASES[direct] || direct;
  if (allowedSet.has(direct)) return direct;

  // 托管 VLM / function calling 常见的 JSON 包裹
  try {
    const obj = JSON.parse(text);
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {
      for (const key of ["token", "action", "answer", "decision"]) {
        const hit = ALIASES[normalise(obj[key])] || normalise(obj[key]);
        if (allowedSet.has(hit)) return hit;
      }
    }
  } catch { /* 不是 JSON，继续往下试 */ }

  const normalised = normalise(text);
  const aliasHits = [...allowedSet].filter((t) => normalised.includes(t));
  if (aliasHits.length === 1) return aliasHits[0];

  // 原文里找词边界，长 token 优先（避免 TURN_L 抢走 TURN_LEFT 之类）
  const upper = text.toUpperCase();
  for (const token of [...allowedSet].sort((a, b) => b.length - a.length)) {
    const re = new RegExp(`(?<![A-Z0-9_])${token}(?![A-Z0-9_])`);
    if (re.test(upper)) return token;
  }

  for (const [alias, token] of Object.entries(ALIASES).sort((a, b) => b[0].length - a[0].length)) {
    if (allowedSet.has(token) && upper.includes(alias.toUpperCase())) return token;
  }

  const hits = [...allowedSet].filter((t) => upper.includes(t.toUpperCase()));
  return hits.length === 1 ? hits[0] : null;
}
