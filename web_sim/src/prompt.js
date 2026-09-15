/**
 * zero-shot 提示词 —— 与 `duck_vlm/prompts/zero_shot.txt` 同一套字段契约。
 *
 * 为什么坚持照抄字段名（TASK / ROBOT STATE / SUBTASK / …）：
 * 这些字段是模型被真实调好的那套接口。字段一改，模型的表现就不可比，
 * 而且 Python 端与浏览器端的对照实验也失去意义。
 * 控制规则里删掉了浏览器暂时做不到的技能（踢球、翻滚、起身），
 * 词表也必须和 actions.js 的 8 个 token 严格一致——**不能提示模型输出没有的动作**。
 */
import { tokenMenu, tokenSemantics } from "./actions.js";

export const ZERO_SHOT_TEMPLATE = `You are controlling a small bipedal duck robot in a simulated room. You see the duck's first-person head camera. Decide exactly one discrete action token at each step.

TASK: {task}
TASK ID: {task_id}
TARGET: {target}
ROBOT STATE: {state}
SUBTASK: {subgoal}
RECENT ACTIONS (newest first): {recent_actions}
PERCEPTION / AFFORDANCE: {affordance}
{proprio}
{recovery}

AVAILABLE TOKENS:
{token_menu}

TOKEN MEANINGS:
{token_semantics}

CONTROL RULES:
- Output exactly one token from AVAILABLE TOKENS and nothing else.
- FWD/BACK move the body in its current facing direction for one short step.
- TURN_L/TURN_R rotate the body in place.
- STOP holds position for one step.
- DONE only when TASK is visibly complete or the target is already achieved.
- Use SUBTASK, ROBOT STATE, and AFFORDANCE as trusted progress measurements. When the target is known but not visible, or its bearing magnitude exceeds 0.6 rad, keep turning consistently toward the target bearing sign; do not alternate TURN_L and TURN_R.
- For a distance-and-heading task, make coarse progress while far away, then use shorter corrections when the remaining distance or heading error is small.
- Avoid oscillating between opposite actions. If the target is not visible, scan by rotating, but do not spin on the spot forever: after a few scans, move forward to a new vantage point and scan again.
- LOOK_DOWN keeps standing while pitching the head down so the duck-cam sees the ground just in front of the duck (roughly 0.15-0.45 m ahead). HEAD_CENTER looks forward again. Use LOOK_DOWN when a target you were pursuing disappears at close range.
- Range and bearing are only reported while the duck-cam actually holds the target. If they are missing you are blind: sweep with TURN_L/TURN_R (or LOOK_DOWN when the target is very close) instead of guessing a direction through walls.
- The geometric facts in ROBOT STATE and AFFORDANCE come from trusted sensors; do not invent coordinates.

Return the single token only, no punctuation and no explanation:`;

/**
 * 渲染一条完整提示词。
 * @param {object} obs 观测（见 agent.js buildObservation）
 * @param {string[]} allowed 允许的 token 列表
 */
export function renderPrompt(obs, allowed) {
  const values = {
    "{task}": obs.task,
    "{task_id}": obs.taskId || "(none)",
    "{target}": obs.target,
    "{state}": obs.stateText,
    "{subgoal}": obs.subgoal || "(none)",
    "{recent_actions}": (obs.recentActions || []).slice(0, 5).join(", ") || "(none)",
    "{affordance}": obs.affordanceText || "(none)",
    "{proprio}": obs.proprioText || "",
    "{recovery}": obs.recoveryHint || "",
    "{token_menu}": tokenMenu(allowed),
    "{token_semantics}": tokenSemantics(allowed),
  };
  let out = ZERO_SHOT_TEMPLATE;
  for (const [key, value] of Object.entries(values)) out = out.split(key).join(String(value));
  return out.trim() + "\n";
}
