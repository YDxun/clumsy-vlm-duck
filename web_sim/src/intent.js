/**
 * 自由文本 → 粗粒度意图 —— `duck_vlm/intent.py` 的移植。
 *
 * 用途：场景包里那些精选任务（task id）仍然权威；但当用户随手写一句话时，
 * 我们需要知道他说的是「走过去」还是「踢到某处」，才能决定：
 *   - 跟踪哪个目标（踢球时要盯球，而不是盯绿色区域）；
 *   - 要不要开启探索（原地转圈类任务就别乱走）。
 * 词表顺序就是优先级，先匹配到的家族胜出。
 */

export const RECOVER = ["recover", "get up", "stand up", "get back up", "on its feet",
                        "起身", "站起来", "恢复", "爬起来"];
export const ORBIT = ["orbit", "circle around", "go behind", "side-rear", "walk around",
                      "绕", "侧后方", "转到后面"];
export const GAIT = ["rough ground", "uneven", "gravel", "corridor", "narrow", "ramp",
                     "不平", "碎石", "走廊", "窄道", "斜坡"];
// 注意这里**没有**光秃秃的 stop：『走到方块旁边然后停下』是接近任务，
// 归成姿态任务会让探索插件被关掉。
export const POSE = ["turn right", "turn left", "turn around", "spin in place", "spin around", "spin",
                     "stand still", "sit", "stay there", "wait there",
                     "原地", "转身", "站着", "别动", "右转", "左转"];
export const MANIPULATE = ["kick", "push", "shove", "nudge", "bring", "retrieve", "carry", "deliver",
                           "take the", "take it", "put it", "pick", "grab", "move it",
                           "put it in", "into the",
                           "踢", "推", "捅", "送到", "运到", "带到", "拿给", "放进", "拿", "捡", "抓"];
export const APPROACH = ["go to", "go over", "walk to", "walk towards", "approach", "reach", "find",
                         "get to", "move to", "head to", "come here", "come over", "come and",
                         "过去", "走向", "走到", "靠近", "找到", "去找", "过来", "来这里",
                         "去", "到", "找", "看看"];
// 否定：「别靠近蓝方块」不能把蓝方块当成目标。
export const NEGATION = ["don't", "do not", "avoid", "stay away", "keep away", "not go",
                         "别", "不要", "不准", "远离", "避免"];

/** @returns {"recover"|"orbit"|"gait"|"pose"|"manipulate"|"approach"|"avoid"|"unknown"} */
export function resolveIntent(taskId, taskText) {
  const text = String(taskText || "").toLowerCase();
  if (NEGATION.some((w) => text.includes(w))) return "avoid";
  // 操作类排在绕行/地形之前：『把球绕着柱子推进区域』是操作任务，只是在避障。
  for (const [name, words] of [["recover", RECOVER], ["manipulate", MANIPULATE], ["orbit", ORBIT],
                               ["gait", GAIT], ["pose", POSE], ["approach", APPROACH]]) {
    if (words.some((w) => text.includes(w))) return name;
  }
  return "unknown";
}

/** 走向航点会和目标打架的意图（此时不该开探索/寻路）。 */
export function isBlockingIntent(taskId, taskText) {
  return ["recover", "orbit", "gait", "pose"].includes(resolveIntent(taskId, taskText));
}

/**
 * 从「句子里提到的实体」里挑出该被跟踪的 body。
 * hits 来自 SceneIndex.scanText，最具体的标签在前。
 * 「把球踢进绿色区域」里最长匹配是区域，但要操作的是球 → 操作类意图优先非 zone 实体。
 */
export function pickTarget(hits, taskText) {
  if (!hits || !hits.length) return { body: "ball", label: null };
  const intent = resolveIntent("", taskText);
  if (intent === "avoid") return { body: "ball", label: null };   // 别去追用户让你躲开的东西
  if (intent === "manipulate") {
    for (const [label, body, kind] of hits) if (kind !== "zone") return { body, label };
  }
  const [label, body] = hits[0];
  return { body, label };
}
