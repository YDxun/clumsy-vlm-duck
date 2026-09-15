/**
 * 场景元数据索引 —— `duck_vlm/scenes.py` 的 SceneSpec 在浏览器里的对应物。
 *
 * 存在的意义：用户不会照着 MJCF 里的 body 名说话。场景包写的是「红色方块」，
 * 人说的是「红方块」；写的是「橙色球」，人说的是「球」。
 * 所以这里做两层匹配：
 *   1. 标签（semantic_labels）直接出现在句子里；
 *   2. 「颜色词 + 名词」同时在句子里 —— 但必须**相邻**，否则
 *      「把红方块推到蓝色区域」会同时命中「蓝色方块」而选错东西。
 */

export const norm = (text) => String(text ?? "").trim().toLowerCase().replace(/[\s_\-]+/g, " ").trim();

const COLORS = {
  red: ["red", "红"], blue: ["blue", "蓝"], green: ["green", "绿"],
  orange: ["orange", "橙"], yellow: ["yellow", "黄"],
};
const NOUNS = {
  cube: ["cube", "block", "box", "方块", "立方体"],
  ball: ["ball", "sphere", "球"],
  cup: ["cup", "cylinder", "杯", "圆柱", "筒"],
  zone: ["zone", "area", "区域"],
  beacon: ["beacon", "marker", "信标", "标记"],
};

function looseEntityMatch(label, text) {
  const lab = norm(label);
  const compact = norm(text).replace(/\s+/g, "");
  if (!lab || !compact) return false;
  const colors = Object.keys(COLORS).filter((c) => COLORS[c].some((w) => lab.includes(w)));
  const nouns = Object.keys(NOUNS).filter((n) => NOUNS[n].some((w) => lab.includes(w)));
  if (!colors.length || !nouns.length) return false;
  for (const c of colors) for (const n of nouns) {
    for (const cw of COLORS[c]) for (const nw of NOUNS[n]) {
      if (compact.includes(cw + nw) || compact.includes(nw + cw)) return true;
    }
  }
  return false;
}

export function labelInText(label, text) {
  const token = norm(label);
  if (token && norm(text).includes(token)) return true;
  return looseEntityMatch(label, text);
}

export class SceneIndex {
  constructor(metadata = {}, tasks = {}) {
    this.metadata = metadata || {};
    this.tasks = tasks || {};
    const add = (out, body, kind, names) => {
      if (!body) return;
      for (const name of names) if (name) out.push([String(name), String(body), kind]);
    };
    const out = [];
    for (const obj of this.metadata.objects || []) {
      add(out, obj.body || obj.id, "object", [obj.id, obj.body, ...(obj.semantic_labels || [])]);
    }
    for (const person of this.metadata.people || []) {
      add(out, person.body || person.id, "person", [person.id, person.body, ...(person.semantic_labels || [])]);
    }
    for (const zone of this.metadata.zones || []) {
      add(out, zone.body || zone.id, "zone", [zone.id, zone.body, ...(zone.semantic_labels || [])]);
    }
    const beacon = this.metadata.beacon || {};
    add(out, beacon.body, "beacon", [beacon.body, ...(beacon.semantic_labels || [])]);
    // 长标签在前：越具体的说法越先被匹配到
    out.sort((a, b) => norm(b[0]).length - norm(a[0]).length);
    this.entities = out;
  }

  /** 精确命中优先，其次「最长的标签被包含在问句里」。 */
  resolveBody(query) {
    const q = norm(query);
    if (!q) return null;
    const robot = this.metadata.robot || {};
    if (q === norm(robot.body_name) || q === norm(robot.root_body)) return robot.body_name || robot.root_body;
    if (["beacon", "target beacon", "信标"].includes(q)) return (this.metadata.beacon || {}).body || null;
    for (const [name, body] of this.entities) if (norm(name) === q) return body;
    // 严格包含之后还要走宽松匹配：场景包写「红色方块」，用户说「红方块」，
    // 少一个字就匹配不上——这正是 labelInText 里「颜色词+名词相邻」那条规则管的事。
    for (const [name, body] of this.entities) {
      if (name && (q.includes(norm(name)) || labelInText(name, q))) return body;
    }
    return null;
  }

  entityNames() { return this.entities.map(([label, body, kind]) => [label, body, kind]); }

  /** 句子里提到的实体，最具体的在前（同一个 body 只留一次）。 */
  scanText(text) {
    if (!norm(text)) return [];
    const hits = [], seen = new Set();
    for (const [label, body, kind] of this.entities) {
      if (seen.has(body)) continue;
      if (labelInText(label, text)) { seen.add(body); hits.push([label, body, kind]); }
    }
    return hits;
  }

  listTasks() { return this.tasks?.tasks || []; }

  task(taskId) {
    return this.listTasks().find((t) => t.id === taskId) || null;
  }

  /** 任务 id / 中英文指令的精确或互相包含匹配。 */
  matchTask(text) {
    const q = norm(text);
    if (!q) return null;
    const exact = this.task(String(text).trim());
    if (exact) return exact;
    for (const item of this.listTasks()) {
      if (q === norm(item.id)) return item;
      for (const key of ["instruction_zh", "instruction_en"]) {
        const value = norm(item[key]);
        if (value && (value.includes(q) || q.includes(value))) return item;
      }
    }
    return null;
  }

  defaultTarget() {
    return (this.metadata.objects || [])[0]?.body || "ball";
  }

  /** 目标类的默认靶子：踢/推这类操作先看球。 */
  at(kind) { return this.entities.find(([, , k]) => k === kind) || null; }
}
