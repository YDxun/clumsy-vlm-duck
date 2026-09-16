/**
 * 任务 schema 校验 —— 让"场景包与网页端对不上"这件事**只能被发现，不能被忽略**。
 *
 * 起因：网页端曾经去读顶层 `curated.target`，而 tasks.yaml 的 schema 根本没有这个字段
 * （目标写在 `success` 的判据里：`distance_xy_le.target` / `body_in_zone.body`）。
 * 结果是**全部 14 个任务**都在靠措辞推断目标，只是碰巧推对了 —— 加任务时一旦措辞里
 * 没有可识别名词，就会静默退化成默认的"球"，然后整局按错误目标跑。
 *
 * 这个脚本直接调用运行时的推导函数（`src/scene.js` 的 `SceneIndex.taskTarget`），
 * **不抄第二份规则** —— 校验的正是网页真正会用的那套逻辑。
 *
 * 检查四项：
 *   1. 每个任务都能推导出目标（或明确属于"没有目标物体"的姿态/恢复类任务）
 *   2. 推导出的目标/区域在 metadata.yaml 里真的存在
 *   3. success/failure 判据里引用的每个 body / zone 都存在（拼错就报）
 *   4. id 唯一、中英文指令至少有一个
 *
 * 用法：node tools/check_task_schema.mjs [--json]
 * 退出码：0 = 全部合规；1 = 有问题
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SceneIndex } from "../src/scene.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const AS_JSON = process.argv.includes("--json");

/** 判据里出现的 body / zone 字段名（`target` 是"要去的对象"，也一起查存在性）。 */
const REF_FIELDS = ["body", "target", "zone"];

function addressable(meta) {
  const out = new Set();
  for (const o of meta.objects || []) out.add(o.body || o.id);
  for (const z of meta.zones || []) out.add(z.body || z.id);
  for (const p of meta.people || []) out.add(p.body || p.id);
  if (meta.beacon?.body) out.add(meta.beacon.body);
  if (meta.robot?.body_name) out.add(meta.robot.body_name);
  return out;
}

function predicates(task) {
  return [...(task.success?.all || []), ...(task.success?.any || []),
          ...(task.failure?.all || []), ...(task.failure?.any || [])];
}

async function checkScene(scene) {
  const dir = path.join(ROOT, "assets", scene.dir || scene.id);
  const problems = [];
  const rows = [];
  let meta, tasks;
  try {
    meta = JSON.parse(await readFile(path.join(dir, "metadata.json"), "utf8"));
    tasks = JSON.parse(await readFile(path.join(dir, "tasks.json"), "utf8")).tasks || [];
  } catch (e) {
    return { id: scene.id, problems: [`读不到摊平后的产物（先跑 flatten_scene.py）：${e.message}`], rows };
  }
  const bodies = addressable(meta);
  const index = new SceneIndex(meta, { tasks });
  const seen = new Set();

  for (const t of tasks) {
    const label = `${scene.id}/${t.id || "(无 id)"}`;
    if (!t.id) problems.push(`${label}: 缺 id`);
    else if (seen.has(t.id)) problems.push(`${label}: id 重复`);
    seen.add(t.id);
    if (!t.instruction_zh && !t.instruction_en) problems.push(`${label}: 中英文指令都没有`);
    if (!t.success) problems.push(`${label}: 没有 success 判据 —— 没法判成功，也推不出目标`);

    // 判据里引用的东西是否真实存在
    for (const p of predicates(t)) {
      for (const f of REF_FIELDS) {
        if (p[f] && !bodies.has(p[f])) {
          problems.push(`${label}: 判据 ${p.type} 里的 ${f}="${p[f]}" 在 metadata 里不存在`);
        }
      }
      for (const z of p.zones || []) {
        if (!bodies.has(z)) problems.push(`${label}: 判据 ${p.type} 里的 zone "${z}" 在 metadata 里不存在`);
      }
    }

    // 目标推导
    const d = index.taskTarget(t);
    rows.push({ task: t.id, target: d.target, zone: d.zone, radius: d.radius, source: d.source });
    if (!d.target) {
      const isPoseTask = predicates(t).some((p) =>
        ["robot_pose_region", "ever_fallen", "upright", "hold_condition_s"].includes(p.type)) &&
        !predicates(t).some((p) => ["body_in_zone", "distance_xy_le"].includes(p.type));
      if (!isPoseTask) {
        problems.push(`${label}: 推导不出目标（判据里既没有 distance_xy_le.target，也没有 body_in_zone）`);
      }
    } else if (!bodies.has(d.target)) {
      problems.push(`${label}: 推导出的目标 "${d.target}" 在 metadata 里不存在`);
    }
    if (d.zone && !bodies.has(d.zone)) {
      problems.push(`${label}: 推导出的区域 "${d.zone}" 在 metadata 里不存在`);
    }
  }
  return { id: scene.id, problems, rows };
}

async function main() {
  const manifest = JSON.parse(await readFile(path.join(ROOT, "scenes.json"), "utf8"));
  const results = [];
  for (const s of manifest.scenes || []) results.push(await checkScene(s));
  const total = results.reduce((n, r) => n + r.problems.length, 0);

  if (AS_JSON) {
    console.log(JSON.stringify({ ok: total === 0, results }, null, 2));
  } else {
    console.log("=== 任务 schema 校验（网页端真实使用的推导路径）===");
    for (const r of results) {
      console.log(`${r.problems.length ? "✗" : "✓"} ${r.id}`);
      for (const row of r.rows) {
        const tgt = row.target || "（无目标物体）";
        const z = row.zone ? ` → ${row.zone}${row.radius ? `@${row.radius}m` : ""}` : "";
        console.log(`    ${String(row.task).padEnd(28)} ${tgt.padEnd(16)} [${row.source}]${z}`);
      }
      for (const p of r.problems) console.log(`    !! ${p}`);
    }
    console.log(total ? `\n结果: FAIL —— ${total} 处需要修`
                      : "\n结果: PASS —— 每个任务的目标都能从成功判据推导出来，且指向真实存在的对象");
  }
  process.exit(total ? 1 : 0);
}

main().catch((e) => { console.error("异常:", e); process.exit(2); });
