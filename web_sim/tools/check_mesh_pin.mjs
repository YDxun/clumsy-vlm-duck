/**
 * 本地网格 vs 上游固定 commit 的**内容**比对。
 *
 * 之前只比过文件名（38/38 对上），但名字对上不代表字节一致 ——
 * 而"线上从上游取、本地拿自己那份验证"这种组合，只要内容有差，
 * 验证过的几何和访客实际跑的几何就不是一个东西。
 * 用 git blob SHA-1 比（GitHub tree API 直接给），不用下载 20 MB。
 */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LOCAL = path.resolve(HERE, "../../duck_scenes/robot/assets");
const SCENE = path.resolve(HERE, "../assets/_shared");

/** git 的 blob 哈希：sha1("blob <len>\0" + content) */
const gitBlobSha = (buf) =>
  createHash("sha1").update(`blob ${buf.length}\0`, "utf8").update(buf).digest("hex");

const tree = await (await fetch(
  "https://api.github.com/repos/pollen-robotics/microduck_rl/git/trees/HEAD?recursive=1",
  { headers: { "user-agent": "duck-vlm" } },
)).json();
const upstream = new Map(
  tree.tree
    .filter((x) => x.path.includes("robot/microduck/assets") && x.path.toLowerCase().endsWith(".stl"))
    .map((x) => [x.path.split("/").pop(), x.sha]),
);

// 只比对我们实际用到的那 38 个
const used = readdirSync(SCENE).filter((f) => f.toLowerCase().endsWith(".stl"));
let same = 0;
const diffs = [];
for (const name of used) {
  const local = gitBlobSha(readFileSync(path.join(LOCAL, name)));
  const up = upstream.get(name);
  if (up === local) same++;
  else diffs.push({ name, local: local.slice(0, 10), upstream: up ? up.slice(0, 10) : "(上游没有)" });
}

console.log(`用到 ${used.length} 个网格 | 与上游一致 ${same} | 不一致 ${diffs.length}`);
if (diffs.length) {
  console.log("不一致的：");
  for (const d of diffs.slice(0, 12)) console.log(`  ${d.name.padEnd(38)} 本地 ${d.local}  上游 ${d.upstream}`);
  if (diffs.length > 12) console.log(`  …还有 ${diffs.length - 12} 个`);
}
