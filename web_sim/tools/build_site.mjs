/**
 * 打包可发布的静态站点 —— 产物丢到任何静态托管就能跑，不需要 node_modules。
 *
 * 两种资产模式（这一点直接关系到**能不能合法公开**）：
 *
 *   --assets=inline     assets/ 全拷（含 22 MB 网格）→ 自包含，但**会重新分发 CC BY-SA-NC 的网格**
 *   --assets=external   **公开站点用这个**：不拷 `_shared/` 里的机器人网格，
 *                       scenes.json 里写上 meshBase 指向上游固定 commit，
 *                       浏览器运行时自己去取（照 quackd 的做法，不重新分发 NC 资产）。
 *                       场景 XML 与 ONNX 策略仍随站点发布 —— 前者是我们自己的，
 *                       后者是 Apache-2.0（见 THIRD_PARTY_NOTICES.md）。
 *
 * importmap 会被改写成 CDN（jsDelivr，版本号取自 package.json，避免"本地能跑线上不能跑"）。
 *
 * 用法：
 *   node tools/build_site.mjs --out _site
 *   node tools/build_site.mjs --out _site --assets=external --asset-base=https://huggingface.co/datasets/XenderYang/duck-sim-assets/resolve/main
 */
import { cp, mkdir, readFile, writeFile, rm, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const arg = (name, dflt = null) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};
const OUT = path.resolve(ROOT, arg("out", "_site"));
const ASSETS = arg("assets", "inline");
const ASSET_BASE = arg("asset-base", "");
/** 给 Hugging Face Space 用时写一份带 front-matter 的 README（sdk: static）。 */
const SPACE_CARD = arg("space-card", "");
/** 机器人网格的上游固定版本。CC BY-SA-NC 的硬件设计文件，公开站点不分发、运行时取。 */
const UPSTREAM_MESH = {
  repo: "pollen-robotics/microduck_rl",
  commit: arg("mesh-commit", "cb70b792312d"),
  path: "src/mjlab_microduck/robot/microduck/assets",
};
const UPSTREAM_MESH_BASE =
  `https://raw.githubusercontent.com/${UPSTREAM_MESH.repo}/${UPSTREAM_MESH.commit}/${UPSTREAM_MESH.path}`;

const CDN = "https://cdn.jsdelivr.net/npm";

async function main() {
  const pkg = JSON.parse(await readFile(path.join(ROOT, "package.json"), "utf8"));
  const dep = (name) => pkg.dependencies[name].replace(/^[\^~]/, "");
  const version = (name) => dep(name).split(".")[0] + "." + dep(name).split(".")[1] + "." + dep(name).split(".")[2];

  await rm(OUT, { recursive: true, force: true });
  await mkdir(OUT, { recursive: true });

  // 1) 页面：importmap 换成 CDN
  const html = await readFile(path.join(ROOT, "index.html"), "utf8");
  const importMap = {
    three: `${CDN}/three@${version("three")}/build/three.module.js`,
    "three/addons/": `${CDN}/three@${version("three")}/examples/jsm/`,
    "@mujoco/mujoco": `${CDN}/@mujoco/mujoco@${version("@mujoco/mujoco")}/mujoco.js`,
    "onnxruntime-web": `${CDN}/onnxruntime-web@${version("onnxruntime-web")}/dist/ort.min.mjs`,
  };
  const rewritten = html.replace(/<script type="importmap">[\s\S]*?<\/script>/,
    `<script type="importmap">\n${JSON.stringify({ imports: importMap }, null, 2)}\n</script>`);
  if (rewritten === html) throw new Error("没找到 importmap，index.html 的结构变了？");
  await writeFile(path.join(OUT, "index.html"), rewritten, "utf8");

  // 2) 代码与配置
  await cp(path.join(ROOT, "src"), path.join(OUT, "src"), { recursive: true });
  for (const f of ["policies.json", "scenes.json"]) {
    await cp(path.join(ROOT, f), path.join(OUT, f));
  }
  const runtimeBase = ASSET_BASE || `${CDN}/huggingface/...`; // 没给就用现场提示
  const ortWasm = `${CDN}/onnxruntime-web@${version("onnxruntime-web")}/dist/`;
  // ORT 的 wasm 走 CDN（否则产物里还得放一份 20 MB 的 wasm）
  await writeFile(path.join(OUT, "site-config.json"),
    JSON.stringify({ ortWasmPaths: ortWasm, generatedAt: new Date().toISOString() }, null, 2));

  // 3) 资产
  if (ASSETS === "inline") {
    await cp(path.join(ROOT, "assets"), path.join(OUT, "assets"), { recursive: true });
  } else if (ASSETS === "external") {
    // 拷 assets/，但**跳过 _shared/**（那 38 个 STL 是 CC BY-SA-NC 的硬件设计文件）
    await cp(path.join(ROOT, "assets"), path.join(OUT, "assets"), {
      recursive: true,
      filter: (src) => !src.split(path.sep).includes("_shared"),
    });
    const meshBase = arg("mesh-base", UPSTREAM_MESH_BASE);
    const manifest = JSON.parse(await readFile(path.join(ROOT, "scenes.json"), "utf8"));
    manifest.meshBase = meshBase;
    manifest._meshBase说明 =
      `机器人网格（CC BY-SA-NC 的硬件设计文件）不随本站分发，浏览器运行时从 ${UPSTREAM_MESH.repo} ` +
      `的固定 commit ${UPSTREAM_MESH.commit} 取。要升级请显式改 pin，别用 main。`;
    if (ASSET_BASE) manifest.assetBase = ASSET_BASE;
    await writeFile(path.join(OUT, "scenes.json"), JSON.stringify(manifest, null, 2) + "\n");
    console.log(`[build] 网格: 不分发，运行时取 ${meshBase}`);
  } else {
    throw new Error(`--assets 只支持 inline / external，收到 ${ASSETS}`);
  }

  // 4) 产物清单
  const size = async (p) => {
    try {
      const s = await stat(p);
      if (!s.isDirectory()) return s.size;
      let total = 0;
      for (const e of await (await import("node:fs/promises")).readdir(p, { withFileTypes: true })) {
        total += await size(path.join(p, e.name));
      }
      return total;
    } catch { return 0; }
  };
  const bytes = await size(OUT);
  console.log(`[build] 产物: ${OUT}`);
  console.log(`[build] 资产模式: ${ASSETS}${ASSET_BASE ? `（assetBase=${ASSET_BASE}）` : ""}`);
  console.log(`[build] importmap: jsDelivr（three ${version("three")} / mujoco ${version("@mujoco/mujoco")} / ort ${version("onnxruntime-web")}）`);
  console.log(`[build] 大小: ${(bytes / 1048576).toFixed(1)} MB`);
  console.log(`[build] 本地预览: node tools/serve.mjs 8787 ${path.relative(ROOT, OUT)}`);

  // 5) Hugging Face Space 的卡片（sdk: static 不需要任何构建步骤）
  if (SPACE_CARD) {
    const card = `---
title: DuckVLM Simulator
emoji: 🦆
colorFrom: yellow
colorTo: blue
sdk: static
pinned: false
license: other
short_description: Zero-shot VLM control of a simulated duck, in your browser
---

# DuckVLM 浏览器仿真（机器鸭 zero-shot 驾驶舱）

物理（MuJoCo WASM）、策略（onnxruntime-web）和 VLM 调用**全部在你的浏览器里**跑，
没有后端：API key 由你自带，只发给所选厂商。

- 三个场景（工作台 / 家庭跨房间 / 障碍地形），任务下拉直接列场景包里的 task id
- 8 个基础动作 token + 通过验证的技能（翻滚 / 跳舞）；推、踢由规则闭环站位
- 鼠标可缩放/旋转/平移；一键三视角拼图

模型与网格**不在本 Space 里重新分发**：页面运行时从上游固定版本取（见 scenes.json 的 assetBase）。

## 第三方资产与许可

| 内容 | 许可 | 关系 |
| --- | --- | --- |
| 本项目的代码（web_sim/） | Apache-2.0 | 我们的 |
| 机器人 3D 模型（38 个 STL） | **CC BY-SA-NC**（上游 README 原文："Hardware design files are licensed under Creative Commons BY-SA-NC"） | **不在本 Space 里**：浏览器运行时从 pollen-robotics/microduck_rl 的固定 commit 取，本站不留任何字节。非商业与相同方式共享条款适用于**你**对这些文件的使用 |
| ONNX 策略 | Apache-2.0（上游模型卡） | 随本站发布，署名见下 |
| MuJoCo WASM / onnxruntime-web / three.js | Apache-2.0 / MIT / MIT | 由 CDN 在运行时加载 |

上游项目：[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)、
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)、
[microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)。
本项目与 Pollen Robotics 无隶属或背书关系，"Microduck" 仅用于述明兼容性。
`;
    await writeFile(path.join(OUT, "README.md"), card, "utf8");
    console.log(`[build] 已写 Space 卡片（sdk: static）`);
  }
}

main().catch((e) => { console.error("打包失败:", e.message); process.exit(1); });
