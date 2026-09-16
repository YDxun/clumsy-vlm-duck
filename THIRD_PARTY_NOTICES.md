# 第三方组件与资产声明

> **English summary** — This project's own code is Apache-2.0. The robot 3D models are
> upstream **hardware design files licensed CC BY-SA-NC** (NonCommercial + ShareAlike);
> they are **not distributed by this repository or by the live site** — the browser
> fetches them at runtime from a pinned upstream commit
> (`pollen-robotics/microduck_rl@2fa62b86fd08`) and each file is verified against the
> sha256 manifest in `duck_scenes/robot/assets.SHA256SUMS`, so the BY / SA / NC terms
> apply to *your* use of those files rather than to our distribution. The ONNX policies
> are Apache-2.0. Details in Chinese below.

本项目（Clumsy VLM Duck / DuckVLM 浏览器仿真）自己的代码以 Apache-2.0 发布。
下面列的是**别人做的东西**、它们的许可、以及我们和它们的关系。

一句话结论：**CC BY-SA-NC 的机器人网格不在本仓库、也不在线上站点里**——
浏览器运行时从上游固定 commit 取，所以"署名 / 非商业 / 相同方式共享"这三条
适用于**你**对这些文件的使用，与本项目的分发无关。

---

## 1. 机器人 3D 模型（38 个 STL）—— **CC BY-SA-NC，本仓库不分发**

| | |
| --- | --- |
| 来源 | [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) `src/mjlab_microduck/robot/microduck/assets/` |
| 许可 | **Creative Commons BY-SA-NC**（NonCommercial + ShareAlike） |
| 依据 | 上游 README 第 190–191 行原文：<br>`This project is licensed under the Apache 2.0 License. See the LICENSE file for details.`<br>`Hardware design files are licensed under Creative Commons BY-SA-NC.`<br>注意：仓库根有 Apache-2.0 的 LICENSE，但 README 对**硬件设计文件**做了明确除外；上游未注明 CC 版本号，文件里也没有许可头。 |
| 我们的做法 | **不分发**。`web_sim/scenes.json` 里的 `meshBase` 指向上游固定 commit<br>`https://raw.githubusercontent.com/pollen-robotics/microduck_rl/2fa62b86fd08/src/mjlab_microduck/robot/microduck/assets`<br>由访客浏览器在运行时下载，本站不存储、不打包、不镜像。 |
| 你要注意 | 只要你在用这些网格，**BY（署名）/ SA（相同方式共享）/ NC（非商业）**就适用于你。想商用请自行替换模型或另行获得授权。 |
| 完整性 | 本地摊平（`flatten_scene.py`）逐个校验 `duck_scenes/robot/assets.SHA256SUMS` 里的 sha256，对不上就报错而不是静默使用。浏览器端目前只校验文件名（可再加一层，待办）。 |

固定 commit `2fa62b86fd08` 是**刻意钉死的，而且必须钉在这一版**：它是本项目场景与策略
实际验证时用的那份几何。上游在这之后改过机器人模型（2026-07-28 之后的提交），
钉到 HEAD 会让"本地验证过的几何"和"访客实际跑的几何"变成两套东西。
依据：38 个网格的 git blob SHA 与该 commit **逐个一致**、与 HEAD **逐个不一致**
（比对脚本 `web_sim/tools/check_mesh_pin.mjs`）。升级要显式改 pin 并重跑验证。

## 2. ONNX 策略（9 个）—— Apache-2.0，随本项目发布

| | |
| --- | --- |
| 来源 | 官方策略来自 [`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies)（模型卡 front matter 为 `license: apache-2.0`）；`alpha_standup.onnx` 是本项目自己训练的产物 |
| 许可 | Apache-2.0 |
| 我们的做法 | 随站点发布，`web_sim/policies.json` 里登记清单与用途 |

## 3. 运行时通过 CDN 加载的库

| 组件 | 许可 | 用途 |
| --- | --- | --- |
| [MuJoCo](https://github.com/google-deepmind/mujoco)（官方 WASM 构建） | Apache-2.0 | 物理仿真 |
| [onnxruntime-web](https://github.com/microsoft/onnxruntime) | MIT | 跑策略 |
| [three.js](https://github.com/mrdoob/three.js) | MIT | 渲染 |

三者在发布产物里都不打包，由 `index.html` 的 importmap 指向 jsDelivr，
版本号从 `web_sim/package.json` 读取（避免线上与测试版本不一致）。

## 4. 参考与致谢

- [`rokbenko/quackd`](https://github.com/rokbenko/quackd) —— 浏览器内跑物理 + BYO key 的**架构样式**参考。
  本项目是独立实现，未拷贝其代码；许可处置（运行时取上游资产、不分发）与之一致。
- [`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck) —— 机器人本体固件栈（Apache-2.0），我们参照其 JSON-RPC 契约。
- [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) —— 训练栈（代码 Apache-2.0）与策略的观测契约（`obs[61] → act[14]`，50 Hz）。

本项目与 Pollen Robotics、Hugging Face 均无隶属或背书关系；
"Microduck" 仅用于述明兼容性。
