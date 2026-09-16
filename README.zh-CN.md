<p align="center">
  <img src="web_sim/favicon.svg" alt="Clumsy VLM Duck" width="150">
</p>

[English](README.md) | **简体中文**

# Clumsy VLM Duck

**在线试用（无需安装、无需注册）：<https://xenderyang-duck-vlm-simulator.static.hf.space/index.html>**

（也可以从 Space 页面进：<https://huggingface.co/spaces/XenderYang/duck-vlm-simulator>）

一个跑在浏览器里的机器鸭仿真器：物理（MuJoCo WASM）、策略（onnxruntime-web）、
视觉语言模型调用**全部在访客的浏览器里**，没有后端、没有服务器成本。
VLM 每一步输出一个**离散动作符号**（`FWD` / `TURN_L` / `LOOK_DOWN` / `DONE` …），
由解释器翻译成具体运动，执行后重新拍照，再喂回模型 —— 一个真正的闭环。

> 不用 API key 也能玩：规则模式完整可用。想试 VLM zero-shot，就带自己的 key
> （Qwen / Gemini / Claude 都支持），key 只存在你本机，只发给你选的厂商。

## 怎么开 VLM 模式

右边「决策」面板里点 **VLM 模式（填自己的 key）**，然后只做三件事：

1. 「用哪家的模型」保持默认的 **Qwen** 就行（想试 Gemini / Claude 再换）
2. 把 API key 粘进「把 API key 粘到这里」
3. 点「测试一下能不能用」——出现 ✅ 就是通了，直接点「▶ 开始」

模型名是个下拉，选一个即可；base URL 也会自动带对，都在
「高级：换模型 / 换地址」里，平时用不到。
key 只存在你本机浏览器的 localStorage，只发给你选的那家厂商的接口 ——
这个项目没有后端，也没有任何地方能收到你的 key。

<img src="web_sim/artifacts/vlm_panel_live_ok.png" alt="VLM 配置面板：选 Qwen、粘 key、点测试" width="390">

---

## 它能做什么

| | |
| --- | --- |
| 场景 | 3 个（工作台 / 家庭跨房间 / 障碍地形），页面里热切换，不重启 |
| 任务 | 23 条（场景包自带，下拉里标了 ★ 难度，简单的排在前面），也可以直接写一句自然语言 |
| 完成判定 | 精选任务的**成没成由场景包的判据说了算**（走进区域、到点停稳都实时判定），不靠模型自己喊完成 |
| 动作编排 | 「前进1米，再翻滚一次，最后跳舞」这类**没有目标物体**的指令逐步实测验收（走了几米、转了几度都量出来），不需要视觉、也不用调模型 |
| 动作词表 | 8 个基础 token + 通过本地验证的技能（翻滚 / 跳舞）；推、踢由规则闭环执行 |
| 交互 | 鼠标缩放/旋转/平移，一键三视角拼图，决策日志里能看到**模型当时看到的那张图** |

> **鸭子主要靠眼睛**：目标不在它视野里，它就得先转圈找。「穿越台阶 / 跨房间找球」
> 这类任务都要求目标（或路上的地标）能被它看见 —— 想跑通先挑 ★ 简单的，
> 或者直接用动作编排（那条路完全不依赖视觉）。

**已实测通过的场景任务**：`push_red_cube`（方块推进蓝区）、
`kick_ball_to_zone`（球踢进绿区），以及真模型 zero-shot 的
「go to the orange ball and stop next to it」。

## 仓库结构

```
web_sim/       浏览器仿真器（主线，所有验证都在这）
duck_scenes/   场景包 —— 仿真器的输入（场景 XML / 元数据 / 任务定义）
duck_vlm/      服务端 Python 版参考实现（同一套提示词、动作词表、意图解析）
duck_play/     Python 侧鸭子库（相机模型等，duck_vlm 依赖）
docs/          复现与扩展文档
THIRD_PARTY_NOTICES.md   第三方资产与许可
```

## 本地跑起来

```powershell
git clone https://github.com/YDxun/clumsy-vlm-duck && cd clumsy-vlm-duck/web_sim
npm install

# 摊平场景（网格不在仓库里，会自动从上游固定 commit 取并校验 sha256）
F:\anaconda_ydx\python.exe tools\flatten_scene.py duck_workspace_v1

# 策略 ONNX 不随仓库发布，按 policies.json 的清单放到 assets/policies/
#   alpha_walking / alpha_stand → pollen-robotics/microduck-policies (Apache-2.0)
#   alpha_standup             → 本项目自行训练的产物

node tools\serve.mjs            # http://127.0.0.1:8787/
```

> 更详细的工程记录（每一步为什么这么做、踩过哪些坑）在
> [`web_sim/README.md`](web_sim/README.md)，目前只有中文版。

## 凭什么说它是真的能跑

这个项目里每个结论都有一条能复跑的命令，而不是"看起来对"：

| 层 | 命令 | 现状 |
| --- | --- | --- |
| 任务 schema | `node tools/check_task_schema.mjs` | PASS（14 条任务的目标都能从成功判据推出） |
| 网格 pin | `node tools/check_mesh_pin.mjs` | PASS（38/38 与上游固定 commit 字节一致） |
| 物理一致性 | `node tools/verify_wasm.mjs <场景>` | PASS ×3（与 Python MuJoCo 逐 body + 全 geom 哈希一致） |
| 渲染 | `node tools/verify_render.mjs` | 14/14（像素级：画面里的鸭子就是物理里的鸭子） |
| 决策层单测 | `node tools/test_agent.mjs` | 124/124（含动作序列解析） |
| 闭环 | `node tools/verify_agent.mjs` | 37/37（含真 HTTP + 跨域） |
| 页面 | `node tools/verify_page.mjs` | 76/76（点按钮、填 key、测连通、换指令复跑、动作序列、下拼图） |
| 真模型 | `node tools/verify_real_vlm.mjs` | 7~8/8（真 Qwen3-VL，自己走到球边收工；最后一条看这趟走没走到 0.45 m） |
| 发布产物 | `node tools/smoke_site.mjs _site` | 8/8（无 node_modules，全 CDN） |
| 线上 HTML | `python tools/check_space_encoding.py` | PASS（线上 0 个坏字符） |

`npm run verify` 一次跑完（真模型那层没 key 会自动跳过）。

## 复现与扩展

- **加一个场景 / 加一个任务** → [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
- **训练一个新策略并导出 ONNX** → 同上（含服务器上的训练命令）
- **发布到公网** → 同上（一条命令打包 + 冒烟 + 上传）

## 已知限制（如实列出）

- **首次加载 ~20–35 s**：瓶颈是 22 MB 机器人网格的解析，不是网络。真机有 GPU 会快一些，
  但这个量级是这类方案的固有代价。还没有进度条。
- **官方的"跳舞"和"起身"不能直接用**（实测）：`happy_hop`（原"跳舞"）会把鸭子一点点
  歪到侧躺、最后躺平；`alpha_standup`（原"起身"）从那个姿态起不来。唯一稳的是
  `roulade`（翻滚）—— 3/3 次、约 4~5 s 能从躺平翻回站立。所以："跳舞"已从 VLM 词表
  和示例里撤下（动作序列里遇到会**跳过并说明原因**），「起身」按钮改用 roulade，
  动作序列收尾还会自动检查站姿，躺下就翻回来。
- **圆球不能用推的**：15 g 的球，走路接触就是一次冲量，推多慢都会被踢飞 —— 球走踢球策略，
  推留给方块/杯子这类不滚的物体。
- **Home 场景的跨房间搜索**还没跑过完整任务。
- 遮挡判断用渲染器实现（把目标渲成品红数像素），因为官方 WASM 的 `mj_ray` 拿不到 geomid。

## 许可

本项目**自己的代码**以 Apache-2.0 发布（见 `LICENSE`）。

机器人 3D 模型是上游的**硬件设计文件，CC BY-SA-NC**（非商业 + 相同方式共享），
**本仓库与线上站点都不分发**：浏览器运行时从上游固定 commit 取，并逐个校验 sha256。
ONNX 策略是 Apache-2.0，随站点发布。完整说明见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

架构样式参考 [rokbenko/quackd](https://github.com/rokbenko/quackd)（独立实现，未拷贝其代码）。
与 Pollen Robotics、Hugging Face 均无隶属或背书关系。
