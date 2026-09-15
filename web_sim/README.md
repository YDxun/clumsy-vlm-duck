# DuckVLM 浏览器端仿真（零安装体验版）

> 目标：照 `rokbenko/quackd` 的**架构样式**做适配实现——静态页面、浏览器里跑物理与策略、
> 访客自带 API key——而不是把 quackd 的代码搬过来。本目录是我们自己的实现，
> 并把「哪些地方必须为浏览器改、为什么」记录在这里。

**当前进度：① 渲染器 ✅ ｜ ② 最小决策层 ✅ ｜ ③ 页面 ✅**

四层验证当前状态：`verify_wasm` PASS ｜ `verify_render` 14/14 ｜ `test_agent` 87/87 ｜
`verify_agent` 37/37 ｜ `verify_page` 47/47

## 零、多场景与扩展性（怎么加一个新场景）

页面里的场景来自 **`web_sim/scenes.json`**，这是唯一的场景来源：

```powershell
# 1) 把场景包摊平（内联 include、烘力矩限幅、登记清单、生成跨引擎参考数据）
F:\anaconda_ydx\python.exe tools\flatten_scene.py duck_obstacle_v1
# 2) 刷新页面 —— 下拉里就有了，不用改代码、不用重启服务
```

摊平时会做四件事：

| 产物 | 作用 |
| --- | --- |
| `assets/<id>/scene.xml` | 内联了 include、力矩限幅已烘进去的自包含场景 |
| `assets/_shared/*.stl` | **所有场景共用**一份机器人网格（3 个场景 22 MB，不是 66 MB） |
| `assets/<id>/{metadata,tasks}.json` | 自然语言→body 的解析表、任务下拉的 id |
| `assets/<id>/reference.json` | 由**磁盘上的 Python MuJoCo** 跑固定协议生成的参考值 |
| `scenes.json` | 场景清单（标题/描述由人手写，体积等字段每次重新测量） |

`reference.json` 是这套东西能扩展的关键：验证脚本不再写死某几个 body 的期望值，
而是逐 body + 全 geom 位置哈希地比。三个场景都是这样验的：

```
一致性 : trunk OK | 24 个 body OK | geom 位置哈希 OK
```

**自定义/外部场景**：把摊平后的目录 + 一份 `scenes.json` 放到任意静态托管（S3、GitHub Pages、
HF Space…），在页面「自定义场景清单 URL」里填进去即可；清单里的 `assetBase` 指到场景目录所在位置。

### 换场景为什么不用重启

换场景只需重建 MuJoCo 模型；ONNX 会话、渲染器、灯光、相机都复用。
实测（无头 Chrome，软件光栅化，最坏情况）：

| | 耗时 |
| --- | --- |
| 首次进入一个场景 | ~16 s（20 MB STL / 21 万顶点解析，全是这一步） |
| **换回已加载过的场景** | **~1.2 s**（模型在缓存里，渲染重建 12 ms） |

换场景期间会暂停渲染循环 —— 不暂停的话软件光栅化会抢走 CPU，同样是解析，
有头测试里要 24 s。缓存最多留 3 个场景的模型，被挤出去时会显式 `dispose()`
（embind 对象不主动释放的话 MuJoCo 堆只涨不降）。

### 美化

场景包里的材质全是中性灰，实物鸭子不是这样。渲染层按 body 名重新配色
（嘴橙、脚深、腿灰、外壳奶白），另外加了三点光、天空渐变、雾、地面网格和
鸭子脚下的接触阴影。一键「原始配色」可以切回场景原本的材质做对照 ——
**这些只动渲染，不动物理**。

### 手动遥控为什么以前"没用"

实测（3 秒定速，见 `duck.js` 里 `GAIT_FLOOR` 的注释）：

```
指令 wz 0.80 -> 实际 0.056 rad/s   7%   ← 「原地转 0.8 没用」的真相
指令 wz 1.10 -> 实际 0.083 rad/s   8%
指令 wz 1.15 -> 实际 0.703 rad/s  61%   ← 拐点
指令 wz 1.50 -> 实际 0.940 rad/s  63%   ← 我们 TURN token 用的值
指令 wz 2.00 -> 实际 0.940 rad/s  47%   ← 物理饱和，再大是浪费
```

低于步态地板的指令会被走路策略"吃掉"（腿在动、身体不动）。现在
`normalizeTwist()` 会把整条 twist 抬到地板之上，手动按钮也改成下发
**离散 token**（一次点击 = 走一步 0.043 m / 转 32°），而不是一条永不停止的速度指令。

## 一、可行性（已实测，不是推测）

| 环节 | 结论 | 证据 |
| --- | --- | --- |
| 物理 | 官方 MuJoCo WASM（`@mujoco/mujoco@3.12.0`）能直接加载我们的场景 | `tools/verify_wasm.mjs`：与 Python 端 200 步物理**逐位一致**（trunk `0.13079, 0.00058, 0.05089`） |
| 策略 | `onnxruntime-web@1.29.0` 能跑我们的策略 ONNX | `tools/verify_policy.mjs` |
| 闭环 | 61 维观测 + 50 Hz + 4 次物理子步，行为与 Python 端等价 | 站立 4 s upright 1.000；前进 8 s 走 **1.0166 m**（Python 同条件 1.0189 m，差 2 mm） |
| 渲染 | three.js 按 `geom_xpos/geom_xmat` 直接画，**画面与物理逐像素对得上** | `tools/verify_render.mjs` 14 项全过：走得越远，画面里鸭子位移 64.4 px vs 几何投影 63.1 px（差 1.3 px） |
| 性能 | 浏览器里 **648 控制步/秒 = 13× 实时** | 同上 T6（含 ONNX 推理，1.54 ms/步） |
| LLM | 浏览器直连现有 Qwen 端点**被 CORS 允许** | `OPTIONS https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions` → `access-control-allow-origin: <origin>`，允许 POST + `authorization,content-type` |
| 决策 | 8 个离散 token 的闭环在浏览器里跑通 | `tools/verify_agent.mjs` 33 项全过；`tools/test_agent.mjs` 87 项全过 |

## 二、为浏览器做的**两处必要适配**（踩出来的，别改回去）

### 1. 执行器力矩上限必须烘进 XML
`_dev_cockpit/sim_server.py` 的 `LocalSim` 会把力矩上限从 XML 的 `±0.96` 覆写成
`±(KT*1.75)=±0.605454`（策略就是按这个权限训练的）。而官方 WASM 的
`model.actuator_forcerange` / `actuator_forcelimited` 在 JS 侧**不可写**，连读都会抛
`BindingError: unknown type N10emscripten11memory_viewIbEE`。
→ 所以 `flatten_scene.py` 在摊平阶段就把这条覆写写进每个 `<position>` 执行器。
（不补这一步，浏览器里的驱动力矩比策略预期高 59%。）

### 2. `mjtObj` 枚举必须从模块读，不能硬编码
曾把 `mjOBJ_SENSOR` 写成 `12`（实际是 `mjOBJ_HFIELD`），于是 `sensor_adr[undefined]`
退化成 `undefined`、陀螺仪观测**恒为 0**，策略失去角速度反馈，**鸭子一走就摔**
（upright 1.0 → 0.30）。正确值从 `mujoco.mjtObj.mjOBJ_SENSOR.value` 读（= 20）。
**这个 bug 不报任何错**，只是鸭子走得歪。

### 3. `geom_size` 对 MESH 型 geom **不是缩放**
MuJoCo 会给 mesh geom 自动填 `geom_size` = 该 mesh 的包围盒半长，而真正的缩放存在
`mesh_scale`（本场景全为 1）。照 `geom_size` 去 scale 会把鸭子放大成怪物。

## 三、① 渲染器：怎么做的

官方 `@mujoco/mujoco` WASM **没有光栅化器**（没有 `mjr_render`/`mjr_readPixels`，
连 `mjv_makeScene` 都没有），所以画面得自己出：

1. 建场景时遍历 `model.geom_*`，每个 geom 建一个 three.js mesh；顶点直接用
   `model.mesh_vert / mesh_face`——MuJoCo 已经解析过 STL 了，不需要第二个副本、也不用 STL 解析器。
2. 每帧只把 `data.geom_xpos` / `data.geom_xmat` 灌进 mesh 的变换矩阵。
3. MuJoCo 是 z-up、three.js 是 y-up，整个 world 组绕 x 轴转 -90°：**(X,Y,Z)→(X,Z,-Y)**，
   上面的算术全部保持 MuJoCo 坐标系。

### 四种「看得见」的相机（场景相机都从模型里读，不硬编码数字）

| 模式 | 来源 | 用途 |
| --- | --- | --- |
| `workspace` | 场景自带 `cam_workspace`（3/4 全景，fovy 52） | **一眼看全场景 + 鸭子在哪** |
| `overhead` | 场景自带 `cam_overhead`（正上方 5.2 m，fovy 50） | 平面图视角，复跑打分时看路线 |
| `follow` | 我们的轨道参数 | 看鸭子动作细节 |
| `duck` | 逐字复刻 Python `render_headcam` | **将来喂给 VLM 的就是这张图** |

### 头摄的三个数字（web 与 Python 必须一致，否则没法排查「同一句指令两边不一样」）

```
位置 = 头体原点 + 朝向*0.09 m + 世界上方*0.03 m
视线 = 头体朝向再向下俯 20°     ← 固定俯角，不跟着身体俯仰走
垂直 FOV = 45°（模型里的 vis.global_.fovy）
```

> 我先写错过一次：拿头体的真实 pitch 当相机俯仰、FOV 写死 62°，画面 2/3 是天空。
> 又误判过一次 Python 端「朝后看」——实测 `scn.camera[0].pos == campos`、`forward == v`，
> Python 是对的，MuJoCo 的 azimuth/elevation 描述的是**视线方向**而不是相机方位。

## 四、目录

```
web_sim/
├── index.html              # ① 验证页：四视角切换 + 站立/前进/转向 + 状态面板（③ 扩成正式页面）
├── src/
│   ├── duck.js             # DuckSim：MuJoCo WASM 物理 + ONNX 策略 + 观测/相机/几何查询
│   ├── view.js             # DuckView：geom -> three.js mesh，四种相机
│   ├── app.js              # 页面装配 + window.__sim 测试钩子（像素分析都在页面内做）
│   └── （② 决策层，见下）
│       ├── actions.js      # 8 个离散 token + 别名 + 容错解析
│       ├── interpreter.js  # token -> 速度指令 / 锁存的头部姿态
│       ├── state.js        # 本体感知 + 「只有看见才给距离」的闸门
│       ├── scene.js        # metadata.json 索引：自然语言 -> body
│       ├── intent.js       # 自由文本 -> 意图（approach/manipulate/pose/…）
│       ├── prompt.js       # 与 Python 端同字段的 zero-shot 提示词
│       ├── llm.js          # BYO key：Qwen/OpenAI 兼容、Gemini、Claude
│       ├── rules.js        # 可消融的规则层 + 无 key 的 reflex 兜底
│       └── agent.js        # 感知-推理-动作闭环状态机
├── tools/
│   ├── flatten_scene.py    # 摊平 duck_scenes 场景（含 include、mesh、力矩限幅）
│   ├── verify_wasm.mjs     # 无浏览器：WASM 能否加载、物理是否与 Python 一致
│   ├── verify_policy.mjs   # 无浏览器：物理+策略闭环
│   ├── verify_render.mjs   # 真 Chrome：14 项像素级断言
│   ├── test_agent.mjs      # 纯 Node：决策层 87 项单元测试
│   ├── verify_agent.mjs    # 真 Chrome：33 项闭环断言（含真 HTTP + 跨域）
│   ├── verify_page.mjs     # 真 Chrome：30 项页面断言（点按钮、填 key、下载拼图）
│   ├── verify_real_vlm.mjs # 真 Chrome + 真模型：让 Qwen 自己控制鸭子（无 key 则 SKIP）
│   ├── mock_llm.mjs        # 假 OpenAI 端点，验证 BYO key 链路
│   └── serve.mjs           # 极简静态服务器（.wasm/.onnx 的 MIME 很关键）
├── artifacts/              # verify_render 出的四视角截图
└── assets/                 # 生成物（mesh、ONNX），默认不进 git
```

## 五、怎么跑

```powershell
cd web_sim; npm install

# 1) 摊平场景（内联 include、复制用到的 mesh、写入力矩限幅）
F:\anaconda_ydx\python.exe tools\flatten_scene.py duck_workspace_v1

# 2) 准备策略（本仓库不跟踪 ONNX）
copy ..\research\artifacts\official_policies\alpha_stand.onnx  assets\policies\
copy ..\research\artifacts\official_policies\alpha_walking.onnx assets\policies\

# 3) 四层验证，从便宜到贵
node tools\verify_wasm.mjs      duck_workspace_v1
node tools\verify_policy.mjs    alpha_walking 8 0.3 0
node tools\verify_render.mjs                     # 真 Chrome，出截图
node tools\test_agent.mjs                        # 决策层单测，秒级
node tools\verify_agent.mjs                      # 真 Chrome 跑闭环，约 4 分钟
node tools\verify_page.mjs                       # 真 Chrome 点页面，约 1 分钟

# 4) 交互体验
node tools\serve.mjs            # 打开 http://127.0.0.1:8787/
```

## 六、③ 页面：访客看到什么

打开就是三个区域：

| 区域 | 内容 |
| --- | --- |
| 左：画面 | 四种视角（全景 / 跟随 / 上帝视角 / 鸭子眼）+ 三视角拼图下载 |
| 右上：任务与决策 | 任务下拉（**直接列场景包的 task id**，不是让人手打）、自由文本输入框、规则/VLM 模式切换、BYO key 表单、开始/暂停/复位/停止 |
| 右下：状态与日志 | 实时状态（阶段/决策次数/位置/朝向/直立度/最近距离）+ **决策日志卡片**（每张卡片就是模型当时看到的那张图 + 它选的 token + 规则有没有接管） |

几个刻意的取舍：

- **任务用下拉而不是输入框**：输入对不上 task id 就没法评分，是之前踩过的坑；
  想随手写就用「（用下面那句话）」选项，`scene.js` 会把「去红方块」解析成 `obj_cube_red`。
- **key 只在切到 VLM 模式时出现**，`type=password`，存本机 localStorage，
  请求直接从浏览器发往所选厂商，我们的服务器不参与。
- **决策日志一定带图**：不然「模型为什么选了 TURN_L」永远说不清。
- **三视角拼图**：一键把「全景 / 全场俯视 / 鸭子眼」拼成一张 PNG，
  汇报和录 demo 用同一张图，不用再多开一个窗口。

## 七、还没做的（诚实清单）

- **三视角录像还没有视频**：现在只能下载一张拼图 PNG。要录完整过程，
  下一步用 `canvas.captureStream()` + `MediaRecorder` 把拼接画面录成 WebM。
- **技能 token 还没进最小集**：KICK_L/KICK_R/SIT/STAND_UP/ROLL/DANCE 需要各自的 ONNX
  状态机（`_dev_cockpit/local_kick.py` 那一套），所以「踢球进区域」「摔倒起身」这类任务
  目前跑不了。词表放大的代价是 zero-shot 选择变难，值得单独一步做。
- **遮挡判断没用 `mj_ray`**：这个 WASM 构建的 `mj_ray` 要 9 个参数（C 只有 8 个），
  两个候选槽位都写不进 geomid，拿不到「打到的是谁」就没法判断是不是目标自己。
  改成 `view.maskVisible()`：把目标临时换成不受光照影响的品红色渲一张头摄图，
  数像素 —— 它测的就是 VLM 那张图，连 FOV 和贴图遮挡都算进去，比射线更贴近事实。
- **VLM 观测图分辨率**：Python 头摄是 320×240（4:3），网页画布随窗口变。
  `captureHeadCam()` 已经用 4:3 视口单独渲一次再缩到 320×240，两边构图一致。
- **体积**：单场景 20.6 MB mesh + 9.9 MB WASM + 0.76 MB ONNX。首次加载要进度条。
- **发布形态**：`index.html` 的 importmap 现在指向 `node_modules/`（本地离线可用）；
  发成静态站点时换成 jsDelivr 固定版本即可，代码不用动。
- **资产许可**：mesh 的许可仍需与上游确认；结论明确前建议照 quackd 的做法
  **不打包、运行时从上游固定 commit 拉取**。

## 八、② 决策层：做了什么、怎么验证的

### 8 个 token（`src/actions.js`）

```
FWD / BACK / TURN_L / TURN_R / STOP / LOOK_DOWN / HEAD_CENTER / DONE
```

语义、时长、速度指令都对齐 `duck_vlm/actions.py`，只是砍到最小可用集。
技能类 token（踢球、翻滚、坐下、起身）**有意不放进最小集**：词表越大，
zero-shot 越容易选错，而且每个技能都要带一套 ONNX 状态机。

### 「看不见就不给距离」

`src/state.js` 只在**目标真的落在头摄画面里**、且视线没被挡时，
才把 `range/bearing/elevation` 写进提示词；否则是 `not_found`。
鸭子没有深度传感器，这条闸门就是防止仿真真值偷偷泄漏给模型。

验证方式不是读代码，而是**对像素**：把头摄画面里隐藏球再渲一张，差异像素就是球在
画面里的位置。正对球时状态给的 `uv` 与像素质心差 **1.2%**；背对球时状态是 `not_found`，
差分也是 **0 像素**。

遮挡也一起验了：把球放在圆柱正后方、鸭子站另一边，视线穿过圆柱轴心 ——
球**仍在相机视锥内**（uv 0.50, 0.57），但状态必须是看不见。
这条用例专门用来区分「真的被挡住」和「只是出画」：换成 `mj_ray` 那套会误判成可见。

### 低头（LOOK_DOWN）覆盖多远（实测）

归一化图像里的 v 坐标（0=上沿，1=下沿，`!` = 出画）：

```
偏移\距离   0.14   0.18   0.22   0.30   0.40   0.55   0.90
   0       !1.95  !1.48  !1.20   0.86   0.62   0.42   0.21   ← 正常前视
   0.35    !1.27   0.98   0.79   0.52   0.30   0.10  !-0.14
   0.60     0.87   0.65   0.49   0.24   0.02  !-0.21  !-0.50  ← LOOK_DOWN
   1.00     0.35   0.17   0.01  !-0.26 !-0.54  !-0.88  !-1.39
```

`LOOK_DOWN`(+0.60) 覆盖 **0.14~0.42 m**，与提示词里写的「0.15-0.45 m」一致。
实测复现并解决了原始死锁：贴到球前 0.18 m → 球掉到画面下方（差分 0 像素，状态 `not_found`）
→ 规则触发 `LOOK_DOWN` → 头部关节压到 `head_pitch` ctrl 0.949 → 球回到画面 `uv=(0.49, 0.68)`。

### 规则层（`src/rules.js`，每条可单独关掉做消融）

| 规则 | 作用 |
| --- | --- |
| `fallen_stop` | 摔倒时拒绝位移动作 |
| `anti_stuck` | 连续 FWD 但位移 < 2 cm → 改左转脱困 |
| `anti_spin` | 左右互摆地原地转 → 改前进换视角 |
| `look_down_when_lost` | 0.6 m 内丢目标 → 强制低头找 |

没有 key 时用 `reflexToken()` 兜底（Python 端 dry_run 的对应物），
所以**离线也能演示完整闭环**。

### 又踩到一个坑：旋转矩阵的行 ≠ 列

读头部朝向时写成 `xmat[body*9 + 0..2]`（第 0 **行**），而「局部 x 轴在世界里的方向」
是旋转矩阵的第 0 **列**（下标 0/3/6）。两者互为转置：小角度下几乎一样，
几十度就直接反向 —— 于是「低头」看起来像「抬头」，头摄画面 2/3 是天空。
Python 侧同条件测到 -60.3° 俯角，JS 侧测到 +50.5°，一比就露馅。
现在 `duck.js` 的 `bodyForward()` 统一走列，并在 `verify_agent.mjs` 的 T0 里
加了回归测试（默认俯角 29.3°、低头 → 56.0°、抬头 → -18.7°）。

## 九、真模型端到端（Qwen3-VL-Plus 实测）

`tools/verify_real_vlm.mjs` 把真 key 塞给浏览器里的 agent，让模型自己飞。
key 只从环境变量读，**不落盘、不进仓库**：

```powershell
$env:DUCK_VLM_KEY="sk-…"
node tools\verify_real_vlm.mjs "go to the orange ball and stop next to it" 12
```

测的是百炼（ap-southeast-1）的 `qwen3-vl-plus`，base 用
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。一次真实回合：

| 步 | 模型原文 | 延迟 | 它看到的目标 | 它选的动作 |
| --- | --- | --- | --- | --- |
| 0 | `"FWD"` | 1000 ms | 0.90 m / 0.00 rad | 前进 |
| 3 | `"FWD"` | 919 ms | 0.68 m / 0.03 rad | 前进 |
| 6 | `"FWD"` | 631 ms | 0.46 m / -0.03 rad | 前进 |
| 7 | `"FWD"` | 533 ms | 0.39 m / -0.00 rad | 前进 |
| 8 | `"DONE"` | 544 ms | 0.32 m / -0.09 rad | **判断到了，收工** |

9 次决策、1120 个控制步、墙钟 51 s（无头 Chrome 用软件光栅化，真机上有 GPU 会快得多）。
距离 0.90 m → 0.32 m 单调缩短，直立度 1.000，**0 条解析失败**，
并且它自己在 0.32 m 处收手——不是走到撞上去才停。

`verify_real_vlm.mjs` 的 8 项断言全过；没设 key 时它打印 SKIP 并正常退出，
所以可以放心放进 `npm run verify`。

> 顺带说明为什么值得单独跑这一步：假端点只能证明「链路通」，
> 证明不了「提示词好不好、模型会不会玩」。上面这张表是那一步的证据。

### 第二局：中文自由文本 + 要转身 + 要分辨红/蓝

```
node tools\verify_real_vlm.mjs "去红方块旁边停下" 12
```

先由 `scene.js` 把「红方块」解析成 `obj_cube_red`（场景包里写的是「红色方块」，
用户少说一个字也能对上；蓝色方块是干扰项，同一句里若出现「蓝色区域」不会被误配）。

| 步 | 模型原文 | 延迟 | 它看到的 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | `"TURN_R"` | 933 ms | 看不见 | 方块在视野外，它自己决定先右转去找 |
| 1–3 | `"FWD"` | 562–1186 ms | 0.95 → 0.81 m | 找到之后开始靠近 |
| 4 | `"TURN_R"` | 530 ms | 0.75 m / -0.30 rad | 航向偏了，补一次修正 |
| 5–10 | `"FWD"` | 504–1005 ms | 0.74 → 0.39 m | 稳定收口 |
| 11 | `"DONE"` | 955 ms | 0.33 m / 0.24 rad | 判断到了，收工 |

终点 **(0.44, -0.46)**，红方块在 (0.70, -0.65)、橙色球在 (0.90, 0.00) ——
它朝的是方块不是球，说明**红/蓝/橙是靠画面分出来的，不是靠真值**。
12 次决策 0 条解析失败，直立度 1.000。
