# DuckVLM 浏览器端仿真（零安装体验版）

> 目标：照 `rokbenko/quackd` 的**架构样式**做适配实现——静态页面、浏览器里跑物理与策略、
> 访客自带 API key——而不是把 quackd 的代码搬过来。本目录是我们自己的实现，
> 并把「哪些地方必须为浏览器改、为什么」记录在这里。

**当前进度：① 渲染器 ✅ ｜ ② 最小决策层 ⬜ ｜ ③ 页面 ⬜**

## 一、可行性（已实测，不是推测）

| 环节 | 结论 | 证据 |
| --- | --- | --- |
| 物理 | 官方 MuJoCo WASM（`@mujoco/mujoco@3.12.0`）能直接加载我们的场景 | `tools/verify_wasm.mjs`：与 Python 端 200 步物理**逐位一致**（trunk `0.13079, 0.00058, 0.05089`） |
| 策略 | `onnxruntime-web@1.29.0` 能跑我们的策略 ONNX | `tools/verify_policy.mjs` |
| 闭环 | 61 维观测 + 50 Hz + 4 次物理子步，行为与 Python 端等价 | 站立 4 s upright 1.000；前进 8 s 走 **1.0166 m**（Python 同条件 1.0189 m，差 2 mm） |
| 渲染 | three.js 按 `geom_xpos/geom_xmat` 直接画，**画面与物理逐像素对得上** | `tools/verify_render.mjs` 14 项全过：走得越远，画面里鸭子位移 64.4 px vs 几何投影 63.1 px（差 1.3 px） |
| 性能 | 浏览器里 **648 控制步/秒 = 13× 实时** | 同上 T6（含 ONNX 推理，1.54 ms/步） |
| LLM | 浏览器直连现有 Qwen 端点**被 CORS 允许** | `OPTIONS https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions` → `access-control-allow-origin: <origin>`，允许 POST + `authorization,content-type` |

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
├── index.html              # ① 验证页：四视角切换 + 站立/前进/转向 + 状态面板
├── src/
│   ├── duck.js             # DuckSim：MuJoCo WASM 物理 + ONNX 策略 + 观测/相机/几何查询
│   ├── view.js             # DuckView：geom -> three.js mesh，四种相机
│   └── app.js              # 页面装配 + window.__sim 测试钩子（像素分析都在页面内做）
├── tools/
│   ├── flatten_scene.py    # 摊平 duck_scenes 场景（含 include、mesh、力矩限幅）
│   ├── verify_wasm.mjs     # 无浏览器：WASM 能否加载、物理是否与 Python 一致
│   ├── verify_policy.mjs   # 无浏览器：物理+策略闭环
│   ├── verify_render.mjs   # 真 Chrome：14 项像素级断言
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

# 3) 三层验证，从便宜到贵
node tools\verify_wasm.mjs      duck_workspace_v1
node tools\verify_policy.mjs    alpha_walking 8 0.3 0
node tools\verify_render.mjs                     # 真 Chrome，出截图

# 4) 交互体验
node tools\serve.mjs            # 打开 http://127.0.0.1:8787/
```

## 六、还没做的（诚实清单）

- **② 最小决策层**：`duck_vlm/`（Python，约 2700 行）要在浏览器里重写成 JS——至少包含
  6~8 个动作 token + 步态解释器、提示词渲染、`intent.py` 的目标与意图推断。
  现在页面上的「站立/前进/转向」是写死的三个指令，**还不接受自然语言**。
- **③ 页面**：任务输入、动作日志、BYO key 面板、三视角录像。
- **VLM 观测图分辨率**：Python 头摄是 320×240（4:3），网页画布随窗口变。
  接决策层时要用独立的离屏 4:3 渲染目标，别直接把网页画布喂给 VLM。
- **体积**：单场景 20.6 MB mesh + 9.9 MB WASM + 0.76 MB ONNX。首次加载要进度条。
- **发布形态**：`index.html` 的 importmap 现在指向 `node_modules/`（本地离线可用）；
  发成静态站点时换成 jsDelivr 固定版本即可，代码不用动。
- **资产许可**：mesh 的许可仍需与上游确认；结论明确前建议照 quackd 的做法
  **不打包、运行时从上游固定 commit 拉取**。
