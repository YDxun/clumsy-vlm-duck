# DuckVLM 浏览器端仿真（零依赖体验版）

> 目标：照 `rokbenko/quackd` 的**架构样式**做适配实现——静态页面、浏览器内跑物理与策略、
> 访客自带 API key——而不是把 quackd 的代码搬过来。
> 本目录是**我们自己的实现**，并把"哪些地方必须为浏览器改"记录在这里。

## 为什么可以这么做（已实测，不是推测）

| 环节 | 结论 | 证据 |
| --- | --- | --- |
| 物理 | 官方 MuJoCo WASM（`@mujoco/mujoco@3.12.0`）**能直接加载我们的场景** | `tools/verify_wasm.mjs`：与 Python 端 200 步物理**逐位一致**（trunk `0.13079, 0.00058, 0.05089`） |
| 策略 | `onnxruntime-web@1.29.0` 能跑我们的策略 ONNX | `tools/verify_policy.mjs` |
| 闭环 | 61 维观测 + 50 Hz + 4 次物理子步，**行为与 Python 端等价** | 站立 4 s upright 1.000；前进 8 s 走 **1.0166 m**（Python 端同条件 1.0189 m，差 2 mm） |
| LLM | 浏览器直连现有 Qwen 端点**被 CORS 允许** | `OPTIONS https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions` → `access-control-allow-origin: <origin>`，允许 POST + `authorization,content-type` |

## 为浏览器做的两处**必要适配**（踩出来的）

### 1. 执行器力矩上限必须烘进 XML
`_dev_cockpit/sim_server.py` 的 `LocalSim` 会把力矩上限从 XML 的 `±0.96` 覆写成
`±(KT*1.75)=±0.605454`（策略就是按这个权限训练的）。而官方 WASM 的
`model.actuator_forcerange` / `actuator_forcelimited` 在 JS 侧**不可写**、连读都会抛
`BindingError: unknown type N10emscripten11memory_viewIbEE`。
→ 所以 `flatten_scene.py` 在摊平阶段就把这条覆写写进每个 `<position>` 执行器。
（不补这一步，浏览器里的鸭子力矩比策略预期高 59%。）

### 2. `mjtObj` 枚举必须从模块读，不能硬编码
曾把 `mjOBJ_SENSOR` 写成 `12`（实际是 `mjOBJ_HFIELD`），于是 `sensor_adr[undefined]`
退化成 `undefined`、陀螺仪观测**恒为 0**，策略失去角速度反馈，**鸭子一走就摔**
（upright 1.0 → 0.30）。正确值从 `mujoco.mjtObj.mjOBJ_SENSOR.value` 读（= 20）。

## 目录

```
web_sim/
├── tools/
│   ├── flatten_scene.py    # 把 duck_scenes 的场景摊平成浏览器可加载形式（含力矩限幅）
│   ├── verify_wasm.mjs     # 无浏览器验证：WASM 能否加载我们的场景、物理是否与 Python 一致
│   └── verify_policy.mjs   # 无浏览器验证：WASM 物理 + onnxruntime-web 策略的闭环
├── assets/                 # 生成物（mesh 与策略，默认不进 git）
└── package.json
```

## 怎么跑

```powershell
cd web_sim; npm install

# 摊平场景（内联 include、复制用到的 mesh、写入力矩限幅）
F:\anaconda_ydx\python.exe web_sim\tools\flatten_scene.py duck_workspace_v1

# 准备策略（本仓库不跟踪 ONNX，需要自己放）
copy research\artifacts\official_policies\alpha_stand.onnx   web_sim\assets\policies\
copy research\artifacts\official_policies\alpha_walking.onnx web_sim\assets\policies\

# 验证（不需要浏览器）
node web_sim\tools\verify_wasm.mjs   duck_workspace_v1
node web_sim\tools\verify_policy.mjs alpha_stand  4 0   0     # 站立
node web_sim\tools\verify_policy.mjs alpha_walking 8 0.3 0    # 前进 0.3 m/s
```

## 还没做的部分（诚实清单）

- **渲染**：官方 `@mujoco/mujoco` 只暴露物理与 `mjv_updateScene`，**没有光栅化器**
  （没有 `mjr_render` / `mjr_readPixels`）。页面需要自己写 WebGL 渲染器
  （quackd 的 `view.js` 就是干这个的），或换用带完整渲染的第三方 WASM 构建。
- **决策层**：`duck_vlm/` 是 Python（约 2700 行），浏览器里要重写成 JS——至少包括
  动作词表/解释器、主循环、提示词渲染，以及 `intent.py` 的目标与意图推断。
- **页面与交互**：尚未编写。
- **体积**：单场景 20.59 MB mesh + 10.1 MB WASM；首次加载需要进度条与长效缓存
  （quackd 同样是 22 MB mesh 级别，这是这类方案的固有代价）。
- **资产许可**：mesh 的许可仍需与上游确认；结论明确前建议照 quackd 的做法
  **不打包、运行时从上游固定 commit 拉取**。
