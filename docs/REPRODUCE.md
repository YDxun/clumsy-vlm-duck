# 复现与扩展

这份文档记录**怎么从头把整套东西跑起来、怎么加东西、以及训练机上有哪些配置**。
命令都在仓库根目录下执行。

## 0. 环境

| | 本机（开发/验证） | 训练机（GPU） |
| --- | --- | --- |
| 系统 | Windows | Linux |
| Python | `F:\anaconda_ydx\python.exe`（3.12） | 系统 python3 |
| Node | v20 | — |
| 关键依赖 | `npm install`（web_sim 目录内） | `microduck_rl` 训练栈（mjlab / rsl_rl / mujoco） |

浏览器端物理是官方 MuJoCo WASM、策略是 onnxruntime-web、渲染是 three.js，
版本写在 `web_sim/package.json`，发布产物里的 CDN 版本号从同一处读取 ——
避免"本地测过的版本"和"线上加载的版本"不一致。

## 1. 从场景包到能跑的仿真器

```
duck_scenes/scenes/<场景id>/          场景包（人写）
   scene.xml          场地与物体，include 了 robot/robot_allcollisions.xml
   metadata.yaml      物体/区域/信标的 body 名 + 语义标签 + 相机
   tasks.yaml         任务：id / 中英文指令 / robot_spawn / success 判据 / failure
        │
        │  python web_sim/tools/flatten_scene.py <场景id>
        ▼
web_sim/assets/<场景id>/              摊平产物（生成物，不进仓库）
   scene.xml          内联 include、meshdir 指向 ../_shared、力矩限幅已烘入
   metadata.json      由 YAML 转换（浏览器不装 YAML 解析器）
   tasks.json         同上
   reference.json     用**磁盘上的 Python MuJoCo** 跑固定协议生成的参考值
web_sim/assets/_shared/*.stl          机器人网格：从上游固定 commit 取，逐个校验 sha256
web_sim/scenes.json                   场景清单（页面唯一的场景来源）
```

网格不在仓库里（上游是 CC BY-SA-NC 的硬件设计文件）。摊平时本地若没有，
会从 `pollen-robotics/microduck_rl` 的**固定 commit `2fa62b86fd08`** 下载，
并校验 `duck_scenes/robot/assets.SHA256SUMS` 里的 sha256。

> **为什么钉在 2fa62b8 而不是 HEAD**：上游在那之后改过机器人模型。
> 本项目 38 个网格的 git blob SHA 与该 commit 逐个一致、与 HEAD 逐个不一致 ——
> 钉错 revision 会让"本地验证过的几何"和"访客实际跑的几何"变成两套东西。
> 复核命令：`node web_sim/tools/check_mesh_pin.mjs`。

## 2. 加一个任务

任务的"要操作什么、送到哪里"**写在 `success` 判据里**，网页端从这里推导：

| 判据 | 推导出的目标 |
| --- | --- |
| `body_in_zone {body≠机器人, zone}` | 搬 `body` 到 `zone`（踢球、推方块） |
| `distance_xy_le {target}` | 去 `target`（走过去、去信标） |
| `body_in_zone {body=机器人, zone}` | 目标就是这个区域 |
| `ordered_zone_sequence {zones}` | 取第一个区域 |
| `robot_pose_region` / `upright` / `ever_fallen` | 没有目标物体（姿态/恢复类） |

步骤：

1. 在场景包的 `tasks.yaml` 里写 `id` / `instruction_zh` / `instruction_en`
2. **把目标写进 `success` 判据**（`distance_xy_le.target` 或 `body_in_zone.body/zone`）
3. 只有姿态/恢复类任务可以没有目标物体
4. 跑校验，绿了再提交：

```powershell
node web_sim\tools\check_task_schema.mjs
```

它直接调用运行时的推导函数（不抄第二份规则），并检查判据里引用的每个 body / zone
在 `metadata.yaml` 里真实存在 —— 拼错就报。

## 3. 加一个场景

1. 在 `duck_scenes/scenes/` 下新建场景包（照现有三个的格式写）
2. `python web_sim/tools/flatten_scene.py <新场景id>`（会自动登记进 `scenes.json`）
3. 在 `scenes.json` 里补 `title` / `description`（给人看的）
4. `node web_sim/tools/verify_wasm.mjs <新场景id>`

## 4. 训练一个新策略

策略在训练机的 `microduck_rl` 里练，导出的 ONNX 放进 `microduck/policies/`。
所有策略共用 `[1,61] → [1,14]` 契约，所以运行时可以任意时刻热切换。

```bash
# 训练（例：倒地起身）—— RTX 5090 单卡，4096 envs × 3000 轮约 1 小时
cd microduck_rl
export WANDB_MODE=disabled
.venv/bin/python -m mjlab_microduck.train_cli Mjlab-StandUp-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 3000 --agent.run-name standup_xxx

# 训练结束会写出 ONNX
cp logs/rsl_rl/microduck_stand/<run>/<run>.onnx ../microduck/policies/alpha_standup.onnx
```

新增策略后：在 `web_sim/policies.json` 登记，并在 `web_sim/src/actions.js` 加动作 token。
**只有本地实测有效的技能才标 `verified: true`** —— 发给模型一个它做不到的动作，
只会让它学会输出一个没反应的词。

### 观测契约（所有策略通用）

```
obs[0:3]    陀螺仪
obs[3:6]    重力方向（机器人本体系）
obs[6:20]   14 个关节位置 − 标称姿态
obs[20:34]  14 个关节速度
obs[34:48]  上一步动作
obs[48:51]  速度指令 (vx, vy, wz)
obs[51:61]  零填充（训练时就是零，不是漏填）
```

控制 50 Hz，每控制步 4 次物理子步（`phys_dt = 0.005`）。
`ctrl = 标称姿态 + 策略输出`，执行器力矩限幅烘焙在场景 XML 里（±0.605454）。

**步态地板**（实测）：vx 低于 **0.22 m/s**、wz 低于 **1.15 rad/s** 会被走路策略"吃掉"
（腿动、身体不动）。所以任何"慢速接近"必须用**脉冲驱动**（单拍给足速度，靠通断比
压平均），不能直接下发小速度。数据见 `web_sim/src/duck.js` 的 `GAIT_FLOOR`。

## 5. 验证（每层都能单独复跑）

```powershell
cd web_sim
node tools\check_task_schema.mjs                  # 任务 schema
node tools\check_mesh_pin.mjs                     # 本地网格 vs 上游 pin
node tools\verify_wasm.mjs duck_workspace_v1      # WASM 物理 vs Python（逐 body + 全 geom 哈希）
node tools\test_agent.mjs                         # 决策层单测（秒级）
node tools\verify_render.mjs                      # 渲染像素级断言
node tools\verify_agent.mjs                       # 闭环（含真 HTTP + 跨域）
node tools\verify_page.mjs                        # 页面交互
$env:DUCK_VLM_KEY="sk-..."; node tools\verify_real_vlm.mjs   # 真模型（无 key 自动跳过）
```

## 6. 发布到公网

```powershell
cd web_sim
node tools\build_site.mjs --out _site --assets=external --space-card=XenderYang/duck-vlm-simulator
node tools\smoke_site.mjs _site                                    # 本地冒烟（无 node_modules）
F:\anaconda_ydx\python.exe -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='XenderYang/duck-vlm-simulator', repo_type='space', folder_path='_site', commit_message='update')"
node tools\smoke_site.mjs https://xenderyang-duck-vlm-simulator.static.hf.space   # 线上冒烟
```

`--assets=external` **不打包机器人网格**（前文所述的许可原因），只打包我们自己的
场景 XML 与 Apache-2.0 的策略 ONNX，产物约 7 MB。

## 7. 训练机上的目录（交接用）

```
/root/microduck_sim/
   microduck_rl/          训练栈（mjlab / rsl_rl / mujoco）
   microduck/policies/    九个 ONNX + README（清单与训练出处）
   service.py             服务端监督进程（start/stop/status）
   sim_server.py          服务端仿真服务（LocalSim + 渲染 + 策略热插拔）
   local_kick.py          技能状态机（站稳→进动作→恢复）—— 网页端站位逻辑的参考
```

## 8. 踩过的坑（别重蹈）

详细记录在 `web_sim/README.md`，这里只列索引：

1. **旋转矩阵的行 ≠ 列**：读头部朝向取第 0 行（应为第 0 列），低头会变成抬头。
2. **`geom_size` 对 MESH 不是缩放**：它是 MuJoCo 自动填的包围盒半长。
3. **`mj_ray` 拿不到 geomid**：官方 WASM 要 9 个参数（C 只有 8 个），遮挡判断改用渲染器。
4. **步态地板**：低于它整条 twist 被吃掉（见上）。
5. **踢球站位精度**：踢球策略对球的位置**看不见**，要求 ±1.5 cm，只能靠规则闭环。
6. **最后 20 cm 看不见**：站位点上球在头摄前方 9 cm，早已出画，必须航位推算。
7. **脚本写错目录**：网格曾写到 `<out>/../_shared`，而 XML 解析到 `<out>/_shared`，
   靠旧文件还在才没暴露 —— 摊平脚本现在会把两个路径都打印出来对账。
8. **pin 错上游 revision**：网格取自 `2fa62b8`，不是 HEAD。
9. **PowerShell 5.1 的 UTF-8 陷阱**：`Get-Content -Raw | Set-Content -Encoding UTF8`
   会按 GBK 解码再写回，**把中文文档整篇转成乱码**。改文档用编辑器/`apply_patch`，
   不要用 PS 的读写往返。
