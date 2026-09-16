# 复现与扩展

> **English note** — This engineering document is Chinese-only for now (it is the
> working log, not the pitch). The [English README](../README.md) covers what the
> project is, how to run it and how to verify it; the commands below are copy-pasteable
> regardless. Translations are welcome.

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
4. 加一个 `difficulty: easy | medium | hard` —— 下拉框会显示成 ★ / ★★ / ★★★，
   简单的排在前面。**能用位姿判据的就别用视觉判据**：鸭子主要靠眼睛找目标，
   "往前 0.5 米""原地左转 90 度"这类靠 `robot_pose_region` 的判据它一定有办法做到，
   而"跨房间找球""绕过障碍"要求目标或地标先进入视野，那是另一个难度级的事
5. 跑校验，绿了再提交：

```powershell
node web_sim\tools\check_task_schema.mjs
```

它直接调用运行时的推导函数（不抄第二份规则），并检查判据里引用的每个 body / zone
在 `metadata.yaml` 里真实存在 —— 拼错就报。

### 判据怎么判"完成"

精选任务**由判据说了算**，不靠模型自己喊 DONE（`web_sim/src/goal.js` 把
`validation_core.py` 的原子判据搬到了浏览器端，语义逐条对齐）：

* 每个控制步都算一次判据，满足后再保持 `success_hold_s`（默认 0.5 s）才判成；
* 判据里有 `speed_xy_le`（"要停住"）时，**位置到位后由 agent 主动刹住**再等计时 ——
  不刹的话鸭子会自己走出区域，"进了区域"这一条反而永远满足不了（踩过）；
* 规则层的 DONE 阈值取判据里的到达半径（`goal.arriveRangeM`），不再是写死的 0.35 m；
* 模型/规则层喊 DONE 但判据没满足时**不认**，会在日志里写明"说完成但判据未满足"。

### 没有目标物体的指令：动作序列

「前进1米，再翻滚一次，最后跳舞」这种指令由 `web_sim/src/sequence.js` 接管：
解析成步骤 → 逐步执行 → **每步用世界状态验收**（走了几米、转了几度、技能跑完没有）。
它不走决策层，所以不花 API 钱，也不依赖视觉。判据不认识这句指令时不会误判成
"去找 ball"（那正是以前"鸭子一直原地转圈"的原因）。


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
F:\anaconda_ydx\python.exe tools\check_space_encoding.py           # 线上 HTML 有没有被注入切坏中文
```

`--assets=external` **不打包机器人网格**（前文所述的许可原因），只打包我们自己的
场景 XML 与 Apache-2.0 的策略 ONNX，产物约 7 MB。

最后那条 `check_space_encoding.py` 专门盯一个 HF 侧的坑（见第 8 节 #11）：
静态空间会往 HTML 里按字节注入一段脚本，切在中文里就会把字劈坏。
它会拿「本地产物 / Hub 存储 / 静态 CDN」三份比字节并数 `U+FFFD`，正常输出长这样：

```
  本地产物     16240 bytes  …  坏字符(U+FFFD)=0
  Hub 存储     16240 bytes  …  坏字符(U+FFFD)=0
  静态 CDN     16341 bytes  …  坏字符(U+FFFD)=0
静态 CDN 注入的脚本：1 处 / 101 字节（这部分是正常的额外内容）
结果: PASS —— 线上没有坏字符，产物是纯 ASCII
```

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
10. **ONNX Runtime 的 `wasmPaths` 必须是绝对 URL**：ORT 拿它按**自己模块**的位置去解析，
    喂一个页面相对的 `./node_modules/...` 会被拼成 `dist/node_modules/...`，动态 import
    直接失败 —— 表现是「8 个策略一个都装不上、页面永远卡在加载」。开发树里现在用
    `new URL(...)` 转绝对地址（发布产物本来就写 CDN 绝对地址，所以只有开发树会踩）。
11. **HF 静态空间会按字节往 HTML 里注入脚本**：它会插一段
    `window.huggingface={variables:{...}}`，切点落在 UTF-8 中文中间时那个字就被劈成两半，
    线上渲染成「模��」。Hub 里存的文件是好的，坏的只有 static CDN 吐出来的那份
    —— 实测线上 4 个 `U+FFFD`，本地与 Hub 都是 0，能直接排除「上传坏了」这条猜测。
    对策已经做进 `build_site.mjs`：产物 `index.html` 里的非 ASCII 全部转成
    `&#NNNN;` 实体 → 纯 ASCII，怎么切都切不坏，浏览器解码后显示与原文一致。
    **发布后跑 `tools/check_space_encoding.py` 确认线上 `U+FFFD=0`。**
12. **VLM 面板的模型名曾经是空 input**：页面加载时走的是「不覆盖已有值」的分支，
    而首次打开（localStorage 为空）时 input 本来就是空的，于是 `model` 是空串，
    请求发出去被服务端回 400「you must provide a model parameter」。
    现在模型是 `<select>`，永远有值。**教训：凡是能空的关键字段，别用空 input 当默认。**
13. **"进了区域却永远不算完成"**：判据常写成「位姿 + `speed_xy_le`（要停住）」，
    而决策层还在继续下发速度指令 → 速度条件永远不成立 → 判据永远不满足 →
    鸭子继续往前走，反而走出区域。对策是 `goal.js` 额外算一个 `positional`
    （除速度外都满足），agent 看到它就主动刹住再等 `success_hold_s`。
    **教训：把"到位"和"停稳"拆开判，别指望模型自己停下来。**
14. **纯动作指令别硬猜目标**：「前进1米，再翻滚一次，最后跳舞」里没有任何物体，
    以前 `scanText` 找不到实体就退化成默认目标 `ball`，于是鸭子满场转圈找球、
    翻滚跳舞一次都没执行。现在 `sequence.js` 先识别成动作序列并接管
    （`intent: "sequence"`、`target: ""`），有目标物体的导航指令才走决策层。
15. **上游技能策略必须实测，别信名字**：
    * `happy_hop`（原本当"跳舞"）实测会把鸭子一点点歪到侧躺（upright 1.00 → 0.45），
      策略结束切回走路后彻底摔平 —— 用户看到的"跳舞变成侧滚翻、后面还站不起来"
      就是这个。它的 `verified` 已改成 `false`（不再进 VLM 词表），动作序列里遇到
      "跳舞"会**跳过并写明原因**，界面按钮也标成「跳一下（会躺倒）」。
    * `alpha_standup`（"起身"）从那个姿态**起不来**：upright 0.59 → -0.13 又躺回去。
    * `roulade`（"翻滚"）反而能稳定地从躺平翻回站立：**3/3 次、约 4~5 s、
      末态 upright 0.95**。所以「起身」按钮改成跑 roulade（`durationS: 6.0`），
      动作序列收尾也会检查站姿、躺下就自动翻回来。
    验证脚本：`web_sim/tools/_probe_dance.mjs`、`_probe_standup.mjs`、`_probe_recover.mjs`。
