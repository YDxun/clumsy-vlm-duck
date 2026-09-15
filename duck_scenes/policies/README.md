# Policies

这些 `.onnx` 是鸭子运行时热插拔的运动/技能策略，**不放进本仓库**。它们托管在独立仓库：

> `XenderYang/microduck-cloud-policies`

全部策略共用同一套观测契约：**`[1,61] -> [1,14]`**
（61 维 obs = 3 陀螺 + 3 重力方向 + 14 关节位置偏差 + 14 关节速度 + 14 上一步动作
 + 3 速度指令 + 10 零填充）。运行时按需热切换，任何策略都能在任意时刻接管。

## 清单与校验

| 文件 | SHA256 |
| --- | --- |
| `alpha_stand.onnx` | `1569268713E40DEEA795DD2922DBA50D3621E15A872855408B6B1B125B1C094B` |
| `alpha_walking.onnx` | `E36332D383997D51401897734CD3E79CF5038406FEDDB18B4D57ECFB141DAA6C` |
| `ball_kick_left.onnx` | `D6928284DCCD3DD61E08BF2F760EFFA74309FBEFD97B2B31AFB2A60F526D196A` |
| `ball_kick_right.onnx` | `147A32C388C6B19111B3AC3B550A9A6DC8B8BF267118AF4D8C3712522EEDB5AF` |
| `alpha_sitstand.onnx` | `C6C40E35E726EABD803D633E090D112994F469921152448367953FBAF9799BC8` |
| `alpha_standup.onnx` | `A28F1C1B56D13570C5DBE8DCA8AC90D397F89D130D65A6B89DE4D937BF8951D1` |

`alpha_sitstand` 支撑 `SIT` 动作；`alpha_standup` 支撑 `STAND_UP`（从倒地起身）。

## `alpha_standup.onnx` 的来历（自行复现）

由 `microduck_rl` 的 `Mjlab-StandUp-Flat-MicroDuck` 任务训练并导出：

```bash
cd microduck_rl
export WANDB_MODE=disabled
# 4096 envs / 3000 轮；RTX 5090 上约 1 小时（1.26 s/轮）
.venv/bin/python -m mjlab_microduck.train_cli Mjlab-StandUp-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 3000 --agent.run-name standup_duckvlm
# 训练结束会自动在 logs/rsl_rl/microduck_stand/<run>/<run>.onnx 写出 ONNX
cp logs/rsl_rl/microduck_stand/<run>/<run>.onnx ../microduck/policies/alpha_standup.onnx
```

该任务的 `ground_state_mix` 课程会让鸭子从 prone / face-up / face-down 状态起步，
所以它练的就是"倒地 → 起身"，不是单纯的坐站转换（那是 `SitStand` 任务）。

实测（经服务真实代码路径 `LocalKick.request('standup')` → 该 ONNX → MuJoCo）：

| 初始姿态 | 结果 |
| --- | --- |
| 正常站立（对照） | upright 保持 1.000 |
| 侧向倒地（upright 0.047） | → 0.86（起身中）→ **0.924**（交给 `alpha_stand` 稳住后） |
| 面朝下倒地（upright -0.019） | → 0.46（起身段）→ **0.983**（稳相后） |

`recover_after_fall` 要求 `upright >= 0.75`，两种倒地姿态最终都满足。
