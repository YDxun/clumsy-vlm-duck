# MicroDuck Duck Together — Cloud Sim + RL + duck_play

RoboGo 云端（RTX 5090）上的 MicroDuck 仿真与强化学习项目：走路基线训练、追球闭环、L1 认人（召唤/跟随/陌生人绕圈）与 M1 组合叙事（找到主人→把球带给他）。

## 仓库内容
| Path | 说明 |
|---|---|
| `duck_play/` | 追球/认人闭环源码（感知→状态机→控制→评测→录制），复用 cloud sim 的 LocalSim/PolicyBank/LocalKick（与网页驾驶舱同一套物理/策略/观测契约） |
| `duck_vlm/` | Show-Harness 风格 VLM 离散动作闭环：Zero-shot、LoRA 微调、可消融插件、GUMI 数据采集、LLaMA-Factory/vLLM 工具链 |
| `research/` | 调研笔记、技术方案、训练运行记录、验收清单、Demo 视频与截图 |
| `research/demo_videos/` | M1 演示视频（多轮追球、组合叙事、L1 认人） |

## DuckVLM Harness

浏览器打开 `http://<host>:8080/vlm`。它使用鸭子的头摄图像、任务、目标、身体状态和最近动作，让 VLM 每一步输出一个动作符号，再由现有 walking ONNX、BallKick、roulade、happy-hop 策略解释执行。支持 `dry_run`、`zero_shot` 和 `finetuned` 三种模式，并逐项记录 `(image, state, action)`。

```bash
# 本地核心测试
python -m unittest discover -s duck_vlm/tests -v

# 导出一段 GUMI/VLM 录制为 LLaMA-Factory 数据
python duck_vlm/train/prepare_dataset.py duck_vlm/runs/<session>

# Qwen3.5-2B LoRA + vLLM 说明
cat duck_vlm/train/README.md
```
## 快速运行（在 RoboGo kickfix 开发机内）
```bash
cd /root/microduck_sim/duck_play
python3 launch/min_chase.py --time 60 --max-touches 3 --arena --auto-restart   # 多轮连续追球
python3 launch/person_demo.py --time 34                                        # L1 认人
python3 launch/composite_demo.py --time 40                                     # 找到主人->带球给他
# 录像：export DUCKPLAY_RECORD=/path/out.mp4（可调 DUCKPLAY_REC_EVERY/_W/_H）
```

## 模型
- 自己训练的走路基线：https://huggingface.co/XenderYang/microduck-rl-ydx-walk
- 当前 Demo 使用的官方策略（cloud）：https://huggingface.co/XenderYang/microduck-cloud-policies

## 致谢 / 上游
- pollen-robotics/microduck & microduck_rl（Apache-2.0）
- 地瓜 RoboGo / Duck Together 社区资料
