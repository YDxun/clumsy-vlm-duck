# duck_home_v1

两室一厅的 P1 家庭场景：地板 6 m x 5 m，中间隔墙在 x = 0 处，门洞净宽约 1.40~1.50 m
（实测 1.013 m 的碰撞自由宽度），沿 x 方向通行。家具为贴地矮件：沙发高 0.10 m、
茶几高 0.08 m。

**本场景包不含人体仿真。** 早期版本里放过 `person_owner` / `person_stranger` 两个静止胶囊，
用于 `come_to_owner` / `follow_owner` / `orbit_stranger` 三个人物任务；
现已整体移除（含 metadata 的 `people` 段与上述三个任务），
`validation_core.py` 的 `HOME_LAYOUT` 与 `SUCCESS_EVALUATOR` 也同步去掉了对人的依赖。
后续项目不再做人形仿真。

## 物体与任务

- 小球 `ball`：默认在 `(2.25, -1.70, 0.035)`，位于东侧房间（跨房间搜索的目标）。
- 信标 `beacon_target`：`(2.25, 2.15, 0.20)`，在沙发附近。
- 绿色区域 `zone_green`：`(-1.70, -1.55)`，半径 0.35 m，用于"把球推进区域"。

保留的 3 个任务：

| 任务 | 成功条件 |
| --- | --- |
| `search_ball_across_rooms` | 穿过门洞找到球，本体距球 <= 0.4 m |
| `go_to_beacon` | 本体距信标 <= 0.35 m |
| `retrieve_ball` | 球进入 `zone_green` 半径 0.30 m 内 |

## 验证

复跑 `validate.py`（需要 `duck_scenes/policies` 下的 ONNX 策略，否则运动类检查会跳过）：

```bash
cd duck_scenes/scenes/duck_home_v1 && python3 validate.py
```

结果写入 `validation.json`，预览图写入 `previews/`。
