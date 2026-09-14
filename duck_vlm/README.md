# DuckVLM / Duck Harness

A Show-Harness-style controller for MicroDuck. A VLM observes the duck head camera
and emits **one discrete action token per step**. `duck_vlm.interpreter` grounds
that token into the existing walking ONNX policy, LocalKick, roulade and happy-hop
policies. The browser can be used by a human, or a GUI Agent, to demonstrate the
same `(image, state, action)` transitions used for LoRA training.

## Two modes

- **Zero-shot**: Gemini, GPT, Qwen-VL or any OpenAI-compatible vision endpoint.
  Full prompt includes task, target, proprioception, recent actions, subgoal,
  affordance and recovery hints.
- **Fine-tuned**: a small local VLM + LoRA served through an OpenAI-compatible
  vLLM endpoint. The prompt contains only `image + task + target + recent actions`;
  the model returns one token.
- **dry-run**: no network/key needed. A deterministic policy drives the loop and
  exercises recording, UI and simulator integration.

## Action vocabulary

`FWD`, `BACK`, `STRAFE_L`, `STRAFE_R`, `TURN_L`, `TURN_R`, `STOP`, `STAND`,
`KICK_L`, `KICK_R`, `ROLL`, `DANCE`, `DONE`.

Motion tokens hold a calibrated velocity command for a short interval, then the
loop takes a fresh image. `KICK_*`, `ROLL`, and `DANCE` are one-shot ONNX skills
managed by `LocalKick`.

## Plugins

Each plugin is backed by one boolean and is inert when disabled:

- `proprioception`: body pose, velocity, upright/fallen, target range/bearing.
- `action_memory`: newest-first recent actions and anti-oscillation warning.
- `subgoal`: task-specific current subtask and completion criterion.
- `affordance`: target visibility/range/bearing plus projected image marker.
- `recovery`: repeated-action and no-progress hints.
- `fall_recovery`: reset after a fall (simulation safety).
- `human_takeover`: a human token preempts the next model decision and is recorded.

## Browser cockpit

After integrating `duck_vlm.host` into `sim_server.py`, open `/vlm`. The page
provides:

- live main/head camera
- task and target fields
- zero-shot / fine-tuned / dry-run mode selector
- plugin switches
- run, pause, step-once, stop and record
- token buttons and keyboard control
- recent action/latency/history telemetry

Keyboard mapping: `W/S` forward/back, `A/D` strafe, `Q/E` turn, `Z/X` kick,
`R` roll, `C` dance, `Space` stop. Human actions are tagged as `source=human`.

## Recording

Sessions are written under `duck_vlm/runs/<timestamp>_<task>/`:

```text
session.json
events.jsonl
transitions.jsonl
frames/00000.jpg
frames/00001.jpg
...
```

Each transition contains the exact prompt, image path, body state, recent actions,
plugin outputs, model metadata and chosen token. Export to LLaMA-Factory format:

```bash
python -m duck_vlm.tools.export_alpaca duck_vlm/runs/<session>
```

The output `llamafactory.json` uses `<image>` in the user message and the action
token as the assistant response.

## Configuration

Environment defaults:

```bash
DUCK_VLM_MODE=zero_shot
DUCK_VLM_PROVIDER=openai
DUCK_VLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions
DUCK_VLM_MODEL=qwen3-vl-plus
DUCK_VLM_API_KEY=...
DUCK_VLM_FT_BASE_URL=http://127.0.0.1:8000/v1
DUCK_VLM_FT_MODEL=duck-vlm-lora
```

The browser config can override non-secret runtime fields. API keys are never
written into recordings or returned by `status()`.

## Relationship to Show-Harness

This package adopts the same core design: a shared semantic vocabulary, an
embodiment-specific interpreter, a planner-free token loop, ablation plugins and
GUMI-style demonstration collection. The duck vocabulary and interpreter are
new; the upstream robot-arm controllers are not reused.