# DuckVLM LoRA training and serving

## 1. Collect demonstrations

Use `/vlm` in dry-run or zero-shot mode. Press **START**, then use the action
buttons or keyboard for human takeover. Every accepted action is recorded under
`duck_vlm/runs/<session>/`.

## 2. Export LLaMA-Factory data

```bash
python duck_vlm/train/prepare_dataset.py duck_vlm/runs/session-a duck_vlm/runs/session-b
```

This writes `duck_vlm/train/data/duck_vlm_sft.json` and `dataset_info.json`.
Each sample has the exact fine-tuned prompt (`image + task + recent actions`) and
one assistant token.

## 3. Train Qwen3.5-2B LoRA

Install LLaMA-Factory in a separate environment first, then:

```bash
CONFIG=duck_vlm/train/configs/qwen3_5_2b_lora.yaml bash duck_vlm/train/scripts/train_lora.sh
```

The config freezes the vision tower, trains language-side LoRA adapters, and uses
the `qwen3_5_nothink` chat template. Adjust `output_dir`, batch size, and epochs
in the YAML if needed.

## 4. Serve through vLLM

```bash
MODEL=Qwen/Qwen3.5-2B ADAPTER=/absolute/path/to/adapter PORT=8000 bash duck_vlm/train/scripts/serve_vllm.sh
```

The service exposes an OpenAI-compatible API at `http://127.0.0.1:8000/v1` with
model name `duck-vlm-lora`. In the browser select **FINE-TUNED LORA**; the
default `DUCK_VLM_FT_MODEL` is `duck-vlm-lora`.

## Contracts

Do not change these independently after collecting data:

- prompt version and wording,
- token names and order,
- recent-action window,
- image size/detail policy,
- action duration/interpreter behavior.

The runtime logs the exact prompt and image path for every transition so these
contracts can be checked before training and after serving.