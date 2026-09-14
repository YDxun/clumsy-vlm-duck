#!/usr/bin/env python3
"""Merge exported DuckVLM sessions into one LLaMA-Factory SFT dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duck_vlm.recorder import export_alpaca


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", help="session directories")
    parser.add_argument("--output", default="duck_vlm/train/data/duck_vlm_sft.json")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = []
    for session in args.sessions:
        session = Path(session)
        export = export_alpaca(session, session / "llamafactory.json")
        merged.extend(json.loads(export.read_text(encoding="utf-8")))
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {"duck_vlm_sft": {"file_name": output.name, "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"}, "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}}
    (output.parent / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(merged)} samples -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())