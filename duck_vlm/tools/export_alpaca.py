#!/usr/bin/env python3
"""Export one GUMI/VLM recording session to LLaMA-Factory format."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duck_vlm.recorder import export_alpaca


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="duck_vlm/runs/<session> directory")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()
    out = export_alpaca(args.session, args.output)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())