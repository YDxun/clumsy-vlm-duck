"""Score target/intent inference on the free-form prompt set.

Run:  python -m duck_vlm.eval_nl [path/to/set.json]

This measures only what can be judged offline and objectively - which body the sensor
would track, and which coarse intent the plugins would act on. Whether the duck then
*succeeds* on an unmatched sentence still needs visual review, because an unmatched
prompt has no truth-based evaluator; the report calls that out per item.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .intent import pick_target, resolve_intent
from .scenes import SceneCatalog

DEFAULT_SET = (Path(__file__).resolve().parent.parent / "research"
               / "nl_zeroshot_set_20260915.json")


def evaluate(path: Path | None = None) -> dict[str, Any]:
    doc = json.loads(Path(path or DEFAULT_SET).read_text(encoding="utf-8"))
    catalog = SceneCatalog()
    scenes: dict[str, Any] = {}
    rows = []
    for item in doc["items"]:
        sid = item["scene"]
        if sid not in scenes:
            scenes[sid] = catalog.load(sid)
        scene = scenes[sid]
        text = item["text"]
        body, matched = pick_target(scene.scan_text(text), text)
        intent = resolve_intent("", text)
        try:
            task = scene.match_task(text)
            if task is None:
                # mirror what the running system does for free-form prompts
                task = scene.match_task_semantic(text, body)
        except Exception:
            task = None
        rows.append({
            "id": item["id"], "text": text, "supported": bool(item.get("supported", True)),
            "want_target": item["target"], "got_target": body, "matched_label": matched,
            "want_intent": item["intent"], "got_intent": intent,
            "target_ok": body == item["target"], "intent_ok": intent == item["intent"],
            "scored": bool(task), "task_id": (task or {}).get("id"),
        })
    return {"name": doc.get("name"), "rows": rows}


def report(res: dict[str, Any]) -> int:
    rows = res["rows"]
    n = len(rows)
    tgt = sum(1 for r in rows if r["target_ok"])
    itn = sum(1 for r in rows if r["intent_ok"])
    scored = sum(1 for r in rows if r["scored"])
    sup = [r for r in rows if r["supported"]]
    unsup = [r for r in rows if not r["supported"]]
    print("%-6s %-44s %-20s %-20s %-6s %-4s" % ("id", "prompt", "target", "intent", "score", "sup"))
    print("-" * 108)
    for r in rows:
        flag_t = "" if r["target_ok"] else " <-- target"
        flag_i = "" if r["intent_ok"] else " <-- intent"
        print("%-6s %-44s %-20s %-20s %-6s %-4s" % (
            r["id"], r["text"][:42], r["got_target"] + flag_t,
            r["got_intent"] + flag_i, "yes" if r["scored"] else "no",
            "" if r["supported"] else "no"))
    print()
    print("target inference : %d/%d = %.0f%%" % (tgt, n, 100.0 * tgt / n))
    print("intent inference : %d/%d = %.0f%%" % (itn, n, 100.0 * itn / n))
    print("truth-scored     : %d/%d = %.0f%%  (the rest need visual review)" % (
        scored, n, 100.0 * scored / n))
    if sup:
        print("  servable prompts (%d): target %d/%d, scored %d/%d" % (
            len(sup), sum(1 for r in sup if r["target_ok"]), len(sup),
            sum(1 for r in sup if r["scored"]), len(sup)))
    if unsup:
        print("  unservable prompts (%d): target %d/%d" % (
            len(unsup), sum(1 for r in unsup if r["target_ok"]), len(unsup)))
    return 0 if tgt == n and itn == n else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return report(evaluate(Path(args[0]) if args else None))


if __name__ == "__main__":
    raise SystemExit(main())
