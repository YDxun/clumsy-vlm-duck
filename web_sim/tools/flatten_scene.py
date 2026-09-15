# -*- coding: utf-8 -*-
"""把 duck_scenes 里的一个场景摊平成浏览器可加载的形式。

浏览器端的官方 MuJoCo WASM 通过 `mujoco.MjVFS` 提供虚拟文件系统：
主 XML 放在 VFS 根，mesh 放在 `<meshdir>/` 下，再用
`MjModel.from_xml_string(xml, vfs)` 加载。我们的场景本来是
`scene.xml` + `../../robot/robot_allcollisions.xml` + `robot/assets/*.stl`
这种多层相对路径，浏览器里没有 symlink 也没有目录遍历的便利，
所以这里把它压平：

    web_sim/assets/<scene>/scene.xml      # include 已内联，meshdir 指向 assets
    web_sim/assets/<scene>/assets/*.stl   # 仅复制该场景实际用到的 mesh

用法：  python web_sim/tools/flatten_scene.py duck_workspace_v1
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "duck_scenes" / "scenes"
ROBOT = ROOT / "duck_scenes" / "robot"
OUT_ROOT = ROOT / "web_sim" / "assets"


def inline_includes(xml: str, base: Path, depth: int = 0) -> str:
    """递归把 <include file="..."/> 替换成被包含文件的正文。"""
    if depth > 8:
        raise RuntimeError("include 嵌套过深，疑似循环")

    def repl(m: re.Match[str]) -> str:
        rel = m.group(1)
        target = (base / rel).resolve()
        if not target.exists():
            raise FileNotFoundError(f"include 目标不存在: {rel} -> {target}")
        body = target.read_text(encoding="utf-8-sig")
        # 被包含的文件往往自带 XML 声明，内联到中间会变成非法 XML（声明只能在文件开头）
        body = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", body)
        # MuJoCo 的 <include> 会把被包含文件里 <mujoco> 的**子元素**并入父模型，
        # 而不是把整个文档文本插进来。所以这里要解包它的根元素，
        # 否则摊平后会出现两个 <mujoco>，解析报 Schema violation。
        m = re.search(r"<mujoco\b[^>]*>(.*)</mujoco>", body, re.S)
        if m:
            body = m.group(1)
        return inline_includes(body, target.parent, depth + 1)

    return re.sub(r'<include\s+file="([^"]+)"\s*/>', repl, xml)


def referenced_meshes(xml: str) -> list[str]:
    return sorted(set(re.findall(r'<mesh\s+file="([^"]+)"', xml)))


# 运行时改模型在浏览器里做不到：
# `_dev_cockpit/sim_server.py` 的 LocalSim 会把执行器力矩上限从 XML 的
# ±0.96 覆写成 ±(KT*1.75)=±0.605454（策略就是按这个权限训练的），
# 而官方 MuJoCo WASM 的 `model.actuator_forcerange` / `actuator_forcelimited`
# 在 JS 侧要么写不进去、要么直接抛 Embind 的
# `BindingError: unknown type N10emscripten11memory_viewIbEE`。
# 不补这一步，浏览器里的鸭子力矩比策略预期高 59%，一走就摔（实测 upright 1.0 → 0.30）。
# 所以把这条覆写**烘进摊平后的 XML**：给每个位置执行器写上 forcerange。
KT = 0.3459739511711113
FORCE_LIMIT = KT * 1.75


def bake_force_range(xml: str, limit: float = FORCE_LIMIT) -> tuple[str, int]:
    """给每个 <position> 执行器写入与 Python 端一致的 forcerange。"""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        tag = m.group(0)
        if "forcerange" in tag:
            tag = re.sub(r'\s+forcerange="[^"]*"', "", tag)
        count += 1
        return tag[:-2].rstrip() + f' forcerange="{-limit:.6f} {limit:.6f}"/>'

    out = re.sub(r"<position\b[^>]*/>", repl, xml)
    return out, count


def flatten(scene_id: str, out_root: Path = OUT_ROOT) -> dict:
    scene_dir = SCENES / scene_id
    scene_xml = scene_dir / "scene.xml"
    if not scene_xml.exists():
        raise SystemExit(f"找不到场景: {scene_xml}")

    raw = scene_xml.read_text(encoding="utf-8-sig")
    flat = inline_includes(raw, scene_dir)

    # Qt/XML 头之外，我们在 <mujoco> 之后插入 meshdir，让 mesh 引用解析到 assets/
    if "meshdir" in flat:
        flat = re.sub(r'(<compiler\b[^>]*?)meshdir="[^"]*"', r"\1", flat)
    if re.search(r"<compiler\b[^>]*/>", flat):
        flat = re.sub(r"<compiler\b([^>]*?)/>",
                      lambda m: f'<compiler{m.group(1)} meshdir="assets"/>', flat, count=1)
    else:
        # 没有自闭合 compiler 标签时，直接补一个
        flat = flat.replace("<mujoco", '<mujoco', 1)
        flat = re.sub(r"(<mujoco[^>]*>)", r'\1\n <compiler meshdir="assets"/>', flat, count=1)

    flat, n_act = bake_force_range(flat)
    meshes = referenced_meshes(flat)
    out_dir = out_root / scene_id
    assets_dir = out_dir / "assets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for name in meshes:
        src = ROBOT / "assets" / name
        if src.exists():
            shutil.copy2(src, assets_dir / name)
            copied.append(name)
        else:
            missing.append(name)

    (out_dir / "scene.xml").write_text(flat, encoding="utf-8", newline="\n")
    total = sum((assets_dir / n).stat().st_size for n in copied)
    return {"scene": scene_id, "out": str(out_dir), "meshes": len(copied),
            "missing": missing, "bytes": total, "xml_bytes": len(flat),
            "actuators": n_act, "force_limit": FORCE_LIMIT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="场景 id，例如 duck_workspace_v1")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    info = flatten(args.scene, Path(args.out))
    print(f"场景      : {info['scene']}")
    print(f"输出      : {info['out']}")
    print(f"XML       : {info['xml_bytes']} bytes")
    print(f"mesh      : {info['meshes']} 个, {info['bytes']/1048576:.2f} MB")
    print(f"力矩限幅  : {info['actuators']} 个执行器写入 ±{info['force_limit']:.6f}（与 Python LocalSim 一致）")
    if info["missing"]:
        print(f"缺失 mesh : {info['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
