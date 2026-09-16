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
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "duck_scenes" / "scenes"
ROBOT = ROOT / "duck_scenes" / "robot"
MANIFEST = Path(__file__).resolve().parents[1] / "scenes.json"
# 注意这是**两个不同的东西**，之前把它们混成一个常量，于是把网格写到了错误的目录
# （out_root/../_shared = web_sim/_shared，而 XML 解析出来的是 out_root/_shared）：
SHARED_MESH_DIR_XML = "../_shared"     # 写进 XML 的 meshdir，相对**各场景目录**解析
SHARED_MESH_DIR_DISK = "_shared"       # 在磁盘上真正的位置：<out_root>/_shared
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


def export_sidecar(scene_dir: Path, out_dir: Path) -> dict:
    """把场景包的 metadata.yaml / tasks.yaml 转成 JSON 一起发到浏览器。

    浏览器里不想塞一个 YAML 解析器，而这两份文件正是决策层要用的：
      - metadata.yaml: objects/zones/beacon 的 body 名 + semantic_labels
        → “去红方块” → obj_cube_red、“绿色区域” → zone_green 的解析表
      - tasks.yaml: 现成的 task id，给页面做任务下拉（避免“输入对不上就没评分”）
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 环境问题，直接说清楚
        raise SystemExit(f"需要 PyYAML 来转换场景元数据: {exc}")
    written = {}
    for name in ("metadata", "tasks"):
        src = scene_dir / f"{name}.yaml"
        if not src.exists():
            continue
        data = yaml.safe_load(src.read_text(encoding="utf-8"))
        dst = out_dir / f"{name}.json"
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        written[name] = dst.name
    return written


def update_manifest(scene_id: str, info: dict) -> dict:
    """把摊平结果登记到 web_sim/scenes.json —— 这就是“加场景 = 跑一次摊平”的那一环。

    清单是**页面唯一的场景来源**：新增场景只要摊平一次，页面下拉里就会自动出现。
    显示用的标题/描述/标签由人手写（清单里已有的字段会被保留），
    而 mesh 体积、任务条数这些是每次摊平后重新测量的。
    """
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    else:
        manifest = {"scenes": []}
    scenes = manifest.setdefault("scenes", [])
    entry = next((s for s in scenes if s.get("id") == scene_id), None)
    if entry is None:
        entry = {
            "id": scene_id,
            "title": scene_id,
            "description": "",
            "tags": [],
        }
        scenes.append(entry)
    entry.update({
        "dir": scene_id,
        "meshCount": info.get("meshes", 0),
        "meshBytes": info.get("bytes", 0),
        "xmlBytes": info.get("xml_bytes", 0),
        "sidecars": sorted((info.get("sidecars") or {}).keys()),
    })
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    return entry


#: 跨引擎一致性检查用的固定协议。两边都必须用同一套，否则比的是协议不是引擎。
REFERENCE_PROTOCOL = "keyframe0 + qvel=0 + ctrl=HOME + 200*mj_step"


def write_reference(scene_id: str, out_dir: Path) -> dict:
    """用**磁盘上的 Python MuJoCo** 跑一遍固定协议，把结果存成 reference.json。

    浏览器里的 WASM 只能跟"另一个引擎"比，所以参考值必须在摊平的时候由 Python 生成，
    而不是写死在验证脚本里 —— 那样每加一个场景都要手改脚本，就不叫可扩展了。

    记录三样东西：
      - 每个 body 的 xpos（21 个 body × 3，很小）
      - trunk 的完整位姿（xmat 9 个数，能抓到旋转差异）
      - 所有 geom 的 xpos 哈希（一条就覆盖整车，漏不掉）
    """
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        return {"error": f"没有 mujoco：{exc}"}
    model = mujoco.MjModel.from_xml_path(str(out_dir / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    ctrl = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491,
                     0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530])
    data.ctrl[:] = ctrl[: model.nu]
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    trunk = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"))
    # 新版 MuJoCo 的 data.xpos[id] 是列向量，numpy 2 下 float() 会报错，统一 ravel
    flat = lambda arr: [round(float(v), 6) for v in np.asarray(arr).ravel()]
    ref = {
        "protocol": REFERENCE_PROTOCOL,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "trunk": flat(data.xpos[trunk]),
        "trunkMat": flat(data.xmat[trunk]),
        "bodies": {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b):
                   flat(data.xpos[b])
                   for b in range(model.nbody)},
        "geomXposHash": _hash_floats(np.asarray(data.geom_xpos).ravel()),
    }
    (out_dir / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    return ref


def _hash_floats(values, ndigits: int = 6) -> str:
    """把一串浮点数量化成字符串后取哈希 —— 用来一次覆盖所有 geom 的位置。"""
    import hashlib
    flat = []
    for v in values:
        if hasattr(v, "__len__"):        # 新版 MuJoCo 的数组是二维的，先摊平
            flat.extend(float(x) for x in v)
        else:
            flat.append(float(v))
    payload = ",".join(f"{v:.{ndigits}f}" for v in flat)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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

    # 三个场景用的是同一批机器人网格，各拷一份是 3×20 MB 的浪费。
    # 统一放到 out_root/_shared，场景 XML 里的 meshdir 指过去。
    flat, n_act = bake_force_range(flat)
    flat = re.sub(r'(<compiler\b[^>]*?)meshdir="[^"]*"', r"\1", flat)
    if re.search(r"<compiler\b[^>]*/>", flat):
        flat = re.sub(r"<compiler\b([^>]*?)/>",
                      lambda m: f'<compiler{m.group(1)} meshdir="{SHARED_MESH_DIR_XML}"/>', flat, count=1)
    else:
        flat = re.sub(r"(<mujoco[^>]*>)", rf'\1\n <compiler meshdir="{SHARED_MESH_DIR_XML}"/>', flat, count=1)
    meshes = referenced_meshes(flat)
    out_dir = out_root / scene_id
    assets_dir = out_root / SHARED_MESH_DIR_DISK          # <out_root>/_shared，与 XML 解析结果一致
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)      # 网格搬去共享目录后，这里得自己建
    assets_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for name in meshes:
        dst = assets_dir / name
        if dst.exists():          # 已经拷过（所有场景共用同一批网格）
            copied.append(name)
            continue
        src = ROBOT / "assets" / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(name)
        else:
            missing.append(name)

    (out_dir / "scene.xml").write_text(flat, encoding="utf-8", newline="\n")
    sidecars = export_sidecar(scene_dir, out_dir)
    total = sum((assets_dir / n).stat().st_size for n in copied if (assets_dir / n).exists())
    info = {"scene": scene_id, "out": str(out_dir), "meshes": len(copied),
            "missing": missing, "bytes": total, "xml_bytes": len(flat),
            "actuators": n_act, "force_limit": FORCE_LIMIT, "sidecars": sidecars}
    ref = write_reference(scene_id, out_dir)
    info["reference"] = "ok" if "error" not in ref else ref["error"]
    update_manifest(scene_id, info)
    return info


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
    # 打印网格实际落盘位置，并和 XML 里 meshdir 的解析结果对一下 ——
    # 这两者曾经不一致（脚本写到 <out>/../_shared，XML 却解析到 <out>/_shared），
    # 靠"旧文件还在"掩盖了很久，从零摊平会直接坏掉。
    meshd = Path(info["out"]) / ".." / "_shared"
    print(f"网格目录  : {Path(info['out']).parent / '_shared'}")
    print(f"XML 解析的: {meshd.resolve()}")
    print(f"力矩限幅  : {info['actuators']} 个执行器写入 ±{info['force_limit']:.6f}（与 Python LocalSim 一致）")
    print(f"场景清单  : {MANIFEST.relative_to(MANIFEST.parents[1])} 已更新")
    print(f"参考数据  : {info.get('reference')}")
    if info["missing"]:
        print(f"缺失 mesh : {info['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
