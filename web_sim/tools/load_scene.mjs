/**
 * Node 侧加载摊平后的场景 —— 只此一份实现，浏览器那边对应 `DuckSim.load()`。
 *
 * 两个地方必须一致，否则「Node 验证通过、浏览器里跑不起来」：
 *   - mesh 目录从 XML 的 `meshdir` 读（现在指向共享的 `../_shared`）；
 *   - 注册到 MjVFS 的路径必须和 XML 里写的一模一样（MuJoCo 把 VFS 路径当字符串比对）。
 */
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

export const meshdirOf = (xml) => (xml.match(/meshdir="([^"]+)"/) || [, "assets"])[1];

export async function readSceneFiles(sceneDir) {
  const xml = await readFile(path.join(sceneDir, "scene.xml"), "utf8");
  const meshdir = meshdirOf(xml);
  const meshDirAbs = path.resolve(sceneDir, meshdir);
  const names = await readdir(meshDirAbs);
  const meshes = [];
  for (const name of names) {
    meshes.push([`${meshdir}/${name}`, await readFile(path.join(meshDirAbs, name))]);
  }
  return { xml, meshdir, meshes };
}

/** 塞进 VFS 并建模型。 */
export async function loadScene(mujoco, sceneDir, { keyframe = 0 } = {}) {
  const { xml, meshdir, meshes } = await readSceneFiles(sceneDir);
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer("scene.xml", new TextEncoder().encode(xml));
  for (const [name, buf] of meshes) vfs.addBuffer(name, buf);
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);
  if (keyframe >= 0) mujoco.mj_resetDataKeyframe(model, data, keyframe);
  mujoco.mj_forward(model, data);
  return { model, data, xml, meshdir, meshCount: meshes.length };
}
