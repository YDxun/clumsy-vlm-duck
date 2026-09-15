/**
 * 把 MuJoCo 的模型画出来。
 *
 * 官方 @mujoco/mujoco WASM 只给物理与 mjv_updateScene，**没有光栅化器**
 * （没有 mjr_render / mjr_readPixels），所以画面得自己出。做法与 quackd 同源、
 * 但这里是我们自己的实现：
 *
 *   1. 建场景时遍历 `model.geom_*`（静态几何），每个 geom 建一个 three.js mesh；
 *      mesh 的顶点直接来自 `model.mesh_vert / mesh_face` —— MuJoCo 已经解析过 STL，
 *      所以我们不需要第二个模型副本，也不需要 STL 解析器。
 *   2. 每帧只把 `data.geom_xpos / data.geom_xmat` 灌进 mesh 的变换矩阵。
 *   3. MuJoCo 是 z-up、three.js 是 y-up，所以整个 world 组绕 x 轴转 -90°，
 *      上面两处的算术都保持 MuJoCo 的坐标。
 *
 * 与 quackd 的三处**有意不同**（适配我们自己的场景与相机需求）：
 *   - THREE 由外部注入（浏览器可从 CDN 取、Node 验证用 npm 包），本模块不写死 import；
 *   - 修正了 capsule 的朝向（three.js 的 capsule 沿 y，MuJoCo 沿 z，必须与 cylinder 一样转过来）；
 *   - 相机支持四种模式：workspace（场景自带的 3/4 全景 cam_workspace）、overhead（正上方全场
 *     cam_overhead）、follow（第三人称跟随，我们的轨道参数）、duck（鸭子第一人称，逐字复刻
 *     Python 的 render_headcam）。
 *
 * 相机数值的来源很重要：workspace/overhead 两个是**从模型里读** `cam_pos/cam_quat/cam_fovy`，
 * 不是把数字抄进代码——场景一改，视角自动跟着改。
 */

// MuJoCo mjtGeom 的取值
const PLANE = 0, SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, CYLINDER = 5, BOX = 6, MESH = 7;

/** MuJoCo 世界坐标 (x,y,z) -> three.js (x,z,-y)；方向向量同理（平移分量不参与）。 */
const toThree = (v) => [v[0], v[2], -v[1]];

export class DuckView {
  /**
   * @param {object} opts
   * @param {HTMLCanvasElement} opts.canvas
   * @param {object} opts.THREE      three.js 命名空间（注入，避免写死 CDN 或 npm 路径）
   * @param {object} opts.duck       DuckSim 实例（需要 .model / .data / .pose / .headPose()）
   */
  constructor({ canvas, THREE, duck }) {
    this.THREE = THREE;
    this.canvas = canvas;
    this.duck = duck;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x0b1118, 1);

    this.scene = new THREE.Scene();
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;   // MuJoCo z-up -> three.js y-up
    this.scene.add(this.world);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x555f6b, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(2, 4, 3);
    this.scene.add(sun);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.02, 80);
    this.mode = "follow";
    // follow / overhead 共用的一组轨道参数
    this.orbit = { azimuth: 2.3, elevation: 0.42, distance: 1.4 };
    // 场景里的固定相机缓存（按需从模型里读）
    this._cams = {};

    this.meshes = [];
    this.build();
  }

  build() {
    const { model } = this.duck;
    for (let g = 0; g < model.ngeom; g++) {
      const type = model.geom_type[g];
      const size = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];
      const geometry = this.geometryFor(type, size, model.geom_dataid[g]);
      if (!geometry) continue;
      const rgba = [0, 1, 2, 3].map((i) => model.geom_rgba[g * 4 + i]);
      const material = new this.THREE.MeshLambertMaterial({
        color: new this.THREE.Color(rgba[0], rgba[1], rgba[2]),
        transparent: rgba[3] < 1,
        opacity: rgba[3],
      });
      const mesh = new this.THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      this.world.add(mesh);
      this.meshes.push({ index: g, mesh });
    }
  }

  geometryFor(type, size, dataid) {
    const T = this.THREE;
    switch (type) {
      case PLANE: {
        const hx = size[0] > 0 ? size[0] : 4;
        const hy = size[1] > 0 ? size[1] : hx;
        return new T.PlaneGeometry(hx * 2, hy * 2);
      }
      case SPHERE:
        return new T.SphereGeometry(size[0], 20, 14);
      case ELLIPSOID: {
        const s = new T.SphereGeometry(1, 20, 14);
        s.scale(size[0], size[1], size[2]);
        return s;
      }
      case CAPSULE: {
        // three.js 的 capsule 沿 y 轴；MuJoCo 沿 z 轴。必须转过来，
        // 否则胶囊会躺在地上（quackd 把这条标成了已知问题）。
        const c = new T.CapsuleGeometry(size[0], size[1] * 2, 6, 12);
        c.rotateX(Math.PI / 2);
        return c;
      }
      case CYLINDER: {
        const c = new T.CylinderGeometry(size[0], size[0], size[1] * 2, 24);
        c.rotateX(Math.PI / 2);
        return c;
      }
      case BOX:
        return new T.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
      case MESH:
        return this.meshGeometry(dataid);
      default:
        return null;
    }
  }

  meshGeometry(id) {
    const { model } = this.duck;
    if (id < 0) return null;
    const vAt = model.mesh_vertadr[id] * 3, vN = model.mesh_vertnum[id];
    const fAt = model.mesh_faceadr[id] * 3, fN = model.mesh_facenum[id];
    const vertices = new Float32Array(vN * 3);
    for (let i = 0; i < vN * 3; i++) vertices[i] = model.mesh_vert[vAt + i];
    const indices = new Uint32Array(fN * 3);
    for (let i = 0; i < fN * 3; i++) indices[i] = model.mesh_face[fAt + i];
    const g = new this.THREE.BufferGeometry();
    g.setAttribute("position", new this.THREE.BufferAttribute(vertices, 3));
    g.setIndex(new this.THREE.BufferAttribute(indices, 1));
    g.computeVertexNormals();
    return g;
  }

  setMode(mode) { this.mode = mode; }

  /** 把一组 MuJoCo 相机参数（世界坐标）装到 three.js 相机上。 */
  applyCamera({ pos, dir, up, fovy }) {
    const T = this.THREE;
    if (this.camera.fov !== fovy) { this.camera.fov = fovy; this.camera.updateProjectionMatrix(); }
    const p = toThree(pos), d = toThree(dir);
    this.camera.position.set(p[0], p[1], p[2]);
    if (up) { const u = toThree(up); this.camera.up.set(u[0], u[1], u[2]); }
    else this.camera.up.set(0, 1, 0);
    this.camera.lookAt(p[0] + d[0], p[1] + d[1], p[2] + d[2]);
  }

  /** 场景自带的固定相机（名字与 MuJoCo 里一致）。 */
  fixedCamera(name) {
    if (!this._cams[name]) this._cams[name] = this.duck.fixedCamera(name);
    return this._cams[name];
  }

  /** 每帧：把 MuJoCo 的 geom 位姿搬进 three.js 矩阵，然后渲染。 */
  frame() {
    const { data } = this.duck;
    for (const { index, mesh } of this.meshes) {
      const p = index * 3, m = index * 9;
      mesh.matrix.set(
        data.geom_xmat[m],     data.geom_xmat[m + 1], data.geom_xmat[m + 2], data.geom_xpos[p],
        data.geom_xmat[m + 3], data.geom_xmat[m + 4], data.geom_xmat[m + 5], data.geom_xpos[p + 1],
        data.geom_xmat[m + 6], data.geom_xmat[m + 7], data.geom_xmat[m + 8], data.geom_xpos[p + 2],
        0, 0, 0, 1,
      );
    }
    this.placeCamera();
    const w = this.canvas.clientWidth || this.canvas.width;
    const h = this.canvas.clientHeight || this.canvas.height;
    const ratio = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.floor(w * ratio) || this.canvas.height !== Math.floor(h * ratio)) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.render(this.scene, this.camera);
  }

  placeCamera() {
    const T = this.THREE;
    if (this.mode === "duck") {
      // 头摄：MuJoCo 的自由相机在朝向已知时会用“最接近世界上方”的 up，
      // 这里照抄同一规则，保证画面不会绕视线轴旋转。
      const { pos, dir, fovy } = this.duck.headCamPose();
      const up = this.worldUpFor(dir);
      this.applyCamera({ pos, dir, up, fovy });
      return;
    }
    if (this.mode === "overhead" || this.mode === "workspace") {
      const cam = this.fixedCamera(`cam_${this.mode}`);
      if (cam) { this.applyCamera({ pos: cam.pos, dir: cam.forward, up: cam.up, fovy: cam.fovy }); return; }
      // 场景里没有这个相机时才退化到内置参数
      if (this.mode === "overhead") {
        this.applyCamera({ pos: [0, 0, 5.2], dir: [0, 0, -1], up: [0, 1, 0], fovy: 50 });
        return;
      }
    }
    if (this.camera.fov !== 45) { this.camera.fov = 45; this.camera.updateProjectionMatrix(); }
    // follow：第三人称跟随鸭子
    const pose = this.duck.pose;
    const { azimuth, elevation, distance } = this.orbit;
    const target = new T.Vector3(pose.x, 0.12, -pose.y);
    this.camera.position.set(
      target.x + distance * Math.cos(elevation) * Math.cos(azimuth),
      target.y + distance * Math.sin(elevation),
      target.z + distance * Math.cos(elevation) * Math.sin(azimuth),
    );
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(target);
  }

  /** 与 MuJoCo 自由相机一致的 up 向量：世界上方在视线垂直方向上的分量。 */
  worldUpFor(dir) {
    const z = [0, 0, 1];
    const d = z[0] * dir[0] + z[1] * dir[1] + z[2] * dir[2];
    let up = [z[0] - d * dir[0], z[1] - d * dir[1], z[2] - d * dir[2]];
    const n = Math.hypot(up[0], up[1], up[2]);
    if (n < 1e-6) up = [1, 0, 0];      // 视线与世界上方平行时的退化处理
    else up = up.map((v) => v / n);
    return up;
  }

  dispose() {
    for (const { mesh } of this.meshes) { mesh.geometry.dispose(); mesh.material.dispose(); }
    this.renderer.dispose();
  }
}
