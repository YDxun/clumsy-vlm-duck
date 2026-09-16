/**
 * 极简静态服务器（只用 node 标准库，零额外依赖）。
 *
 * 存在的理由只有一个：浏览器不允许 `file://` 下用 fetch 取 wasm/onnx/xml，
 * 而我们要验证的正是“浏览器里能不能跑”。
 *
 * 用法：
 *   node tools/serve.mjs            # http://127.0.0.1:8787
 *   node tools/serve.mjs 9000       # 换端口
 *
 * 有意不做的事：不做 gzip、不做缓存头、不做目录列表美化。
 * 这是本地验证工具，不是生产服务器（正式发布走静态托管）。
 */
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HOST = "127.0.0.1";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".wasm": "application/wasm",          // 必须是这个，否则 WebAssembly.instantiateStreaming 拒绝
  ".onnx": "application/octet-stream",  // 二进制，别让浏览器猜成文本
  ".stl": "model/stl",
  ".png": "image/png",
  ".svg": "image/svg+xml",   // favicon 用 SVG：Chrome 对 MIME 严格，给错会拒绝渲染
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
};

export function createHandler(root = ROOT) {
  return async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      let rel = decodeURIComponent(url.pathname);
      if (rel.endsWith("/")) rel += "index.html";
      const abs = path.resolve(root, "." + rel);
      // 目录穿越防护：解析后的路径必须还在 web_sim/ 里
      if (!abs.startsWith(root + path.sep) && abs !== root) {
        res.writeHead(403).end("forbidden");
        return;
      }
      const st = await stat(abs);
      if (st.isDirectory()) {
        res.writeHead(302, { location: rel.replace(/\/?$/, "/") + "index.html" }).end();
        return;
      }
      res.writeHead(200, {
        "content-type": MIME[path.extname(abs).toLowerCase()] || "application/octet-stream",
        "content-length": st.size,
        "cache-control": "no-store",
      });
      if (req.method === "HEAD") { res.end(); return; }
      createReadStream(abs).pipe(res);
    } catch (e) {
      res.writeHead(e.code === "ENOENT" ? 404 : 500, { "content-type": "text/plain; charset=utf-8" })
         .end(e.code === "ENOENT" ? "not found" : String(e));
    }
  };
}

/** 起一个服务器，返回 { url, close }。port=0 时由系统分配（验证脚本用，避免端口冲突）。 */
export function startServer({ port = 8787, host = HOST, root = ROOT } = {}) {
  const server = createServer(createHandler(root));
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      const { port: p } = server.address();
      resolve({
        server,
        url: `http://${host}:${p}/`,
        root,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}

// 直接 `node tools/serve.mjs [port]` 时才自启
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  // 用法：node tools/serve.mjs [port] [相对根目录，例如 _site]
  const root = process.argv[3] ? path.resolve(ROOT, process.argv[3]) : ROOT;
  const { url } = await startServer({ port: Number(process.argv[2] || 8787), root });
  console.log(`[serve] -> ${url}`);
  console.log(`[serve] 根目录 ${root}`);
}
