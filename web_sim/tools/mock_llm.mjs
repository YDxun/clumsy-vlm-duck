/**
 * 假的 OpenAI 兼容端点，只为了验证“浏览器直连厂商”这条链路本身。
 *
 * 它能证明的事：请求体结构对不对、图片有没有以 data URL 发出去、跨域头是否齐全、
 * 返回的 token 会不会被真的执行。它不能证明的事：真实厂商的限流/鉴权/模型行为。
 * 换真 key 只需要把 baseUrl 指回厂商，代码一行都不用改。
 */
import { createServer } from "node:http";

/**
 * @param {object} opts
 * @param {string|string[]|((body:object, n:number)=>string)} opts.reply 回复内容（可以是脚本序列）
 * @param {number} opts.delayMs 模拟推理延迟
 */
export function startMockLlm({ reply = "STOP", delayMs = 0 } = {}) {
  const requests = [];
  const server = createServer((req, res) => {
    const cors = {
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "POST, OPTIONS",
      "access-control-allow-headers": "content-type, authorization, user-agent, x-api-key, anthropic-version",
      "access-control-max-age": "600",
    };
    if (req.method === "OPTIONS") { res.writeHead(204, cors).end(); return; }
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      const n = requests.length;
      let parsed = null;
      try { parsed = JSON.parse(body || "{}"); } catch { /* 留着原文，测试里会看到 */ }
      requests.push({ url: req.url, headers: req.headers, body: parsed, raw: body });
      let text = reply;
      if (Array.isArray(reply)) text = reply[Math.min(n, reply.length - 1)];
      else if (typeof reply === "function") text = reply(parsed, n);
      const send = () => {
        res.writeHead(200, { "content-type": "application/json", ...cors });
        res.end(JSON.stringify({
          id: "mock-" + n, object: "chat.completion", model: parsed?.model || "mock",
          choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        }));
      };
      if (delayMs) setTimeout(send, delayMs); else send();
    });
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}`,
        requests,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}

// 直接跑：node tools/mock_llm.mjs [reply]
if (process.argv[1] && process.argv[1].endsWith("mock_llm.mjs")) {
  const mock = await startMockLlm({ reply: process.argv[2] || "FWD" });
  console.log(`[mock-llm] ${mock.url}/v1/chat/completions  回复固定为 ${process.argv[2] || "FWD"}`);
}
