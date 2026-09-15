/**
 * BYO key：浏览器**直接**连厂商，中间没有我们的服务器。
 *
 * 三个适配器对应 `duck_vlm/vlm.py` 里的同一套请求体，所以同一句提示词、
 * 同一个模型，在 Python 端和网页端得到的行为可以直接对照。
 *
 * 已实测：DashScope 兼容模式对浏览器跨域是放行的
 * （OPTIONS 回 `access-control-allow-origin: <origin>`，允许 POST + authorization/content-type），
 * 所以 Qwen-VL 不需要任何代理。其它厂商各自的要求见下表：
 *
 * | provider  | 说明 |
 * | ---       | ---  |
 * | openai    | 兼容 OpenAI /chat/completions 的一切：Qwen(DashScope)、DeepSeek、Moonshot、
 * |           | 本地 vLLM、Ollama /v1、llama.cpp server |
 * | gemini    | Google AI Studio，key 走 query，图走 inline_data |
 * | anthropic | 必须显式打开浏览器直连头 `anthropic-dangerous-direct-browser-access` |
 */

const DEFAULT_BASE = {
  openai: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  gemini: "https://generativelanguage.googleapis.com/v1beta",
  anthropic: "https://api.anthropic.com/v1",
};

export const PROVIDERS = {
  openai: {
    label: "Qwen / OpenAI 兼容",
    defaultModel: "qwen3-vl-plus",
    defaultBaseUrl: DEFAULT_BASE.openai,
    hint: "DashScope 兼容模式 / OpenAI / DeepSeek / Moonshot / 本地 vLLM、Ollama 都走这个",
  },
  gemini: {
    label: "Gemini",
    defaultModel: "gemini-2.5-flash",
    defaultBaseUrl: DEFAULT_BASE.gemini,
    hint: "Google AI Studio 的 key；模型名可换成任何支持图片的 Gemini",
  },
  anthropic: {
    label: "Claude",
    defaultModel: "claude-sonnet-4-5",
    defaultBaseUrl: DEFAULT_BASE.anthropic,
    hint: "需要浏览器直连头（本适配器已带）",
  },
};

/** 与 Python `_endpoint` 一致：base 后面补 /v1/chat/completions，但别补重复。 */
export function resolveEndpoint(baseUrl) {
  const url = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!url) throw new Error("base URL 是空的");
  if (url.endsWith("/chat/completions")) return url;
  if (url.endsWith("/v1")) return url + "/chat/completions";
  return url + "/v1/chat/completions";
}

export function splitDataUrl(dataUrl) {
  const m = /^data:([^;]+);base64,(.*)$/s.exec(String(dataUrl || ""));
  if (!m) throw new Error("图片必须是 data URL（data:image/jpeg;base64,...）");
  return { mime: m[1], base64: m[2] };
}

async function postJson(url, payload, headers, timeoutMs = 40000) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(payload),
      signal: ac.signal,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 600)}`);
    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`返回的不是 JSON：${text.slice(0, 300)}`);
    }
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`请求超时（${timeoutMs} ms）`);
    if (e instanceof TypeError) {
      throw new Error(`连不上或跨域被拒：${e.message}（检查 base URL、网络、以及该厂商是否允许浏览器直连）`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function openaiText(data) {
  const msg = data?.choices?.[0]?.message;
  if (!msg) throw new Error(`不像 OpenAI 兼容响应：${JSON.stringify(data).slice(0, 400)}`);
  let content = msg.content ?? "";
  if (Array.isArray(content)) {
    content = content.filter((p) => p && p.type === "text").map((p) => p.text || "").join("\n");
  }
  return String(content || "").trim();
}

export const adapters = {
  openai({ prompt, imageDataUrl, model, apiKey, baseUrl, temperature, maxTokens, imageDetail, reasoningEffort, timeoutMs }) {
    const payload = {
      model,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: imageDataUrl, detail: imageDetail || "high" } },
        ],
      }],
      temperature,
      max_tokens: maxTokens,
    };
    if (reasoningEffort) payload.reasoning_effort = reasoningEffort;
    const headers = { "user-agent": "duck-vlm-web/0.1" };
    if (apiKey) headers.authorization = "Bearer " + apiKey;
    return postJson(resolveEndpoint(baseUrl), payload, headers, timeoutMs)
      .then((data) => ({ text: openaiText(data), raw: data }));
  },

  gemini({ prompt, imageDataUrl, model, apiKey, baseUrl, temperature, maxTokens, timeoutMs }) {
    const { mime, base64 } = splitDataUrl(imageDataUrl);
    const base = String(baseUrl || "").replace(/\/+$/, "");
    const url = `${base}/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey || "")}`;
    const payload = {
      contents: [{ role: "user", parts: [{ text: prompt }, { inline_data: { mime_type: mime, data: base64 } }] }],
      generationConfig: { temperature, maxOutputTokens: maxTokens },
    };
    return postJson(url, payload, { "user-agent": "duck-vlm-web/0.1" }, timeoutMs).then((data) => {
      const parts = data?.candidates?.[0]?.content?.parts;
      if (!parts) throw new Error(`不像 Gemini 响应：${JSON.stringify(data).slice(0, 400)}`);
      return { text: parts.filter(Boolean).map((p) => p.text || "").join("\n").trim(), raw: data };
    });
  },

  anthropic({ prompt, imageDataUrl, model, apiKey, baseUrl, temperature, maxTokens, timeoutMs }) {
    const { mime, base64 } = splitDataUrl(imageDataUrl);
    const payload = {
      model,
      max_tokens: maxTokens,
      temperature,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: mime, data: base64 } },
          { type: "text", text: prompt },
        ],
      }],
    };
    const headers = {
      "x-api-key": apiKey || "",
      "anthropic-version": "2023-06-01",
      // 没有这个头，浏览器里必被 CORS 拦掉
      "anthropic-dangerous-direct-browser-access": "true",
    };
    return postJson(String(baseUrl || "").replace(/\/+$/, "") + "/messages", payload, headers, timeoutMs)
      .then((data) => {
        const text = (data?.content || []).filter((p) => p.type === "text").map((p) => p.text || "").join("\n");
        if (!text) throw new Error(`不像 Anthropic 响应：${JSON.stringify(data).slice(0, 400)}`);
        return { text: text.trim(), raw: data };
      });
  },
};

/**
 * 问模型：现在该做哪个动作。
 * @returns {Promise<{text:string, raw:object, latencyMs:number}>}
 */
export async function askVlm(config, { prompt, imageDataUrl }) {
  const provider = config.provider || "openai";
  const adapter = adapters[provider];
  if (!adapter) throw new Error(`未知 provider: ${provider}`);
  const t0 = performance.now();
  const out = await adapter({
    prompt, imageDataUrl,
    model: config.model,
    apiKey: config.apiKey,
    baseUrl: config.baseUrl || PROVIDERS[provider].defaultBaseUrl,
    temperature: config.temperature ?? 0,
    maxTokens: config.maxTokens ?? 48,
    imageDetail: config.imageDetail ?? "high",
    reasoningEffort: config.reasoningEffort ?? "",
    timeoutMs: config.timeoutMs ?? 40000,
  });
  return { ...out, latencyMs: performance.now() - t0 };
}
