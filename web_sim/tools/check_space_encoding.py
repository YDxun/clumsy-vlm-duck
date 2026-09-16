"""检查线上静态空间吐出来的 index.html 有没有被改坏（中文被劈成乱码字）。

背景（详见 docs/REPRODUCE.md 踩坑 #11）：
Hugging Face 的静态空间在服务 index.html 时，会往文件里注入一段
`window.huggingface={variables:{...}}`。注入是在**字节层面**切的，切点落在
UTF-8 中文字符中间时，那个字就被劈成两半，浏览器渲染成「模��」。
Hub 里存的文件是好的，坏的只有 static CDN 服务出来的那一份。

本脚本同时取三个副本比字节，出问题时能直接看出是哪一层坏的：
  本地产物  <--  build_site 的输出，必须是纯 ASCII（非 ASCII 已转成 &#NNNN; 实体）
  Hub 存储  <--  resolve/main，应当与本地逐字节一致
  静态 CDN  <--  公开访问的那份，应当只是「多出来的注入脚本」，U+FFFD 必须为 0

用法：
    F:\\anaconda_ydx\\python.exe tools\\check_space_encoding.py
    F:\\anaconda_ydx\\python.exe tools\\check_space_encoding.py --url https://....hf.space/index.html
"""
import argparse
import hashlib
import pathlib
import re
import sys
import urllib.request

REPLACEMENT = b"\xef\xbf\xbd"          # U+FFFD，字符被劈坏后浏览器显示的那个「�」
INJECT_MARK = b"<script>window.huggingface="
DEFAULT_URL = "https://xenderyang-duck-vlm-simulator.static.hf.space/index.html"
DEFAULT_HUB = ("https://huggingface.co/spaces/XenderYang/duck-vlm-simulator"
               "/resolve/main/index.html")


def fetch(url):
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    return urllib.request.urlopen(req, timeout=90).read()


def describe(tag, data):
    bad = data.count(REPLACEMENT)
    print(f"  {tag:10} {len(data):>7} bytes  "
          f"sha256={hashlib.sha256(data).hexdigest()[:16]}  坏字符(U+FFFD)={bad}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site", help="本地产物目录")
    ap.add_argument("--url", default=DEFAULT_URL, help="公开访问的 index.html")
    ap.add_argument("--hub", default=DEFAULT_HUB, help="Hub 上存储的 index.html")
    args = ap.parse_args()

    local_path = pathlib.Path(args.out) / "index.html"
    if not local_path.exists():
        print(f"找不到本地产物 {local_path} —— 先跑 tools/build_site.mjs")
        return 1

    print("三份 index.html 对比：")
    local = local_path.read_bytes()
    describe("本地产物", local)
    describe("Hub 存储", fetch(args.hub))
    live = fetch(args.url)
    live_bad = describe("静态 CDN", live)

    non_ascii = [b for b in local if b > 0x7F]
    print(f"\n本地产物是纯 ASCII：{not non_ascii}"
          + ("" if not non_ascii else f"（还剩 {len(non_ascii)} 个非 ASCII 字节）"))

    injected = re.findall(re.escape(INJECT_MARK) + rb"[^\n]*?</script>", live)
    print(f"静态 CDN 注入的脚本：{len(injected)} 处 / {sum(len(x) for x in injected)} 字节"
          "（这部分是正常的额外内容）")
    for one in injected:
        print("   ", one.decode("utf-8", errors="replace")[:120])

    ok = live_bad == 0 and not non_ascii
    print("\n结果: " + ("PASS —— 线上没有坏字符，产物是纯 ASCII" if ok
                        else "FAIL —— 线上有被劈坏的中文，或产物里还有非 ASCII"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
