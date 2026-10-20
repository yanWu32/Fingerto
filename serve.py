"""一键启动入口：运行一个文件即可在浏览器打开前端页面。

用法（在本仓库根目录执行）：

    python serve.py                 # 默认 127.0.0.1:5173，并自动打开浏览器
    python serve.py --port 8080     # 自定义端口
    python serve.py --no-browser    # 不自动打开浏览器
    python serve.py --backend       # 同时拉起真实后端（app.app）并把 /api 反代到它

说明：
- 前端是构建好的 Vue SPA（frontend/dist），默认处于 mock 模式（VITE_USE_MOCK=true），
  无需后端即可完整演示「手语视频 → 文本 → 任务执行」全链路（含 /demo 页面）。
- 带 --backend 时，会在子线程启动 app.app（Gradio 后端，需已安装 gradio/mediapipe 等），
  并把所有 /api/* 请求反代到后端端口，从而切换到真实数据。
- 纯标准库实现，不依赖第三方包；SPA 历史路由（如 /demo 深链）自动回退到 index.html。
"""
from __future__ import annotations

import argparse
import http.client
import http.server
import mimetypes
import os
import sys
import threading
import webbrowser
from urllib.parse import urlparse

# 确保 .js / .mjs 以正确的 MIME 返回（避免某些环境下被当成 text/plain 触发 CSP 拦截）
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIST = os.path.join(ROOT, "frontend", "dist")
DEFAULT_PORT = 5173
DEFAULT_BACKEND_PORT = 8000


class SpaHandler(http.server.SimpleHTTPRequestHandler):
    """静态服务 SPA：文件存在则正常返回，否则回退到 index.html（支持 history 路由）。"""

    def __init__(self, *args, backend_port: int | None = None, **kwargs):
        self.backend_port = backend_port
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:  # noqa: D401 - override to add SPA headers
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _serve_spa_index(self) -> None:
        index_path = os.path.join(self.directory, "index.html")  # type: ignore[attr-defined]
        try:
            with open(index_path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404, "index.html not found (run `npm run build` first)")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        # 非 SPA 资源（含扩展名的文件）按静态文件处理；其余交给 index.html
        path = urlparse(self.path).path
        rel = path.lstrip("/")
        candidate = os.path.normpath(os.path.join(self.directory, rel))  # type: ignore[attr-defined]
        if rel and os.path.isfile(candidate):
            return super().do_GET()
        return self._serve_spa_index()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.backend_port is None or not urlparse(self.path).path.startswith("/api"):
            self.send_error(404, "No backend proxy enabled for this path")
            return
        self._proxy_to_backend("POST")

    def _proxy_to_backend(self, method: str) -> None:
        parsed = urlparse(self.path)
        conn = http.client.HTTPConnection("127.0.0.1", self.backend_port)  # type: ignore[arg-type]
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        try:
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001 - 代理失败给出清晰报错
            self.send_error(502, f"backend proxy error: {e}")
            return
        payload = resp.read()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "connection"):
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # 精简日志，只打印访问行
        sys.stderr.write("[serve] %s - %s\n" % (self.address_string(), fmt % args))


def start_backend(backend_port: int) -> None:
    """在子线程拉起真实后端（app.app），失败则仅告警，不影响前端。"""
    try:
        from app import app as app_mod  # 懒加载：缺依赖时不崩
    except Exception as e:  # noqa: BLE001
        print(f"[serve] 无法导入后端 app.app（可能缺少依赖）：{e}", file=sys.stderr)
        return
    try:
        print(f"[serve] 正在启动真实后端 app.app（端口 {backend_port}）…")
        app_mod.launch(server_name="127.0.0.1", server_port=backend_port, show_error=True)
    except Exception as e:  # noqa: BLE001
        print(f"[serve] 后端启动失败：{e}", file=sys.stderr)


def build_handler(backend_port: int | None, dist_dir: str):
    def factory(*args, **kwargs):
        return SpaHandler(*args, directory=dist_dir, backend_port=backend_port, **kwargs)
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="一键启动前端 SPA（默认 mock 模式）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口（默认 5173）")
    parser.add_argument("--dir", default=DEFAULT_DIST, help="SPA 构建目录（默认 frontend/dist）")
    parser.add_argument("--backend", action="store_true", help="同时拉起真实后端并反代 /api")
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT, help="后端端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    dist_dir = os.path.abspath(args.dir)
    if not os.path.isdir(dist_dir) or not os.path.isfile(os.path.join(dist_dir, "index.html")):
        print(f"[serve] 找不到构建产物：{dist_dir}/index.html", file=sys.stderr)
        print("        请先构建前端：cd frontend && npm run build", file=sys.stderr)
        sys.exit(1)

    backend_port = args.backend_port if args.backend else None
    if args.backend:
        t = threading.Thread(target=start_backend, args=(args.backend_port,), daemon=True)
        t.start()

    handler = build_handler(backend_port, dist_dir)
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    # 让 SimpleHTTPRequestHandler 知道根目录
    http.server.SimpleHTTPRequestHandler.directory = dist_dir

    url = f"http://{args.host}:{args.port}/"
    print("=" * 56)
    print("  手语对话系统 · 前端已启动")
    print(f"  → 打开：{url}")
    print(f"  → 全链路演示页：{url}demo")
    if args.backend:
        print(f"  → 真实后端（反代 /api）：http://127.0.0.1:{args.backend_port}/")
    print("  按 Ctrl+C 停止")
    print("=" * 56)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 已停止。")
        httpd.shutdown()


if __name__ == "__main__":
    main()
