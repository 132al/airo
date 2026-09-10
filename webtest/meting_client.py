# webtest/meting_client.py

import json
import subprocess
import threading
import os
from concurrent.futures import ThreadPoolExecutor


class MetingClient:
    """MCP stdio 客户端，用于和 Meting-Agent 通信"""

    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self._request_id = 0
        self._start()
        self._initialize()

    def _start(self):
        """启动子进程"""
        NODE_PATH = r"C:\Program Files\nodejs\node.exe"
        NPX_CLI = r"C:\Program Files\nodejs\node_modules\npm\bin\npx-cli.js"

        if not os.path.exists(NODE_PATH):
            raise FileNotFoundError(f"找不到 node.exe: {NODE_PATH}")
        if not os.path.exists(NPX_CLI):
            raise FileNotFoundError(f"找不到 npx-cli.js: {NPX_CLI}")

        env = os.environ.copy()
        node_dir = r"C:\Program Files\nodejs"
        env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")

        self.process = subprocess.Popen(
            [NODE_PATH, NPX_CLI, "@eldment/meting-agent"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

    def _send(self, method, params=None):
        """发送 JSON-RPC 请求，返回响应"""
        with self.lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
            }
            if params is not None:
                request["params"] = params

            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            line = self.process.stdout.readline()
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None

    def _initialize(self):
        """初始化 MCP 连接"""
        self._send("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "django", "version": "1.0"},
        })
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        self.process.stdin.write(json.dumps(notification) + "\n")
        self.process.stdin.flush()

    def list_tools(self):
        """列出可用工具"""
        return self._send("tools/list")

    def call_tool(self, name, arguments):
        """调用工具"""
        return self._send("tools/call", {
            "name": name,
            "arguments": arguments,
        })

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=5)


# 全局单例
_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MetingClient()
    return _client


def _parse_content(resp):
    """从 MCP 响应中提取 JSON 数据"""
    if not resp:
        return None
    content = resp.get("result", {}).get("content", [])
    if not content:
        return None
    text = content[0].get("text", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def get_cover_url(song_id, size=300):
    """获取封面 URL"""
    try:
        client = get_client()
        resp = client.call_tool("pic", {
            "platform": "netease",
            "id": str(song_id),
            "size": size,
        })
        data = _parse_content(resp)
        if data and data.get("ok"):
            return data.get("data", {}).get("url", "")
    except Exception as e:
        print(f"[Meting] 封面获取失败 {song_id}: {e}")
    return ""


def search_netease(track_name, artist_name=""):
    """搜索网易云歌曲，返回封面和链接"""
    try:
        client = get_client()
        query = f"{track_name} {artist_name}".strip()

        resp = client.call_tool("search", {
            "platform": "netease",
            "keyword": query,
            "limit": 5,
        })
        data = _parse_content(resp)
        if not data or not data.get("ok"):
            return None

        songs = data.get("data", [])
        if not songs:
            return None

        song = songs[0]
        song_id = song.get("id")
        if not song_id:
            return None

        cover_url = get_cover_url(song_id, size=300)

        return {
            "netease_id": str(song_id),
            "cover_url": cover_url,
            "netease_url": f"https://music.163.com/#/song?id={song_id}",
        }
    except Exception as e:
        print(f"[Meting] 查询失败 {track_name}: {e}")
        return None


def enrich_results_with_netease(results, max_workers=5):
    """为推荐结果批量补充网易云信息"""
    if not results:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in results:
            future = executor.submit(
                search_netease,
                item.get("track_name", ""),
                item.get("artist_name", ""),
            )
            futures.append((item, future))

        for item, future in futures:
            try:
                info = future.result(timeout=10)
            except Exception:
                info = None

            if info:
                item["cover_url"] = info["cover_url"]
                item["netease_url"] = info["netease_url"]
                item["netease_id"] = info["netease_id"]
            else:
                item["cover_url"] = ""
                item["netease_url"] = ""
                item["netease_id"] = ""

    return results