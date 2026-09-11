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

    def call_tool(self, name, arguments):
        return self._send("tools/call", {
            "name": name,
            "arguments": arguments,
        })

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=5)


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


def search_netease(track_name, artist_name=""):
    """搜索网易云歌曲，严格匹配歌名+歌手，找不到返回 None"""
    try:
        cli = get_client()
        query = f"{track_name} {artist_name}".strip()

        resp = cli.call_tool("search", {
            "platform": "netease",
            "keyword": query,
            "limit": 10,   # 多取几条，提高匹配概率
        })

        data = _parse_content(resp)
        if not data or not data.get("ok"):
            return None

        songs = data.get("data", [])
        if not songs:
            return None

        track_lower = track_name.lower().strip()
        artist_lower = artist_name.lower().strip() if artist_name else ""

        for song in songs:
            song_name = str(song.get("name", "")).lower().strip()

            # 歌名匹配：包含关系（双向）
            name_match = (track_lower in song_name) or (song_name in track_lower)

            # 歌手匹配
            song_artists = song.get("artist", [])
            if isinstance(song_artists, list):
                song_artist_str = " ".join(str(a) for a in song_artists).lower()
            else:
                song_artist_str = str(song_artists).lower()

            artist_match = (not artist_lower) or (artist_lower in song_artist_str)

            if name_match and artist_match:
                song_id = song.get("id")
                if not song_id:
                    continue
                return {
                    "netease_id": str(song_id),
                    "netease_url": f"https://music.163.com/#/song?id={song_id}",
                }

        # 没找到匹配的
        print(f"[Meting] 未找到匹配: {track_name} - {artist_name}")
        return None

    except Exception as e:
        print(f"[Meting] 查询失败 {track_name}: {e}")
        return None


def enrich_results_with_netease(results, max_workers=5):
    """为推荐结果批量补充网易云跳转链接"""
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
                item["netease_url"] = info["netease_url"]
                item["netease_id"] = info["netease_id"]
            else:
                item["netease_url"] = ""
                item["netease_id"] = ""

    return results