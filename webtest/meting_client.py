# webtest/meting_client.py

import json
import os
import subprocess
import threading
import time
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

# ---------------------------------------------------------------------------
# 进程内 TTL 缓存
# ---------------------------------------------------------------------------
# 为什么要缓存：
#   Meting 的每次查询都是一次 stdio 往返，且被全局锁串行化。
#   同一首歌（尤其热门歌）会在不同用户的推荐结果里反复出现，
#   缓存能把这部分开销完全消掉。
#
# 正缓存（找到）TTL 较长，负缓存（找不到）TTL 较短 ——
# 避免"当时没找到"的结论长期生效。
NET_CACHE_TTL_HIT = 24 * 3600      # 24 小时
NET_CACHE_TTL_MISS = 6 * 3600      # 6 小时
NET_MAX_TOTAL_SECONDS = 6.0        # 单次 enrich 的总时间预算上限

_net_cache = {}
_net_cache_lock = threading.Lock()


def _cache_key(track_name, artist_name):
    return f"{(track_name or '').strip().lower()}||{(artist_name or '').strip().lower()}"


def _get_cache(key):
    with _net_cache_lock:
        entry = _net_cache.get(key)
        if not entry:
            return None
        if time.time() > entry["expire"]:
            _net_cache.pop(key, None)
            return None
        return entry["data"]


def _set_cache(key, data):
    ttl = NET_CACHE_TTL_HIT if (data or {}).get("netease_url") else NET_CACHE_TTL_MISS
    with _net_cache_lock:
        _net_cache[key] = {"data": data, "expire": time.time() + ttl}


def net_cache_stats():
    """返回缓存规模（供评测/诊断使用）"""
    with _net_cache_lock:
        now = time.time()
        live = sum(1 for v in _net_cache.values() if v["expire"] > now)
        hits = sum(1 for v in _net_cache.values()
                   if v["expire"] > now and (v["data"] or {}).get("netease_url"))
        return {"总条目": len(_net_cache), "有效": live, "有链接": hits}


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


def search_netease(track_name, artist_name="", use_cache=True):
    """
    搜索网易云歌曲，严格匹配歌名+歌手，找不到返回 None。

    【缓存内建】缓存放这一层（而不是放在 enrich_results_with_netease 里），
    这样任何调用方 —— 包括按需获取单首链接的接口 —— 都自动享受缓存。
    """
    key = _cache_key(track_name, artist_name)

    if use_cache:
        hit = _get_cache(key)
        if hit is not None:
            # 命中缓存：有 url 返回详情，否则返回 None（负缓存）
            return hit if hit.get("netease_url") else None

    info = _search_netease_remote(track_name, artist_name)

    if use_cache:
        _set_cache(key, info or {"netease_url": "", "netease_id": ""})

    return info


def _search_netease_remote(track_name, artist_name=""):
    """真正发起 Meting MCP 查询（无缓存）"""
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


def enrich_results_with_netease(results, max_workers=5, timeout=2.5, cache_only=False):
    """
    为推荐结果批量补充网易云跳转链接。

    【性能说明】
    Meting 是"全局单例 MCP 子进程 + 全局锁"的客户端，所有查询实际串行
    执行（每次一次 stdio 往返）。若对 20 条结果逐条查询，最坏会累积到
    几十秒 —— 实测曾出现单请求 69 秒。

    因此这里的策略是：
    1. 先查进程内 TTL 缓存，命中的直接返回（绝大多数重复请求走这条路）
    2. 只对未命中的条目发起查询，且受总时间预算约束
    3. 超预算的条目直接留空 —— 网易云链接是"锦上添花"，
       不应该让推荐主流程为它等待

    Args:
        max_workers: 保留参数以兼容旧调用（当前实现不再依赖它并行）
        timeout: 单条查询的等待上限（秒）
        cache_only: 只读缓存不发起查询（用于对延迟敏感的路径）
    """
    if not results:
        return results

    # 1. 先填缓存命中的
    pending = []
    for item in results:
        key = _cache_key(item.get("track_name", ""), item.get("artist_name", ""))
        hit = _get_cache(key)
        if hit:
            item["netease_url"] = hit["netease_url"]
            item["netease_id"] = hit["netease_id"]
        elif cache_only:
            item["netease_url"] = ""
            item["netease_id"] = ""
        else:
            pending.append(item)

    if not pending:
        return results

    # 2. 只对未命中的发起查询，受总预算约束
    budget = min(timeout * len(pending), NET_MAX_TOTAL_SECONDS)
    deadline = time.time() + budget
    done = 0
    for item in pending:
        if time.time() >= deadline:
            item["netease_url"] = ""
            item["netease_id"] = ""
            continue
        try:
            info = search_netease(item.get("track_name", ""),
                                  item.get("artist_name", ""))
        except Exception:
            info = None
        if info:
            item["netease_url"] = info["netease_url"]
            item["netease_id"] = info["netease_id"]
            _set_cache(_cache_key(item.get("track_name", ""),
                                  item.get("artist_name", "")), info)
        else:
            item["netease_url"] = ""
            item["netease_id"] = ""
            # 负缓存：避免反复为找不到的歌浪费查询预算
            _set_cache(_cache_key(item.get("track_name", ""),
                                  item.get("artist_name", "")),
                       {"netease_url": "", "netease_id": ""})
        done += 1

    return results