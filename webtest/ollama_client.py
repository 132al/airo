# webtest/ollama_client.py
"""
Ollama 调用封装。

【重要的超时设计】
原实现是 `for attempt in range(3)` + 每次 `timeout=120`，
最坏情况 3*120 + 2*1 = 362 秒。而 Django dev server 是单线程的，
一次这样的阻塞会把整个站点卡住 —— 这是之前"反馈页一直加载中"的根因。

现在的策略：
1. 连接错误（Ollama 未启动 / 端口拒绝 / 空壳托盘）→ **立即失败，不重试**。
   这类错误重试毫无意义，只会白白浪费 6 分钟。
2. 读取超时（模型在跑但慢）→ 重试，但只重试 1 次，且用更短的超时。
3. (connect, read) 拆分为两个超时：连接 5 秒，读取由 read_timeout 控制。
4. 连续失败后进入冷却期，冷却内直接返回，避免反复撞墙。
"""

import threading
import time

import requests

# 连接超时：本机服务，5 秒足够
CONNECT_TIMEOUT = 5

# 读取超时：模型生成可能需要较久，但不应无上限
DEFAULT_READ_TIMEOUT = 90

# 失败后的冷却时间（秒）：期间所有请求直接返回，不再尝试连接
COOLDOWN_SECONDS = 30

# 何时才值得重试：只有"连上了但读超时"才重试
MAX_READ_RETRIES = 1


class OllamaClient:
    def __init__(
        self,
        model="qwen3:4b-instruct-2507-q4_K_M",
        base_url="http://localhost:11434",
        num_ctx=4096,
    ):
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        # 最近一次失败时间戳；用于冷却判断
        self._last_failure = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 冷却控制
    # ------------------------------------------------------------------
    def _in_cooldown(self):
        """是否处于失败冷却期"""
        with self._lock:
            if not self._last_failure:
                return False
            if time.time() - self._last_failure < COOLDOWN_SECONDS:
                return True
            self._last_failure = 0.0
            return False

    def _mark_failure(self):
        with self._lock:
            self._last_failure = time.time()

    def _mark_success(self):
        with self._lock:
            self._last_failure = 0.0

    # ------------------------------------------------------------------
    # 可用性
    # ------------------------------------------------------------------
    def is_available(self, timeout=CONNECT_TIMEOUT):
        """轻量探测：Ollama 服务是否在线"""
        if self._in_cooldown():
            return False
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 主调用
    # ------------------------------------------------------------------
    def generate(self, prompt, max_tokens=800, temperature=0.7, format=None,
                 read_timeout=DEFAULT_READ_TIMEOUT):
        """
        调用 LLM 生成文本。失败时返回空串（调用方需有兜底）。

        Args:
            read_timeout: 读取超时秒数。对"可选增强"类调用（如画像摘要）
                          可以传更小的值以更快失败。
        """
        # 冷却期内直接返回，避免反复撞墙
        if self._in_cooldown():
            print("[LLM] 处于失败冷却期，跳过调用")
            return ""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
        }
        if format:
            payload["format"] = format

        url = f"{self.base_url}/api/chat"
        timeout = (CONNECT_TIMEOUT, read_timeout)

        # 第 1 次
        result, is_conn_error = self._post(url, payload, timeout)
        if result:
            self._mark_success()
            return result

        # 连接错误 → 不值得重试（Ollama 没起来，重试多少次都一样）
        if is_conn_error:
            self._mark_failure()
            print("[LLM] 连接失败（Ollama 未运行？），跳过重试。"
                  "可运行 ollama_doctor.bat 修复")
            return ""

        # 读超时 → 重试有限次数
        for attempt in range(MAX_READ_RETRIES):
            print(f"[LLM] 读取超时，重试 {attempt + 1}/{MAX_READ_RETRIES}")
            time.sleep(1)
            result, is_conn_error = self._post(url, payload, timeout)
            if result:
                self._mark_success()
                return result
            if is_conn_error:
                self._mark_failure()
                return ""

        self._mark_failure()
        return ""

    def _post(self, url, payload, timeout):
        """
        单次 POST。

        Returns: (文本 或 "", 是否连接类错误)
        """
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            # 端口没人监听 / 拒绝连接 —— 重试无意义
            print(f"[LLM] 连接被拒绝: {type(e).__name__}")
            return "", True
        except requests.exceptions.ReadTimeout:
            print("[LLM] 读取超时（模型响应过慢）")
            return "", False
        except requests.exceptions.Timeout:
            print("[LLM] 请求超时")
            return "", False
        except Exception as e:
            # 注意：不要在这里打印 emoji。Windows 控制台默认 GBK，
            # 打印 ⚠️ 之类会抛 UnicodeEncodeError，把真实异常盖掉。
            print(f"[LLM] 调用异常: {type(e).__name__}: {e}")
            return "", False

        if response.status_code != 200:
            print(f"[LLM] HTTP {response.status_code}")
            return "", False

        try:
            data = response.json()
        except Exception:
            print("[LLM] 响应不是合法 JSON")
            return "", False

        return (data.get("message", {}).get("content", "") or "").strip(), False


client = OllamaClient()