# webtest/ollama_client.py
import requests
import time


class OllamaClient:
    def __init__(self, model="qwen3:4b-instruct-2507-q4_K_M", base_url="http://localhost:11434", num_ctx=8192):
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx

    def generate(self, prompt, max_tokens=800, temperature=0.7, format=None):
        for attempt in range(3):
            try:
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

                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()
                    result = data.get("message", {}).get("content", "").strip()
                    if result:
                        return result
            except Exception as e:
                print(f"⚠️ 调用失败: {e}")
                if attempt < 2:
                    time.sleep(1)

        return ""


client = OllamaClient()