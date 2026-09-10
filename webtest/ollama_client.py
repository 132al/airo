# webtest/ollama_client.py
import requests
import time

class OllamaClient:
    def __init__(self, model="qwen3:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt, max_tokens=150, temperature=0.7, top_p=0.9):
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": 40,
                            "repeat_penalty": 1.1,
                            "stop": ["<think", "</think>", "思考"],  # 阻止思考标签
                        }
                    },
                    timeout=1200
                )
                
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    if result and len(result.strip()) > 0:
                        return result
            except Exception as e:
                print(f"⚠️ 调用失败: {e}")
                if attempt < 2:
                    time.sleep(1)
        
        return ""

client = OllamaClient()