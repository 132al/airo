# webtest/intent_parser.py
import json
import re
from .ollama_client import client

class IntentParser:
    def parse(self, user_input, verbose=True):
        prompt = f"""分析用户的音乐请求，只输出JSON，不要其他内容。

用户说："{user_input}"

输出格式（严格JSON）：
{{"search_type": "类型", "keyword": "关键词", "mood": null}}

类型只能是：artist, genre, track, hot, general

示例（严格参考）：
"周杰伦" → {{"search_type": "artist", "keyword": "周杰伦", "mood": null}}
"摇滚" → {{"search_type": "genre", "keyword": "rock", "mood": null}}
"rock" → {{"search_type": "genre", "keyword": "rock", "mood": null}}
"爵士" → {{"search_type": "genre", "keyword": "jazz", "mood": null}}
"推荐几首好听的歌" → {{"search_type": "hot", "keyword": "", "mood": null}}
"Shape of You" → {{"search_type": "track", "keyword": "Shape of You", "mood": null}}
"123" → {{"search_type": "general", "keyword": "", "mood": null}}

只输出JSON，不要解释。"""
        
        response = client.generate(prompt, max_tokens=100)
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"🔍 意图解析 - 用户输入: '{user_input}'")
            print(f"📝 大模型原始响应: {response}")
        
        try:
            result = json.loads(response)
            result['confidence'] = 'high'
            if verbose:
                print(f"✅ 解析结果: {result}")
                print(f"{'='*50}\n")
            return result
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            # 尝试从响应中提取 JSON
            try:
                # 查找 JSON 片段
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    result['confidence'] = 'medium'
                    if verbose:
                        print(f"✅ 从响应中提取JSON: {result}")
                        print(f"{'='*50}\n")
                    return result
            except:
                pass
            
            fallback = self._fallback_parse(user_input)
            fallback['confidence'] = 'low'
            if verbose:
                print(f"⚠️ 使用备用解析: {fallback}")
                print(f"{'='*50}\n")
            return fallback
    
    def _fallback_parse(self, user_input):
        """备用解析：关键词匹配"""
        user_input_lower = user_input.lower()
        
        genre_map = {
            'pop': ['pop', '流行'],
            'rock': ['rock', '摇滚'],
            'jazz': ['jazz', '爵士'],
            'classical': ['classical', '古典'],
            'hip-hop': ['hip hop', 'hiphop', '嘻哈', 'rap', '说唱'],
            'country': ['country', '乡村'],
            'blues': ['blues', '布鲁斯'],
            'electronic': ['electronic', '电子', 'edm'],
            'folk': ['folk', '民谣'],
            'metal': ['metal', '金属'],
            'punk': ['punk', '朋克'],
            'reggae': ['reggae', '雷鬼'],
            'soul': ['soul', '灵魂'],
            'dance': ['dance', '舞曲'],
            'acoustic': ['acoustic', '原声'],
        }
        
        for genre_key, keywords in genre_map.items():
            for kw in keywords:
                if kw in user_input_lower:
                    return {'search_type': 'genre', 'keyword': genre_key, 'mood': None}
        
        if '的歌' in user_input or '歌手' in user_input:
            match = re.search(r'([\u4e00-\u9fa5a-zA-Z]+)的?歌', user_input)
            if match:
                return {'search_type': 'artist', 'keyword': match.group(1), 'mood': None}
        
        return {'search_type': 'hot', 'keyword': '', 'mood': None}

intent_parser = IntentParser()