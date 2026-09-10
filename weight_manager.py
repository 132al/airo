# test_meting.py
from webtest.meting_client import get_client

client = get_client()

# 列出所有工具
tools = client.list_tools()
import json
print("可用工具:")
print(json.dumps(tools, indent=2, ensure_ascii=False))

# 调用搜索
resp = client.call_tool("search", {
    "keywords": "Shape of You Ed Sheeran",
    "platform": "netease",
})
print("\n搜索结果:")
print(json.dumps(resp, indent=2, ensure_ascii=False))