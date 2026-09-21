"""验证：前端不再显示百分比"""
import sys, os, re, json
sys.path.insert(0, r"f:\airo\aipro")
os.chdir(r"f:\airo\aipro")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
import django
django.setup()
from django.test import Client
from django.contrib.auth.models import User

User.objects.filter(username="__vd__").delete()
User.objects.create_user(username="__vd__", password="x")
c = Client()
c.login(username="__vd__", password="x")
r = c.get("/")
html = r.content.decode("utf-8", "replace")

print("VD HTTP", r.status_code)
# 1) 模板源码里不应再有相似的渲染逻辑
print("VD 模板含 'it.similarity) * 100' :", "it.similarity) * 100" in html)
print("VD 模板含 'row-score' 渲染       :",
      "${pct > 0" in html or "row-score\">${pct}" in html)
print("VD 模板含 '匹配度' 字样           :", "匹配度" in html)
print("VD 模板含 sim_percent 引用        :", "sim_percent" in html)
# 2) CSS 定义还在（不影响，仅未被使用）
print("VD CSS .row-score 仍定义（无害）  :", ".row-score" in html)
print("VD CSS .detail-score 仍定义（无害）:", ".detail-score" in html)
# 3) reason 仍传 similarity（用于生成理由，非展示）
print("VD reason 仍传 similarity（保留）  :", "similarity: it.similarity" in html)

User.objects.filter(username="__vd__").delete()
print("VD DONE")
