# -*- coding: utf-8 -*-
"""上线后端到端验收:思维逻辑路该接的接、不该接的不接,延迟没退化。"""
import json, os, time, urllib.request, http.cookiejar

BASE = os.environ.get("DASH_BASE", "http://127.0.0.1:8000")
USER = os.environ.get("DASH_USER", "root")
PASS = os.environ.get("DASH_PASS", "")
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def post(path, obj, timeout=90):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with op.open(req, timeout=timeout) as r:
        body = r.read()
    return (time.perf_counter() - t0) * 1000, json.loads(body.decode("utf-8"))

post("/api/login", {"username": USER, "password": PASS})
post("/api/chat", {"message": "预热", "user_id": "warm"})

CASES = [
    ("百科", "量子纠缠是什么"),
    ("百科", "什么是机器学习"),
    ("招呼", "你好"),
    ("身份", "你叫什么名字"),
    # ↓ 这两条要换成**你自己语料里真有的**任务简报,否则必然不命中。
    ("任务·应命中", "详情页前端会打哪些后端接口,帮我逐个映射到 handler"),
    ("任务·应命中", "只读调研,把发布门禁的校验逻辑摸清楚,判断这个提交会不会被拦"),
    ("任务·无关", "帮我规划一趟五天的京都行程,预算八千"),
    ("任务·无关", "我想学做红烧肉,详细步骤是什么"),
]
for tag, q in CASES:
    ms, j = post("/api/chat", {"message": q, "user_id": "e2e"})
    a = (j.get("response") or "").replace("\n", " / ")
    kind = "思维逻辑" if a.startswith(("这类任务我见过", "不太确定对不对得上")) else "大脑/其它"
    print("%-12s %6.0fms  [%s]  %s | %s" % (tag, ms, kind, q[:28], a[:70]))
