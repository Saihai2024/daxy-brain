# -*- coding: utf-8 -*-
"""离线看思维逻辑路对各类提问会怎么答(不动线上服务)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_config import get_pg_conn
from brain.thinking_index import ThinkingIndex

idx = ThinkingIndex(conn_factory=get_pg_conn)
print("语料 %d 条\n" % len(idx.rows))

QS = [
    "量子纠缠是什么",
    "你好",
    # ↓ 任务类提问要换成**你自己语料里真有的**,否则命不中很正常
    "帮我调研一下这个项目的多模型路由是怎么实现的,各 provider 的用量 header 在哪解析",
    "详情页前端会打哪些后端接口,帮我逐个映射到 handler",
    "我要把一个 FastAPI 服务的响应时间从 300ms 降到 100ms,应该先做什么",
    "帮我规划一趟五天的京都行程,预算八千",
]
for q in QS:
    ans, dbg = idx.answer(q)
    print("=" * 70)
    print("Q:", q)
    print("dbg:", dbg)
    if ans:
        print(ans[:1400])
    else:
        print("→ 不接,落回大脑主路径")
    print()
