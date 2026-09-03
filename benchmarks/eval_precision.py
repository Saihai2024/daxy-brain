"""快速精度评测:30 个 query 的 top-1 topic 是否实义相关"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根
from brain.cognitive_brain import CognitiveBrain, _substantive_overlap

brain = CognitiveBrain()
print(f"memories={len(brain.memory)} encoder=v{getattr(brain,'encoder_version','?')}\n")

CASES = [
    "光速是多少", "量子纠缠是什么", "介绍一下相对论", "黑洞", "DNA是什么",
    "原子是什么", "什么是熵", "牛顿第二定律", "万有引力", "光合作用",
    "电磁波", "薛定谔方程", "进化论", "基因是什么", "细胞分裂",
    "微积分", "勾股定理", "圆周率", "概率论", "线性代数",
    "病毒是什么", "疫苗原理", "抗生素", "宇宙大爆炸", "太阳系",
    "月球", "火星", "银河系", "超新星", "中子星",
]
ok = bad = miss = 0
t0 = time.time()
for q in CASES:
    ans, dbg = brain.answer(q)
    topic = dbg.get("topic")
    if ans is None:
        miss += 1
        mark = "MISS"
    elif _substantive_overlap(q, topic or ""):
        ok += 1
        mark = "ok  "
    else:
        bad += 1
        mark = "BAD "
    sim = dbg.get("similarity")
    sim_s = f"{sim:.3f}" if isinstance(sim, float) else "  -  "
    print(f"  {mark} [{q:12}] -> {topic}  sim={sim_s}")
dt = (time.time() - t0) / len(CASES) * 1000
print(f"\nok={ok} bad={bad} miss={miss} / {len(CASES)}   avg_latency={dt:.0f}ms")
