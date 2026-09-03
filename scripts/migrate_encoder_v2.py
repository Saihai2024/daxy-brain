"""一次性迁移:brain.pkl 编码器 v1 -> v2(unigram + 实义 bigram 双通道)。

CognitiveBrain._load 检测到版本不符会自动在内存重编码全部 keys
(encode_batch 批量 matmul,64 核下几十秒);本脚本负责把结果落盘,
之后所有进程 load 时版本匹配、零开销。

用法(建议先停所有服务):
  python scripts/migrate_encoder_v2.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.cognitive_brain import CognitiveBrain, ENCODER_VERSION

LOCK = ROOT / "logs" / "BRAIN_RELOAD_IN_PROGRESS"


def main():
    LOCK.parent.mkdir(exist_ok=True)
    LOCK.write_text("migrate_encoder_v2")
    try:
        t0 = time.time()
        print(f"[migrate] loading brain (旧版本会自动重编码到 v{ENCODER_VERSION}) ...")
        brain = CognitiveBrain()
        print(f"[migrate] loaded: {len(brain.memory)} memories "
              f"({time.time() - t0:.1f}s)")
        t1 = time.time()
        brain.save()
        print(f"[migrate] saved v{ENCODER_VERSION} brain.pkl ({time.time() - t1:.1f}s)")
        # 快速 sanity:几个 query 的 top-1
        for q in ("光速是多少", "量子纠缠是什么", "黑洞", "你叫什么名字"):
            ans, dbg = brain.answer(q)
            topic = dbg.get("topic")
            sim = dbg.get("similarity")
            sim_s = f"{sim:.3f}" if isinstance(sim, float) else "-"
            print(f"  [{q}] -> topic={topic} sim={sim_s} reason={dbg.get('reason')}")
    finally:
        LOCK.unlink(missing_ok=True)
    print("[migrate] done.")


if __name__ == "__main__":
    main()
