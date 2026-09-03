"""
用闲聊语料给 CognitiveBrain 做底子(可重复跑,在线追加学习)。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根

import argparse
import time
from brain.cognitive_brain import CognitiveBrain


def load_pairs(input_path, target_path, limit=None):
    fi = open(input_path, encoding="utf-8")
    ft = open(target_path, encoding="utf-8")
    n = 0
    for li, lt in zip(fi, ft):
        q = li.strip().replace(" ", "")
        a = lt.strip().replace(" ", "")
        if not q or not a:
            continue
        yield q, a
        n += 1
        if limit and n >= limit:
            break
    fi.close()
    ft.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="corpus/brain/input.txt")
    ap.add_argument("--target", default="corpus/brain/target.txt")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--max_q_len", type=int, default=20)
    ap.add_argument("--max_a_len", type=int, default=20)
    args = ap.parse_args()

    brain = CognitiveBrain()
    print("brain stats before:", brain.stats())

    t0 = time.time()
    seen = 0
    for i, (q, a) in enumerate(load_pairs(args.input, args.target, args.limit)):
        if len(q) > args.max_q_len or len(a) > args.max_a_len:
            continue
        brain.learn_dialogue(q, a, lr=args.lr)
        seen += 1
        if seen % 200 == 0:
            elapsed = time.time() - t0
            rate = seen / elapsed
            print(f"  [{seen}] {rate:.1f} pairs/s  vocab={len(brain.motor_sdr)}  steps={brain.steps}")
        if seen % args.save_every == 0:
            brain.save()
            print(f"  saved checkpoint @ {seen}")

    brain.save()
    print(f"done in {time.time() - t0:.1f}s. final stats: {brain.stats()}")

    # quick demo
    for q in ["你好", "你叫什么名字", "我失恋了", "晚安", "你是谁"]:
        ans = brain.generate(q, max_len=15)
        print(f"  Q: {q}\n  A: {ans}\n")


if __name__ == "__main__":
    main()
