# -*- coding: utf-8 -*-
"""把 brain_v2.pkl 从「稠密 key」迁到「稀疏列号」格式。

背景:每条 key 是 kWTA 二值向量,实测每条恰好 960 个 1 / 16384 维,
按稠密 float32 存 = 4.79GB 里 94% 的字节在存零。改存 uint16 列号后约 140MB。
收益:pkl 5.1GB → ~450MB,load 37.9s → ~3s,`_key_stack` 的 np.stack(23.6s)整步消失。

**先校验再落盘**:逐条比对"展开后的稀疏 key"与"原稠密 key"必须逐元素相同,
有任何一条对不上就中止,不碰原文件。

用法:
  python scripts/migrate_brain_sparse.py --check      # 只校验,不写
  python scripts/migrate_brain_sparse.py --apply      # 校验通过后原地升级
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.cognitive_brain import _key_to_idx

PKL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "brain", "models", "brain_v2.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="校验通过后真的替换原文件")
    ap.add_argument("--path", default=PKL)
    args = ap.parse_args()

    print("源文件 %s (%.2f GB)" % (args.path, os.path.getsize(args.path) / 1e9))
    t = time.time()
    with open(args.path, "rb") as f:
        d = pickle.load(f)
    print("load %.1fs" % (time.time() - t))

    if d.get("key_format") == "idx_u16":
        print("已经是稀疏格式,无需迁移。")
        return 0

    mem = d["memory"]
    n = len(mem)
    print("memories=%d assoc_dim=%s assoc_topk=%s"
          % (n, d.get("assoc_dim"), d.get("assoc_topk")))

    t = time.time()
    keys_idx = [_key_to_idx(m.get("key")) for m in mem]
    print("转稀疏 %.1fs" % (time.time() - t))

    nnz = np.array([len(i) for i in keys_idx])
    print("每条 nnz: 中位=%d 最小=%d 最大=%d" % (np.median(nnz), nnz.min(), nnz.max()))

    # ---- 逐条校验:展开回去必须和原来逐元素相同 ----
    dim = int(d.get("assoc_dim") or mem[0]["key"].shape[0])
    t = time.time()
    bad = 0
    probe = np.zeros(dim, dtype=np.float32)
    for i, (m, idx) in enumerate(zip(mem, keys_idx)):
        probe[:] = 0.0
        probe[idx] = 1.0
        if not np.array_equal(probe, m["key"]):
            bad += 1
            if bad <= 3:
                print("  ★不一致 i=%d topic=%r" % (i, m.get("topic")))
    print("逐条校验 %.1fs: %d/%d 完全相同" % (time.time() - t, n - bad, n))
    if bad:
        print("★ 有 %d 条对不上,中止,不动原文件。" % bad)
        return 2

    # 顺带确认 key 确实是严格二值(否则列号表示法本身就不成立)
    vals = set()
    for m in mem[:2000]:
        vals.update(np.unique(m["key"]).tolist())
    print("抽样 2000 条的取值集合 =", sorted(vals))
    if not vals <= {0.0, 1.0}:
        print("★ key 不是严格二值,列号表示法不成立,中止。")
        return 2

    out = dict(d)
    out["memory"] = [{k: v for k, v in m.items() if k not in ("key", "_idx")}
                     for m in mem]
    out["keys_idx"] = keys_idx
    out["key_format"] = "idx_u16"

    tmp = "%s.sparse.tmp.%d" % (args.path, os.getpid())
    t = time.time()
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    new_sz = os.path.getsize(tmp)
    print("写新格式 %.1fs, %.0f MB (原 %.0f MB, 省 %.1fx)"
          % (time.time() - t, new_sz / 1e6, os.path.getsize(args.path) / 1e6,
             os.path.getsize(args.path) / new_sz))

    # 回读一次,确认新文件真的能加载且内容对得上
    t = time.time()
    with open(tmp, "rb") as f:
        d2 = pickle.load(f)
    print("回读 %.1fs memories=%d key_format=%s"
          % (time.time() - t, len(d2["memory"]), d2.get("key_format")))
    assert len(d2["memory"]) == n and len(d2["keys_idx"]) == n
    assert all(np.array_equal(a, b) for a, b in
               zip(d2["keys_idx"][:500], keys_idx[:500]))
    print("回读校验通过。")

    if not args.apply:
        os.remove(tmp)
        print("\n--check 模式,已删除临时文件。加 --apply 才会替换原文件。")
        return 0

    keep = args.path + ".densebak-%s" % time.strftime("%Y%m%d%H%M")
    os.replace(args.path, keep)
    os.replace(tmp, args.path)
    print("\n已替换。旧稠密版留在 %s" % keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
