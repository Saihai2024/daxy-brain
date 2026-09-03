# -*- coding: utf-8 -*-
"""真实 brain 上的 sims 算法对照。合成数据骗过我一次(以为激活列均匀分布),
   这次全部用生产 key 矩阵 + 真实 prompt。"""
import os, sys, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
from scipy import sparse as sp
from brain.cognitive_brain import CognitiveBrain

print("numpy", np.__version__, "| has bitwise_count:", hasattr(np, "bitwise_count"))
t0 = time.time(); brain = CognitiveBrain(); brain._key_stack()
print("load+stack %.1fs N=%d dim=%d topk=%d"
      % (time.time() - t0, len(brain.memory), brain.assoc_dim, brain.assoc_topk))
K = brain._key_matrix
N, DIM = K.shape

random.seed(3)
topics = [m.get("topic") or "" for m in brain.memory if (m.get("topic") or "").strip()]
prompts = [t if i % 2 else t + "是什么" for i, t in enumerate(random.sample(topics, 24))]
keys = [brain.encode(p) for p in prompts]

# 先看激活列的真实分布 —— 这是上一版失败的根因
csr = sp.csr_matrix(K)
colpop = np.asarray(csr.sum(axis=0)).ravel()
act0 = np.flatnonzero(keys[0])
print("列热度: 全局中位=%d 均值=%.0f | query激活列 中位=%d 均值=%.0f 最大=%d"
      % (np.median(colpop), colpop.mean(),
         np.median(colpop[act0]), colpop[act0].mean(), colpop[act0].max()))

def timeit(fn, n=6):
    fn(keys[0])
    ts = []
    for i in range(n):
        t = time.perf_counter(); fn(keys[i % len(keys)]); ts.append(time.perf_counter() - t)
    return np.median(ts) * 1000

ref = [csr.dot(k) for k in keys]
print("A csr.dot(dense)            %7.1fms" % timeit(lambda k: csr.dot(k)))

# ---- B: 位图 + popcount ----
t0 = time.time()
bits = np.packbits((K > 0).astype(np.uint8), axis=1)      # [N, ceil(DIM/8)]
BW = bits.shape[1]
print("build bitset %.1fs (%.0fMB)" % (time.time() - t0, bits.nbytes / 1e6))
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def sims_bits(key, chunk=8192):
    q = np.packbits((key > 0).astype(np.uint8))
    out = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out[s:e] = LUT[bits[s:e] & q].sum(axis=1, dtype=np.int32)
    return out
b_ms = timeit(sims_bits)
ok_b = all(np.array_equal(ref[i], sims_bits(keys[i])) for i in range(len(keys)))
print("B bitset+LUT popcount       %7.1fms  equal=%s" % (b_ms, ok_b))

if hasattr(np, "bitwise_count"):
    u64 = bits[:, :BW // 8 * 8].view(np.uint64) if BW % 8 == 0 else None
    if u64 is not None:
        def sims_bc(key, chunk=8192):
            q = np.packbits((key > 0).astype(np.uint8)).view(np.uint64)
            out = np.empty(N, dtype=np.float32)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                out[s:e] = np.bitwise_count(u64[s:e] & q).sum(axis=1, dtype=np.int32)
            return out
        c_ms = timeit(sims_bc)
        ok_c = all(np.array_equal(ref[i], sims_bc(keys[i])) for i in range(len(keys)))
        print("B2 bitset+np.bitwise_count  %7.1fms  equal=%s" % (c_ms, ok_c))

# ---- C: CSR 按行分块 + 线程池(scipy sparsetools 会不会放 GIL) ----
from concurrent.futures import ThreadPoolExecutor
for nt in (4, 8, 16):
    blocks = []
    step = (N + nt - 1) // nt
    for s in range(0, N, step):
        blocks.append((s, min(s + step, N), csr[s:min(s + step, N)]))
    pool = ThreadPoolExecutor(max_workers=nt)
    def sims_par(key):
        out = np.empty(N, dtype=np.float32)
        def run(b):
            s, e, m = b
            out[s:e] = m.dot(key)
        list(pool.map(run, blocks))
        return out
    ms = timeit(sims_par)
    ok = np.array_equal(ref[0], sims_par(keys[0]))
    print("C csr 分块 %2d 线程          %7.1fms  equal=%s" % (nt, ms, ok))
    pool.shutdown()
