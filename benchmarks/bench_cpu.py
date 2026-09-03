# -*- coding: utf-8 -*-
"""把认知大脑的每个单线程热点摊开测:各阶段耗时 + 多核扩展性 + 稀疏化收益。
只读,不动线上服务。"""
import os, sys, time, gc, pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
os.environ["BRAIN_SIMS_THREADS"] = "1"      # 先关掉,自己扫
from brain.cognitive_brain import CognitiveBrain

NCPU = os.cpu_count()
print("CPU 逻辑核 =", NCPU)
PATH = "brain/models/brain_v2.pkl"
print("pkl 大小 = %.2f GB" % (os.path.getsize(PATH) / 1e9))

t = time.time(); brain = CognitiveBrain(); t_load = time.time() - t
print("[阶段1] pickle.load          %.1fs" % t_load)

t = time.time(); K = brain._key_stack(); t_stack = time.time() - t
print("[阶段2] _key_stack (np.stack) %.1fs   稠密 %.2f GB" % (t_stack, K.nbytes / 1e9))

t = time.time(); csr = sp.csr_matrix(K); t_csr = time.time() - t
print("[阶段3] csr_matrix(dense)     %.1fs   稀疏 %.0f MB  nnz=%d  密度=%.3f%%"
      % (t_csr, (csr.data.nbytes + csr.indices.nbytes) / 1e6, csr.nnz,
         100.0 * csr.nnz / (K.shape[0] * K.shape[1])))

t = time.time(); brain._ensure_bigram_index(); t_bi = time.time() - t
print("[阶段4] _ensure_bigram_index  %.1fs" % t_bi)
print("        冷启/每次重载合计 ≈ %.0fs" % (t_load + t_stack + t_csr + t_bi))

# ---------- A. sims 的多核扩展性 ----------
import random
random.seed(5)
topics = [m.get("topic") or "" for m in brain.memory if (m.get("topic") or "").strip()]
keys = [brain.encode(t_ + "是什么") for t_ in random.sample(topics, 16)]
ref = csr.dot(keys[0])

def blocks_for(nt):
    n = csr.shape[0]
    step = max(1, (n + nt - 1) // nt)
    d, ind, ip = csr.data, csr.indices, csr.indptr
    out = []
    for s in range(0, n, step):
        e = min(s + step, n)
        d0, d1 = int(ip[s]), int(ip[e])
        out.append((s, e, sp.csr_matrix((d[d0:d1], ind[d0:d1], ip[s:e + 1] - d0),
                                        shape=(e - s, csr.shape[1]), copy=False)))
    return out

print("\n[A] 全表 sims 的多核扩展性(单次查询)")
print("  %-8s %-10s %-8s" % ("线程", "耗时", "加速比"))
base = None
for nt in (1, 2, 4, 8, 16, 24, 32, 48, 64):
    if nt > NCPU:
        break
    bl = blocks_for(nt)
    pool = ThreadPoolExecutor(max_workers=nt) if nt > 1 else None
    out = np.empty(csr.shape[0], dtype=np.float32)
    def run(b):
        s, e, m = b
        out[s:e] = m.dot(keys[0])
    def once(k):
        if pool is None:
            return csr.dot(k)
        def r(b):
            s, e, m = b
            out[s:e] = m.dot(k)
        list(pool.map(r, bl))
        return out
    once(keys[0])
    ts = []
    for i in range(6):
        t0 = time.perf_counter(); r = once(keys[i % len(keys)])
        ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts)) * 1000
    if base is None:
        base = ms
    ok = np.array_equal(ref, once(keys[0]))
    print("  %-8d %-10.1fms %-8.1fx  equal=%s" % (nt, ms, base / ms, ok))
    if pool:
        pool.shutdown()

# ---------- B. 稀疏化存储能省多少 ----------
print("\n[B] key 改稀疏存储的收益(现在每条 memory 存 16128 维稠密 float32)")
nnz_row = np.diff(csr.indptr)
print("  每条 nnz: 中位=%d 最大=%d 最小=%d" % (int(np.median(nnz_row)),
                                              int(nnz_row.max()), int(nnz_row.min())))
dense_b = K.shape[0] * K.shape[1] * 4
sparse_b = csr.nnz * 2          # uint16 列号足够(dim=16128 < 65536)
print("  稠密 %.2f GB  →  稀疏(uint16 列号) %.0f MB   省 %.0fx"
      % (dense_b / 1e9, sparse_b / 1e6, dense_b / sparse_b))

# 实测:只 pickle 稀疏 key 的体积与写盘耗时
sub = min(4000, len(brain.memory))
t = time.time()
blob = pickle.dumps([csr.indices[csr.indptr[i]:csr.indptr[i + 1]].astype(np.uint16)
                     for i in range(sub)], protocol=pickle.HIGHEST_PROTOCOL)
t_pk = time.time() - t
print("  抽样 %d 条稀疏 key: %.1f MB / %.2fs → 全量外推 %.0f MB"
      % (sub, len(blob) / 1e6, t_pk, len(blob) / 1e6 * len(brain.memory) / sub))

# ---------- C. CSR 构建能不能并行 ----------
print("\n[C] csr_matrix(dense) 分块并行(看 scipy 转换放不放 GIL)")
for nt in (1, 8, 16):
    if nt > NCPU:
        break
    n = K.shape[0]
    step = (n + nt - 1) // nt
    rng = [(s, min(s + step, n)) for s in range(0, n, step)]
    pool = ThreadPoolExecutor(max_workers=nt)
    t0 = time.perf_counter()
    parts = list(pool.map(lambda r: sp.csr_matrix(K[r[0]:r[1]]), rng))
    el = time.perf_counter() - t0
    pool.shutdown()
    print("  %-3d 线程  %.1fs" % (nt, el))
    del parts
    gc.collect()
