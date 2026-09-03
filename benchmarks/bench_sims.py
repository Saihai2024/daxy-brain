"""对照实验:全表 sims 的三种算法,用生产同规格(N=71870, dim=16128, topk=945)。
   key 是严格二值 kWTA,所以三者结果必须逐元素相等。"""
import os, sys, time
import numpy as np
from scipy import sparse as sp

N, DIM, TOPK = 71870, 16128, 945
rng = np.random.default_rng(0)

t0 = time.perf_counter()
# 直接造 CSR,不落 4.6GB 稠密
indptr = np.arange(N + 1, dtype=np.int64) * TOPK
indices = np.empty(N * TOPK, dtype=np.int32)
for i in range(N):
    indices[i * TOPK:(i + 1) * TOPK] = np.sort(
        rng.choice(DIM, size=TOPK, replace=False).astype(np.int32))
data = np.ones(N * TOPK, dtype=np.float32)
K = sp.csr_matrix((data, indices, indptr), shape=(N, DIM))
print("build csr %.1fs nnz=%d" % (time.perf_counter() - t0, K.nnz))

# 查询 key:同样 945 个激活位
act = np.sort(rng.choice(DIM, size=TOPK, replace=False)).astype(np.int32)
q = np.zeros(DIM, dtype=np.float32)
q[act] = 1.0

def timeit(fn, n=5):
    fn()  # warm
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return min(ts) * 1000, np.median(ts) * 1000, r

# A. 生产现状:CSR . 稠密向量
a_min, a_med, ra = timeit(lambda: K.dot(q))
print("A csr.dot(dense)      min=%.1fms med=%.1fms" % (a_min, a_med))

# B. 稠密 matmul(小库时的退路)—— 跳过,4.6GB 不值得测

# C. 倒排:转置 CSC 的列求和
t0 = time.perf_counter()
KT = K.T.tocsr()          # [DIM, N]
print("build KT %.1fs" % (time.perf_counter() - t0))
c_min, c_med, rc = timeit(lambda: np.asarray(KT[act].sum(axis=0)).ravel())
print("C KT[act].sum(axis=0) min=%.1fms med=%.1fms  equal=%s"
      % (c_min, c_med, np.array_equal(ra, rc.astype(np.float32))))

# D. 倒排 + bincount(纯 numpy,不碰 scipy 开销)
rows = KT.indices        # int32
rp = KT.indptr
def d():
    segs = [rows[rp[d0]:rp[d0 + 1]] for d0 in act]
    cat = np.concatenate(segs)
    return np.bincount(cat, minlength=N).astype(np.float32)
d_min, d_med, rd = timeit(d)
print("D bincount(inverted)  min=%.1fms med=%.1fms  equal=%s"
      % (d_min, d_med, np.array_equal(ra, rd)))

print("KT bytes: indices=%.0fMB indptr=%.0fMB | CSR bytes=%.0fMB"
      % (rows.nbytes / 1e6, rp.nbytes / 1e6, (K.data.nbytes + K.indices.nbytes) / 1e6))
