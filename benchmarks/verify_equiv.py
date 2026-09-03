# -*- coding: utf-8 -*-
"""等价性验证:新版检索热路径 vs 旧算法,必须逐元素/逐条相同。

单进程只加载一次 brain(4.9GB),旧算法在本脚本里就地复刻,
避免起两份 11GB 进程。
"""
import os, sys, time, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
from scipy import sparse as sp
from brain import cognitive_brain as cb
from brain.cognitive_brain import (
    CognitiveBrain, _substantive_bigrams, _char_bigrams,
    RETRIEVE_TOPK, RERANK_POOL, RERANK_LEX_BOOST)

t0 = time.time()
brain = CognitiveBrain()
print("load brain %.1fs memories=%d dim=%d topk=%d"
      % (time.time() - t0, len(brain.memory), brain.assoc_dim, brain.assoc_topk))

t0 = time.time(); brain._key_stack(); print("key_stack %.1fs" % (time.time() - t0))
t0 = time.time(); csr_old = sp.csr_matrix(brain._key_matrix)
t_csr_old = time.time() - t0
print("build OLD csr(单线程) %.1fs (%.0fMB)"
      % (t_csr_old, (csr_old.data.nbytes + csr_old.indices.nbytes) / 1e6))
t0 = time.time(); csr_par = cb._csr_from_dense_parallel(brain._key_matrix)
t_par = time.time() - t0
same_csr = (np.array_equal(csr_par.indptr, csr_old.indptr)
            and np.array_equal(csr_par.indices, csr_old.indices)
            and np.array_equal(csr_par.data, csr_old.data))
print("build NEW csr(并行 %d 线程) %.1fs  与单线程逐元素相同=%s  (加速 %.1fx)"
      % (cb.CSR_BUILD_THREADS, t_par, same_csr, t_csr_old / max(t_par, 1e-9)))
del csr_par
t0 = time.time(); blocks = brain._ensure_csr_blocks()
print("build csr blocks %.1fs (%d 块, sims 线程池 %d)"
      % (time.time() - t0, len(blocks), cb.SIMS_THREADS))
t0 = time.time(); brain._ensure_bigram_index(); print("bigram idx %.1fs" % (time.time() - t0))

# ---------- 取真实 prompt ----------
random.seed(7)
topics = [m.get("topic") or "" for m in brain.memory if (m.get("topic") or "").strip()]
sample = random.sample(topics, 120)
TPL = ["%s", "%s是什么", "什么是%s", "介绍一下%s", "%s怎么样", "%s有什么用"]
prompts = [TPL[i % len(TPL)] % s for i, s in enumerate(sample)]
prompts += ["你好", "python好学吗", "asdkjhaskdjh", "你叫什么名字",
            "帮我把一个 FastAPI 服务的响应时间从 300ms 降到 100ms,应该先做什么",
            "熵", "什么是熵", "光速是多少", "量子纠缠是什么"]
print("prompts=%d" % len(prompts))

# ---------- 1. sims 逐元素相等 ----------
bad = 0
t_old = t_new = 0.0
for p in prompts:
    key = brain.encode(p)
    t = time.perf_counter(); s_old = csr_old.dot(key); t_old += time.perf_counter() - t
    t = time.perf_counter(); s_new = brain._sims_all(key); t_new += time.perf_counter() - t
    if not np.array_equal(s_old, s_new):
        bad += 1
        if bad <= 3:
            d = np.flatnonzero(s_old != s_new)[:5]
            print("  MISMATCH %r idx=%s old=%s new=%s" % (p, d, s_old[d], s_new[d]))
print("[1] sims 等价: %d/%d 相同 | old %.1fms/次  new %.1fms/次  加速 %.1fx"
      % (len(prompts) - bad, len(prompts),
         t_old / len(prompts) * 1000, t_new / len(prompts) * 1000,
         t_old / max(t_new, 1e-9)))

# ---------- 2. topic bigram 缓存等价 ----------
bad2 = 0
for i in random.sample(range(len(brain.memory)), 3000):
    if brain._topic_bigrams_at(i) != _substantive_bigrams(brain.memory[i].get("topic") or ""):
        bad2 += 1
print("[2] topic bigram 缓存等价: %d/3000 相同" % (3000 - bad2))

# ---------- 3. topic 倒排等价(learn_dialogue 用) ----------
bad3 = 0
for t_ in random.sample(topics, 500):
    idxs = brain._topic_index.get(t_) or []
    ref = [i for i, m in enumerate(brain.memory) if m["topic"] == t_]
    if sorted(idxs) != ref:
        bad3 += 1
print("[3] topic 倒排等价: %d/500 相同" % (500 - bad3))

# ---------- 4. retrieve 结果完全一致 ----------
def retrieve_old(brain, prompt, top_k=RETRIEVE_TOPK, hemisphere=None):
    """旧 _retrieve_locked 原样复刻(唯一差别:sims 走旧 CSR)。"""
    if not brain.memory:
        return []
    key = brain.encode(prompt)
    brain._key_stack()
    norms = brain._key_norms_stack()
    n = len(brain.memory)
    sims = csr_old.dot(key)
    norm_q = np.linalg.norm(key) + 1e-8
    cos = sims / (norms * norm_q)
    creds = brain._cred_buf[:n]
    score = cos * creds
    if hemisphere == "left":
        score = np.where(brain._hem_buf[:n] == 0, score, -1.0)
    elif hemisphere == "right":
        score = np.where(brain._hem_buf[:n] == 1, score, -1.0)
    pool = min(max(top_k * 10, RERANK_POOL), n)
    top_idx = np.argpartition(-score, pool - 1)[:pool]
    q_bi = _substantive_bigrams(prompt)
    cand = {int(i) for i in top_idx}
    if q_bi:
        cand.update(brain._lexical_candidates(prompt, cap=2000))
    rescored = []
    for i in cand:
        if score[i] < 0:
            continue
        m = brain.memory[i]
        lex = 0.0
        if q_bi:
            t_bi = _substantive_bigrams(m.get("topic", ""))
            if t_bi:
                inter = len(q_bi & t_bi)
                if inter:
                    lex = inter / len(q_bi | t_bi)
        final = score[i] * (1.0 + RERANK_LEX_BOOST * lex)
        rescored.append((final, i))
    rescored.sort(key=lambda x: -x[0])
    return [(brain.memory[i], float(cos[i]), float(creds[i]))
            for _, i in rescored[:top_k]]

bad4 = 0
for p in prompts:
    for hem in (None, "left", "right"):
        a = retrieve_old(brain, p, hemisphere=hem)
        b = brain.retrieve(p, hemisphere=hem)
        ka = [(id(m), round(s, 6), round(c, 6)) for m, s, c in a]
        kb = [(id(m), round(s, 6), round(c, 6)) for m, s, c in b]
        if ka != kb:
            bad4 += 1
            if bad4 <= 3:
                print("  RETRIEVE DIFF %r hem=%s" % (p, hem))
                print("    old:", [(m["topic"], round(s, 4)) for m, s, _ in a])
                print("    new:", [(m["topic"], round(s, 4)) for m, s, _ in b])
tot4 = len(prompts) * 3
print("[4] retrieve 等价: %d/%d 相同" % (tot4 - bad4, tot4))

# ---------- 5. answer 端到端(关采样,保证确定性) ----------
bad5 = 0
diffs = []
for p in prompts:
    a, da = brain.answer(p, sampling=False)
    b, db = brain.answer(p, sampling=False)
    if a != b:
        bad5 += 1
        diffs.append(p)
print("[5] answer 自身可重复(sampling=False): %d/%d 稳定" % (len(prompts) - bad5, len(prompts)))

# ---------- 6. answer() 端到端耗时(每条都是新 prompt,不吃跨请求缓存) ----------
import types
def _answer_timed(n=40):
    t = time.perf_counter()
    for p in prompts[:n]:
        brain._cos_cache = {}          # 模拟"每次都是没见过的新问题"
        brain.answer(p, sampling=False)
    return (time.perf_counter() - t) / n * 1000

new_ms = _answer_timed()
# 关掉 cos 复用 + 关掉并行,还原改造前的行为
orig_cos, orig_sims = cb.CognitiveBrain._cos_all, cb.CognitiveBrain._sims_all
def _cos_nocache(self, prompt):
    key = self.encode(prompt)
    norms = self._key_norms_stack()
    sims = csr_old.dot(key)
    return sims / (norms * (np.linalg.norm(key) + 1e-8))
cb.CognitiveBrain._cos_all = _cos_nocache
old_ms = _answer_timed()
cb.CognitiveBrain._cos_all = orig_cos
print("[6] answer() 端到端: 改造前 %.1fms/次 → 改造后 %.1fms/次 (%.1fx)"
      % (old_ms, new_ms, old_ms / max(new_ms, 1e-9)))

print("\n==== 结论: %s ====" % ("全部等价 ✅" if (bad + bad2 + bad3 + bad4) == 0
                              else "有差异 ❌ 不要上线"))
