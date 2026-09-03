# -*- coding: utf-8 -*-
"""验证增量 delta 链路:写 → 读 → 追加后与"全量重载"完全等价。

用临时小脑跑,不碰生产的 brain_v2.pkl,也不吃机器资源。
要证的三件事:
  1. save() 会把「上次保存之后新增的部分」写成 .delta
  2. apply_delta() 追加后,记忆条数/内容/key 与源大脑逐条相同
  3. 追加后的 retrieve() 结果与"重新全量加载一遍"完全一致
     —— 这是关键:delta 不能是"差不多",必须等价
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.cognitive_brain import CognitiveBrain

TOPICS_A = ["量子纠缠", "光速", "机器学习", "熵", "相对论", "神经网络",
            "傅里叶变换", "图灵机", "热力学第二定律", "贝叶斯定理"]
TOPICS_B = ["卷积神经网络", "支持向量机", "马尔可夫链", "梯度下降",
            "反向传播", "注意力机制"]


def build(path, topics):
    b = CognitiveBrain(path=path)
    for i, t in enumerate(topics):
        b.add_memory(t, "关于%s的解释内容第%d条" % (t, i),
                     source="test", credibility=0.9, dedup_threshold=None)
    return b


def main():
    d = tempfile.mkdtemp(prefix="brain_delta_")
    path = os.path.join(d, "b.pkl")
    ok = True
    try:
        # 1) 建源大脑并保存(首次保存不产 delta)
        A = build(path, TOPICS_A)
        A.save()
        assert not os.path.exists(path + ".delta"), "首次保存不该产 delta"
        print("[1] 首次保存 %d 条,未产 delta ✓" % len(A.memory))

        # 2) 读者 B 从同一个 pkl 加载
        B = CognitiveBrain(path=path)
        print("[2] 读者加载 %d 条 ✓" % len(B.memory))
        assert len(B.memory) == len(TOPICS_A)

        # 3) 源大脑新增若干条并保存 → 应产出 delta
        for i, t in enumerate(TOPICS_B):
            A.add_memory(t, "关于%s的解释内容第%d条" % (t, i),
                         source="test", credibility=0.8, dedup_threshold=None)
        A.save()
        assert os.path.exists(path + ".delta"), "新增后应产出 delta"
        print("[3] 新增 %d 条并保存,delta 已产出 (%.0f KB) ✓"
              % (len(TOPICS_B), os.path.getsize(path + ".delta") / 1024))

        # 4) 读者吃 delta
        applied, n_new = B.apply_delta()
        print("[4] apply_delta -> applied=%s n_new=%d" % (applied, n_new))
        assert applied and n_new == len(TOPICS_B), "delta 没被正确应用"
        assert len(B.memory) == len(A.memory)

        # 5) 逐条比对 topic / content / key
        bad = 0
        for i, (ma, mb) in enumerate(zip(A.memory, B.memory)):
            if ma["topic"] != mb["topic"] or ma["content"] != mb["content"]:
                bad += 1
            elif not np.array_equal(np.asarray(ma["key"]), np.asarray(mb["key"])):
                bad += 1
        print("[5] 逐条比对 topic/content/key: %d/%d 相同 %s"
              % (len(A.memory) - bad, len(A.memory), "✓" if not bad else "★"))
        ok = ok and not bad

        # 6) 与"全量重载"对拍:同一批 query 的 retrieve 必须完全一致
        C = CognitiveBrain(path=path)          # 全量重载
        qs = [t for t in TOPICS_A + TOPICS_B] + ["什么是熵", "机器学习是什么"]
        diff = 0
        for q in qs:
            rb = [(m["topic"], round(s, 6)) for m, s, _ in B.retrieve(q)]
            rc = [(m["topic"], round(s, 6)) for m, s, _ in C.retrieve(q)]
            if rb != rc:
                diff += 1
                if diff <= 3:
                    print("   ★ %r\n     delta路: %s\n     全量路: %s" % (q, rb, rc))
        print("[6] retrieve 与全量重载对拍: %d/%d 完全一致 %s"
              % (len(qs) - diff, len(qs), "✓" if not diff else "★"))
        ok = ok and not diff

        # 7) 对不上号的 delta 必须被拒绝(防止错位追加)
        applied2, _ = B.apply_delta()
        print("[7] 重复吃同一个 delta -> applied=%s(应为 False) %s"
              % (applied2, "✓" if not applied2 else "★"))
        ok = ok and not applied2

        print("\n==== %s ====" % ("delta 链路等价 ✅" if ok else "有差异 ❌"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
