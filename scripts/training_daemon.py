"""
持续训练 daemon - 周期重训 fasttext 意图分类器。

每个 cycle:
  1. 从 PG 抽 web_corpus 主题作为新增 __label__qa 样本(用 jieba by-word 分词)
  2. 与原有 corpus/classify/data_by_word_train.txt 合并
  3. 用 fasttext.train_supervised 重新训练
  4. 测试集准确率,记录到 training_log
  5. 如果准确率不下降,覆盖 classify/models/classify_by_word_100_20_2.model
  6. sleep 然后下一轮

gRPC 服务通过 ClassifyReloader 监听模型文件 mtime 自动热重载。
"""
import argparse
import os
import sys
import time
import shutil
import io
from pathlib import Path

# Force UTF-8 stdout (Windows GBK terminal can't render ✓ etc)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# fasttext C++ 内核不能处理 unicode 路径(项目目录有"备份"中文)
# chdir + 相对路径绕过
os.chdir(ROOT)

from utils.db_config import get_pg_conn
from utils import cut
from utils.log_setup import setup_daemon_logger

LOG = setup_daemon_logger("training")
LOG.info("[training] daemon import phase started")

import fasttext
LOG.info("[training] fasttext imported")

ORIG_TRAIN_PATH = "corpus/classify/data_train.txt"
ORIG_TEST_PATH = "corpus/classify/data_test.txt"
# 注意: classify/classify.py 运行时 load 的是 classify_100_20_2.model(jieba 词级)
MODEL_PATH = "classify/models/classify_100_20_2.model"
AUG_TRAIN_PATH = "corpus/classify/data_train_augmented.txt"


import random

# fasttext 默认 thread=CPU 数(本机 64)— 5 个服务进程一起跑会互踩。
# 110k 样本 8 线程足够 ~20s 一轮。
FASTTEXT_THREADS = 8
# qa 增强样本总量不超过 chat 样本的 max_ratio 倍 — 防类别失衡。
# 失衡的危害:整体 acc 虚高,但 chat 召回悄悄退化(qa 把决策面挤过去)。
QA_MAX_RATIO = 2.0


def export_qa_from_pg():
    """把 PG 已学到的 web/sentence corpus topic 生成 __label__qa 样本行(内存)。"""
    conn = get_pg_conn()
    lines = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT topic FROM web_corpus WHERE status='learned' AND topic IS NOT NULL "
            "UNION "
            "SELECT topic FROM sentence_corpus WHERE status='learned' AND topic IS NOT NULL"
        )
        for (topic,) in cur.fetchall():
            t = (topic or "").strip()
            if len(t) < 2:
                continue
            tokens = cut(t)  # 默认 jieba 词级,匹配运行时
            if not tokens:
                continue
            joined = " ".join(tokens)
            lines.append(f"{joined}\t__label__qa\n")
            # 增加 query 变体提升泛化
            lines.append(f"{joined} 是 什么\t__label__qa\n")
            lines.append(f"什么 是 {joined}\t__label__qa\n")
    conn.close()
    return lines


def build_augmented_train():
    """合并原 train + PG 导出的 qa 样本(带类别平衡 cap)。
    返回 (实际并入的 qa 增强行数, 原始 chat 行数, 原始 qa 行数)。
    """
    # 1. 统计原始训练集 label 分布
    n_chat = n_qa_orig = 0
    with open(ORIG_TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            if "__label__qa" in line:
                n_qa_orig += 1
            else:
                n_chat += 1
    # 2. PG qa 行 + cap 采样(可复现:seed 固定到行数,保证同数据同结果)
    qa_lines = export_qa_from_pg()
    cap = max(0, int(n_chat * QA_MAX_RATIO) - n_qa_orig)
    if len(qa_lines) > cap:
        rng = random.Random(len(qa_lines))   # 数据变了才换子集
        qa_lines = rng.sample(qa_lines, cap)
    # 3. 合并写盘:原训练集在前(保留 chat 样本),PG 增强放最后
    with open(AUG_TRAIN_PATH, "w", encoding="utf-8") as out:
        with open(ORIG_TRAIN_PATH, encoding="utf-8") as f:
            shutil.copyfileobj(f, out)
        out.writelines(qa_lines)
    return len(qa_lines), n_chat, n_qa_orig


def _per_label_stats(model, test_path):
    """手工算 per-label P/R。
    不用 model.test_label():这版 fasttext 的 recall 恒为 nan,
    会把验收门 min_recall 永远卡死(nan>=0.90 恒 False,模型永不落盘)。
    """
    texts, golds = [], []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            gold = next((t for t in toks if t.startswith("__label__")), None)
            if gold is None:
                continue
            text = " ".join(t for t in toks if not t.startswith("__label__"))
            texts.append(text)
            golds.append(gold)
    preds, _ = model.predict(texts)
    tp, fp, fn = {}, {}, {}
    for gold, pred in zip(golds, preds):
        p = pred[0] if pred else ""
        if p == gold:
            tp[gold] = tp.get(gold, 0) + 1
        else:
            fp[p] = fp.get(p, 0) + 1
            fn[gold] = fn.get(gold, 0) + 1
    out = {}
    for label in set(tp) | set(fp) | set(fn):
        t = tp.get(label, 0)
        prec = t / (t + fp.get(label, 0)) if (t + fp.get(label, 0)) else 0.0
        rec = t / (t + fn.get(label, 0)) if (t + fn.get(label, 0)) else 0.0
        out[label.replace("__label__", "")] = {
            "p": round(prec, 4), "r": round(rec, 4),
        }
    return out


def train_and_eval(dim=100, epoch=20, wordNgrams=2):
    """train + 在原测试集上算整体准确率和 per-label P/R。
    per-label 是关键:类别失衡时整体 acc 会掩盖 chat 召回的退化。
    """
    model = fasttext.train_supervised(
        AUG_TRAIN_PATH, dim=dim, epoch=epoch, wordNgrams=wordNgrams,
        thread=FASTTEXT_THREADS, verbose=0,
    )
    n, p_at_1, _ = model.test(ORIG_TEST_PATH)
    try:
        per_label = _per_label_stats(model, ORIG_TEST_PATH)
    except Exception:
        per_label = {}
    return model, p_at_1, n, per_label


def log_to_pg(task, note, loss=None, epoch=None):
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO training_log (task, note, loss, epoch) "
                "VALUES (%s, %s, %s, %s)",
                (task, note, loss, epoch),
            )
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=300.0,
                    help="seconds between retrains (default 5min)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--accept_threshold", type=float, default=0.985,
                    help="只有准确率 >= 此值才覆盖在线模型")
    args = ap.parse_args()

    LOG.info(f"[training] starting. interval={args.interval}s "
             f"epoch={args.epoch} accept>={args.accept_threshold}")
    print(f"[training] starting. interval={args.interval}s "
          f"epoch={args.epoch} accept>={args.accept_threshold}")

    cycle = 0
    while True:
        cycle += 1
        t0 = time.time()
        LOG.info(f"[training] === cycle {cycle} ===")
        try:
            n_new, n_chat, n_qa_orig = build_augmented_train()
            LOG.info(f"  train mix: chat={n_chat} qa_orig={n_qa_orig} "
                     f"qa_pg={n_new} (cap=chat*{QA_MAX_RATIO})")
            model, acc, n_test, per_label = train_and_eval(epoch=args.epoch)
            elapsed = time.time() - t0
            LOG.info(f"  trained in {elapsed:.1f}s. test_acc@1={acc:.4f} on {n_test} "
                     f"samples, per-label={per_label}")

            # 双重门槛:整体 acc + 每个 label 的召回都不能塌
            min_recall = min((v["r"] for v in per_label.values()), default=1.0)
            if acc >= args.accept_threshold and min_recall >= 0.90:
                model.save_model(MODEL_PATH)
                note = (f"saved model: acc={acc:.4f} per_label={per_label} "
                        f"samples={n_test} pg_qa={n_new}")
                LOG.info(f"  OK {note}")
                log_to_pg("retrain_classify", note, loss=1.0 - acc, epoch=cycle)
            else:
                note = (f"acc {acc:.4f} (need>={args.accept_threshold}) "
                        f"min_recall {min_recall:.4f} (need>=0.90), NOT saved")
                LOG.info(f"  SKIP {note}")
                log_to_pg("retrain_classify_reject", note, loss=1.0 - acc, epoch=cycle)
        except Exception as e:
            import traceback
            LOG.error(f"[training] error: {e}\n{traceback.format_exc()}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[training] interrupted.")
