"""
gRPC brain service - 纯认知大脑路径(已移除 Seq2Seq Transformer 闲聊兜底)

请求处理流程:
  1. CognitiveBrain.answer 双脑检索(brain.answer)
  2. 命中(sim>=0.35) -> 返回检索答案(右脑加权采样,文字多变)
  3. 低置信联想(0.15..0.35) -> "我猜..." 的 brain_creative 回答
  4. classify=QA -> recall + dnnsort 走原 IT 问答库辅助路径
  5. 全无结果 -> 礼貌 fallback,引导用户教学
  6. 每轮对话异步写 PG,daemon 巩固入脑
"""
import os
import random
import re
import time
import threading
import torch
import torch.nn.functional as F
from concurrent import futures

# dnnsort 模型很小,64 核机器上 torch 默认线程数只会带来调度开销
torch.set_num_threads(4)

import grpc
import config
from brain.cognitive_brain import CognitiveBrain
from classify.classify import Classify
from recall.recall import Recall
from dnnsort.dnnsort import DnnSort
from utils import cut
from utils.db_config import get_pg_conn

from chatbot_grpc import chatbot_pb2, chatbot_pb2_grpc

QA_THRESHOLD = 0.7
BRAIN_SIM_THRESHOLD = 0.35       # 主路径阈值
BRAIN_CREATIVE_MIN = 0.15        # 创造性联想最低 sim
BRAIN_CREATIVE_HEDGE = 0.30      # < 此值时加"我猜..."前缀
FALLBACK_ANSWER = "这个我还没学到 — 你可以教我一句,或者问点别的(比如「光速」「量子纠缠」)。"
MAX_RESPONSE_PREVIEW = 400
# 思维逻辑路的回答是"拆解步骤 + 每步工具 + 坑 + 验收",天然比百科词条长,
# 400 字会把工具计划整段砍掉,单独给一个上限。
MAX_THINKING_PREVIEW = 1800
THINKING_ROUTE = os.environ.get("THINKING_ROUTE", "1") != "0"

# ---------------- chitchat 短路(防止短句 SDR 碰撞导致答非所问) ----------------
# 短文本(招呼/告别/感谢/确认)语义稀薄,encode 后在稀疏向量空间里
# 容易与任意 topic 高 cosine 命中,brain.answer 会返回风马牛不相及的百科。
# 这类输入在最前层直接走模板,不进 brain、不写 PG、不污染巩固。
_CHITCHAT_PATTERNS = {
    "greet": [
        "嗨", "嗨嗨", "hi", "hello", "hey", "哈喽", "哈啰", "你好", "您好",
        "在吗", "在不", "在么", "在不在", "有人吗",
        "早", "早安", "早上好", "中午好", "下午好", "晚上好",
    ],
    "bye": [
        "拜拜", "拜", "再见", "回头见", "回见", "晚安", "bye", "byebye",
        "goodbye", "good night",
    ],
    "thanks": [
        "谢谢", "谢了", "多谢", "感谢", "thanks", "thank you", "thx", "3q",
    ],
    "ack": [
        "好的", "好", "嗯", "嗯嗯", "ok", "okay", "收到", "了解", "明白", "知道了",
    ],
    "who": [
        "你是谁", "你叫什么", "你叫啥", "你是什么", "你是啥", "介绍一下你",
        "自我介绍", "你能做什么", "你会做什么", "你能干什么",
    ],
}
_CHITCHAT_REPLIES = {
    "greet": [
        "你好呀~想聊点什么?可以问我「光速」「量子纠缠」之类的,也可以教我一个新词条。",
        "嗨!我在。问点知识问题、或者教我新东西都行。",
        "你好。我是个还在学习的认知大脑,问我点什么吧。",
    ],
    "bye": [
        "拜拜~下次见。",
        "再见,记得回来教我点新东西。",
        "晚安,慢走。",
    ],
    "thanks": [
        "不客气,能帮上就好。",
        "客气啦,继续问。",
        "举手之劳,还想了解点什么?",
    ],
    "ack": [
        "嗯,继续。",
        "好的,下一个?",
        "OK,有别的想问的吗?",
    ],
    "who": [
        "我是一个认知大脑系统:从网页/百科里学习,用稀疏向量做联想检索,会被你的反馈调整可信度。你可以问我学过的东西,或者教我新词条。",
        "我是个在线学习型的对话大脑 — 没有用大模型生成,而是基于检索+联想。问知识问题最在行。",
    ],
}
# 整句精确匹配(去标点空白后)— 避免"你好厉害"被当成招呼
_PUNCT_RE = re.compile(r"[\s,。.!?!?~、:;\"'`()【】《》<>]+")


def _chitchat_match(msg):
    """返回 (category, normalized_msg) 或 (None, None)"""
    if not msg:
        return None, None
    norm = _PUNCT_RE.sub("", msg).lower().strip()
    if not norm or len(norm) > 12:  # 超过 12 字符的不可能是纯招呼
        return None, None
    for cat, words in _CHITCHAT_PATTERNS.items():
        if norm in words:
            return cat, norm
    return None, None


def _chitchat_response(msg):
    cat, _ = _chitchat_match(msg)
    if cat is None:
        return None
    return random.choice(_CHITCHAT_REPLIES[cat])


class DnnSortRanker:
    def __init__(self):
        self.model = DnnSort().to(config.device)
        self.model.load_state_dict(
            torch.load("./dnnsort/models/model.pkl", map_location=config.device)
        )
        self.model.eval()

    def rank(self, query, candidates):
        if not candidates:
            return None, 0.0
        q_tok = cut(query, by_word=True)
        c_toks = [cut(c, by_word=True) for c in candidates]
        input1 = torch.LongTensor(
            [config.dnn_ws.transform(q_tok, max_len=config.dnnsort_max_len)] * len(candidates)
        ).to(config.device)
        input2 = torch.LongTensor(
            [config.dnn_ws.transform(t, max_len=config.dnnsort_max_len) for t in c_toks]
        ).to(config.device)
        with torch.no_grad():
            prob = F.softmax(self.model(input1, input2), dim=-1)[:, 1]
        scores = prob.cpu().tolist()
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return candidates[best_idx], scores[best_idx]


CLASSIFY_MODEL_PATH = "classify/models/classify_100_20_2.model"


class IntentClassifier:
    def __init__(self):
        self.clf = Classify()
        self._lock = threading.Lock()
        self._mtime = self._mt()
        self._reload_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._reload_thread.start()

    def _mt(self):
        try:
            return os.path.getmtime(CLASSIFY_MODEL_PATH)
        except OSError:
            return 0.0

    def _watch_loop(self, interval=30.0):
        while True:
            time.sleep(interval)
            mt = self._mt()
            if mt > self._mtime + 1.0:
                try:
                    new_clf = Classify()
                    with self._lock:
                        self.clf = new_clf
                    self._mtime = mt
                    print(f"[classify] hot-reloaded.")
                except Exception as e:
                    print(f"[classify] reload error: {e}")

    def is_qa(self, sentence):
        cuted = " ".join(cut(sentence))
        with self._lock:
            label = self.clf.predict(cuted)
        return label == "__label__qa"


class DialogueLogger:
    """异步把对话写到 PG sentence_corpus(status=pending),供后台审核 / 巩固"""
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def log(self, prompt, response, source="dialog"):
        with self._lock:
            self._queue.append((prompt, response, source))

    def _loop(self):
        while not self._stop:
            try:
                self._flush()
            except Exception as e:
                print(f"[dialogue_logger] flush error: {e}")
            time.sleep(2.0)

    def _flush(self):
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()
        try:
            conn = get_pg_conn()
        except Exception as e:
            # 数据库挂了不要崩,塞回去等下次
            with self._lock:
                self._queue[:0] = batch
            print(f"[dialogue_logger] PG unavailable: {e}")
            return
        try:
            with conn:
                with conn.cursor() as cur:
                    for prompt, response, src in batch:
                        cur.execute(
                            "INSERT INTO sentence_corpus (topic, content, status, credibility) "
                            "VALUES (%s, %s, 'pending', 0.5) "
                            "ON CONFLICT (topic) DO UPDATE SET "
                            "  credibility = LEAST(1.0, sentence_corpus.credibility + 0.05)",
                            (prompt[:255], response),
                        )
        finally:
            conn.close()


class BrainReloader:
    """监听 brain.pkl mtime,变更则热重载 brain.memory。

    设计要点:
    - 间隔默认 300s(5min)。30s 太勤,consolidator 每 30s 写一次 brain.pkl,
      reload 每 30s 都触发,_key_matrix(62k × 14k = 3.5GB)反复 invalidate 重建,
      retrieve 撞上重建 → 卡死 gRPC ThreadPool。
    - 重建时持 brain._lock,与 retrieve / answer 互斥。
    - 在锁内**预先**把 _key_matrix 栈起来,避免下次 retrieve 触发重建占用 worker。
    """
    def __init__(self, brain, interval=300.0):
        self.brain = brain
        self.interval = interval
        self._stop = False
        self._mtime = self._current_mtime()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _current_mtime(self):
        try:
            return os.path.getmtime(self.brain.path)
        except OSError:
            return 0.0

    def _loop(self):
        while not self._stop:
            time.sleep(self.interval)
            mt = self._current_mtime()
            if mt <= self._mtime + 1.0:
                continue
            # 先试增量 delta:consolidator 每轮只促入 13 条左右,为这点东西
            # 重新加载整个 5.1GB 大脑(66s)就是「每 12 分钟卡一次」的根源。
            try:
                applied, n_new = self.brain.apply_delta()
                if applied:
                    t0 = time.time()
                    self.brain._ensure_csr_blocks()
                    self.brain._ensure_bigram_index()
                    self._mtime = mt
                    print(f"[brain] delta 追加 {n_new} 条,memories="
                          f"{len(self.brain.memory)} ({time.time() - t0:.1f}s)")
                    continue
            except Exception as e:
                print(f"[brain] delta 应用失败,退回全量重载: {e}")
            # 重建 new_brain 在锁外做(I/O 不阻塞 retrieve),
            # 只有最后的状态替换 + key matrix 预建在锁内做。
            try:
                new_brain = CognitiveBrain(path=self.brain.path)
                # 预先把 key matrix / CSR 镜像 / 倒排索引都建好,锁内只做指针交换
                _ = new_brain._key_stack()
                new_brain._ensure_csr_blocks()
                new_brain._ensure_bigram_index()
            except Exception as e:
                print(f"[brain] reload build error: {e}")
                continue
            try:
                with self.brain._lock:
                    self.brain.memory = new_brain.memory
                    self.brain.W1 = new_brain.W1
                    self.brain.perc_sdr = new_brain.perc_sdr
                    # 维度可能被 consolidator 扩过(grow_neurons)
                    self.brain.assoc_dim = new_brain.assoc_dim
                    self.brain.assoc_topk = new_brain.assoc_topk
                    # key 缓冲全家桶:漏拷会导致老 buf 前段 key 与新 memory 失配
                    for attr in ("_key_buf", "_key_norm_buf", "_key_filled",
                                 "_key_matrix", "_key_norms",
                                 "_cred_buf", "_hem_buf",
                                 "_csr", "_csr_version", "_keys_version",
                                 "_csr_blocks", "_csr_blocks_version",
                                 "_bigram_idx", "_bigram_filled",
                                 "_topic_bi", "_topic_index",
                                 "_short_topic_idx"):
                        if hasattr(new_brain, attr):
                            setattr(self.brain, attr, getattr(new_brain, attr))
                    # 整表换脑,旧 prompt 的 cos 缓存全部作废
                    self.brain._cos_cache = {}
                    # feature_index 也要换(回填特征用得着)
                    if hasattr(new_brain, "feature_index"):
                        self.brain.feature_index = new_brain.feature_index
                self._mtime = mt
                print(f"[brain] hot-reloaded. memories={len(self.brain.memory)}")
            except Exception as e:
                print(f"[brain] reload swap error: {e}")


class ChatServicer(chatbot_pb2_grpc.ChatBotServiceServicer):
    def __init__(self):
        print("[init] cognitive brain...")
        self.brain = CognitiveBrain()
        print(f"  brain: {self.brain.stats()}")
        # 预建 _key_matrix + _key_norms,避免首次 chat 触发 ~6s 冷启动卡顿
        print("[init] prebuild key matrix...")
        t0 = time.time()
        self.brain._key_stack()
        print(f"  key matrix ready: shape={self.brain._key_matrix.shape} "
              f"({time.time() - t0:.1f}s)")
        # CSR 镜像 + 行分块 + topic 倒排也预建:否则第一次 chat 要现场
        # 建 543MB 稀疏矩阵(~17s),表现为"重启后第一句话卡死"。
        t0 = time.time()
        self.brain._ensure_csr_blocks()
        self.brain._ensure_bigram_index()
        print(f"  csr blocks + bigram index ready ({time.time() - t0:.1f}s)")
        print("[init] classify...")
        self.intent = IntentClassifier()
        print("[init] recall (附加 IT 知识库)...")
        self.recall = Recall(method="tfidf")
        print("[init] dnnsort...")
        self.ranker = DnnSortRanker()
        print("[init] 思维逻辑索引(BM25,独立于 SDR 大脑)...")
        self.thinking = None
        if THINKING_ROUTE:
            try:
                from brain.thinking_index import ThinkingIndex
                self.thinking = ThinkingIndex(conn_factory=get_pg_conn)
                print(f"  thinking corpus: {len(self.thinking.rows)} 条")
            except Exception as e:
                print(f"[thinking] 索引未启用(不影响主服务): {e}")
        print("[init] dialogue logger...")
        self.logger = DialogueLogger()
        print("[init] brain hot-reloader...")
        self.reloader = BrainReloader(self.brain)
        self._save_counter = 0
        print("[ready] 纯认知路径,Seq2Seq 已移除")

    def _maybe_save(self):
        """**gRPC 不再写 brain_v2.pkl**(2026-08-26 关掉)。三条理由:

        1. **和 consolidator 抢同一个文件**。`save()` 用固定的
           `brain_v2.pkl.tmp`,两个进程同时写就会把 5.1GB 的正式文件替换成
           交错的垃圾 —— 实测撞到过 `invalid load key '\\x00'`,只能靠 .bak 救。
        2. **本来就是白做功**。它存的只是命中后的 cred/hits 微调,而
           BrainReloader 每次热重载会用磁盘上 consolidator 的版本整个覆盖掉。
        3. 每 100 次请求写 5.1GB,纯属拿磁盘带宽换空气。

        另外它自己也一直在报 `dictionary changed size during iteration`
        —— pickle 遍历 perc_sdr 的同时 encode() 在往里加新字符。

        这份 pkl 的所有者是 consolidator_daemon,让它一家写。
        """
        return

    def _creative_answer(self, msg):
        """大脑低置信联想 — 替代以前的 Seq2Seq 闲聊兜底。
        sim 在 [BRAIN_CREATIVE_MIN, BRAIN_SIM_THRESHOLD) 之间也敢答,
        但加 hedge 前缀("我猜...")提示用户这是模糊匹配。
        """
        from brain.cognitive_brain import (
            _substantive_bigrams, _substantive_overlap)
        results = self.brain.retrieve(msg, top_k=5)
        q_bi = _substantive_bigrams(msg)
        if q_bi:
            # 放宽为实义 bigram 交集(严格 overlap 会把"万有引力"vs"引力场"
            # 这类近邻词条全滤掉);SDR 池漏召时再补词法召回兜底。
            def _rel(t):
                return bool(q_bi & _substantive_bigrams(t or ""))
            results = [r for r in results if _rel(r[0].get("topic", ""))]
            if not results:
                results = [r for r in self.brain._topic_bigram_recall(msg, top_k=5)
                           if _rel(r[0].get("topic", ""))]
        if not results:
            return None, {}
        # 用加权采样选(让回答多变)
        import numpy as np
        cand = [(m, s, c) for m, s, c in results if s >= BRAIN_CREATIVE_MIN]
        if not cand:
            return None, {}
        weights = np.array([s * c for _, s, c in cand[:3]], dtype=np.float64)
        if weights.sum() <= 0:
            return None, {}
        weights = weights / weights.sum()
        pick = int(np.random.choice(len(cand[:3]), p=weights))
        m, sim, cred = cand[pick]
        # 诚实 hedge:SDR 在大库上普遍饱和(sim 常年 >0.35),按 sim 分级
        # 会让模糊猜测显得过于自信;改成看 topic 是否真跟问题严格相关。
        prefix = ""
        if not _substantive_overlap(msg, m.get("topic") or ""):
            prefix = f"我猜你想问的是「{m['topic']}」?\n\n"
        elif sim < BRAIN_SIM_THRESHOLD:
            prefix = "可能跟这个有关:\n"
        ans = prefix + (m["content"] or "")[:300]
        # 记录命中以触发 RL 微涨
        m["hits"] = m.get("hits", 0) + 1
        m["last_seen"] = time.time()
        return ans, {
            "topic": m["topic"], "source": m["source"],
            "similarity": sim, "credibility": cred,
            "creative": True, "hemisphere": "creative",
        }

    def _route(self, msg):
        # 0. chitchat 短路:招呼/告别/感谢等语义稀薄短句直接走模板。
        #    短文本 SDR 必然碰撞,放进 brain.answer 会得到答非所问的高 sim 误命中。
        chit = _chitchat_response(msg)
        if chit is not None:
            return chit, "chitchat", {"category": _chitchat_match(msg)[0]}

        # 0.5 思维逻辑路:任务类提问答"怎么拆 + 每步用什么工具",不是百科词条。
        #     两道门都要过:looks_like_task(够长或带任务动词)+ BM25 相对分阈值。
        #     百科短问题("量子纠缠是什么")第一道门就被挡掉,不影响主路径。
        if self.thinking is not None:
            try:
                ans, dbg = self.thinking.answer(msg)
                if ans:
                    return ans[:MAX_THINKING_PREVIEW], "thinking", dbg
            except Exception as e:
                print(f"[thinking] error: {e}")

        # 1. 主路径:双脑认知检索
        ans, dbg = self.brain.answer(msg, sim_threshold=BRAIN_SIM_THRESHOLD)
        if ans:
            preview = ans[:MAX_RESPONSE_PREVIEW]
            hem = dbg.get("hemisphere", "?")
            sampled = dbg.get("sampled", False)
            return preview, f"brain[{hem}{'~' if sampled else ''}]", dbg

        # 2. 附加路径:经典 QA 召回(只在意图分类判为 qa 时)
        if self.intent.is_qa(msg):
            try:
                from brain.cognitive_brain import _char_bigrams, _substantive_overlap
                recalled = self.recall.predict(msg)
                # 同款实义 2-gram gate:候选 question 必须和 query 有实义字符重叠,
                # 否则 dnnsort 即便给 0.74 也是伪命中。
                if _char_bigrams(msg):
                    recalled = [c for c in recalled if _substantive_overlap(msg, c)]
                best, score = self.ranker.rank(msg, recalled)
                if best is not None and score > QA_THRESHOLD:
                    return self.recall.qa_dict[best]["ans"], "qa_recall", {"score": score}
            except Exception as e:
                print(f"[qa_recall] error: {e}")

        # 3. 创造性联想兜底:brain 低置信检索 (替代 Seq2Seq)
        ans, dbg = self._creative_answer(msg)
        if ans:
            return ans[:MAX_RESPONSE_PREVIEW], "brain_creative", dbg

        # 4. 真没结果 — 引导用户教学
        return FALLBACK_ANSWER, "fallback_unknown", {}

    def Chatbot(self, request, context):
        msg = request.user_message
        try:
            response, route, dbg = self._route(msg)
        except Exception as e:
            response, route, dbg = FALLBACK_ANSWER, "error", {"err": str(e)}

        # 在线学习: 把对话喂给大脑 + 写 PG
        # 只学高置信主路径命中;creative/qa_recall 是低置信猜测,把自己的
        # 回答再写回语料会形成错误自我强化闭环(答错一次→consolidator
        # 提升进记忆→以后 sim=1.0 精确命中,永远答错)。
        try:
            if route.startswith("brain["):
                self.brain.learn_dialogue(msg, response, source="live", credibility=0.7)
                self.logger.log(msg, response, source=route)
        except Exception as e:
            print(f"[learn] error: {e}")

        self._maybe_save()
        print(f"[{route}] Q: {msg[:30]} | A: {response[:30]} | dbg={dbg}")
        return chatbot_pb2.ResponsedMessage(
            user_response=response,
            create_time=int(time.time()),
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_pb2_grpc.add_ChatBotServiceServicer_to_server(ChatServicer(), server)
    server.add_insecure_port("[::]:9999")
    server.start()
    print("gRPC ChatBotService listening on :9999")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()

