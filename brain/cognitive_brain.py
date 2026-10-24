"""
认知大脑 v2 - 类脑稀疏关联检索网络

设计目标:
  1. 模拟人类思考: 联想 + 记忆检索(海马体式情景记忆)
  2. 持续学习: 每轮对话立即写入记忆,无需 epoch
  3. CPU 飞快: 纯 numpy,2048 神经元,推理 < 5ms
  4. 区分真实数据置信度: 每条记忆带 credibility(来源加权)
  5. 多源融合: 接 PG 永久存储(sentence/web/vision/对话历史)

架构(2048 神经元):
  Perception (512)  -> Association (1024)  -> Concept (512)
  - Perception: 字 -> SDR (5% 稀疏)
  - Association: Hebbian 编码器(query 编码为稀疏激活模式 = 检索 key)
  - Concept: 实体/主题标签 SDR(reserved,后续用于跨概念联想)

记忆机制:
  - 短期工作记忆: 最近 N 字的上下文(秒级)
  - 情景记忆(memory bank): list of {key, content, source, credibility, hits, ...}
  - 长期记忆: W1 权重矩阵本身(slow consolidation)

学习模式:
  - 一次性灌库: bootstrap_from_pg() - 用 sentence_corpus + web_corpus 训 W1
  - 在线学习: learn_dialogue() - 新对话立即加 memory + Hebbian 巩固
  - 自然遗忘: 长期不命中的记忆 credibility 缓慢衰减

检索:
  - encode(prompt) -> 1024-d 稀疏激活
  - 与所有记忆 key 求 cosine,取 top-K,credibility 加权
"""
import os
import re
import time
import pickle
import hashlib
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    from scipy import sparse as _sp
except ImportError:
    _sp = None

# 命名实体启发式: 2+ 中文字 / Capitalized 英文
_re_entity = re.compile(r"[一-鿿]{2,}|[A-Z][a-zA-Z]{1,}")

PERC_DIM = 512
ASSOC_DIM = 1024
CONCEPT_DIM = 512  # reserved
ACTIVE_BITS = 25
CONTEXT_LEN = 16
CONTEXT_DECAY = 0.85
ASSOC_TOPK = 60
# ===== 编码器 v2:unigram + 实义 bigram 双通道 =====
# v1 只有字符袋(unigram+位置衰减),没有词序信息,导致不相关短语
# cosine 虚高(0.98+),答非所问只能靠 gate 硬挡。
# v2 加 bigram 通道:实义 bigram(剔除 _STOP_BIGRAMS)各自有独立 SDR,
# "量子纠缠" 和 "你会什么" 的编码从根上拉开。
ENCODER_VERSION = 2
BIGRAM_ALPHA = 0.45   # 合成权重:ctx = α·norm(unigram) + (1-α)·norm(bigram)
LEARN_RATE = 0.05
WEIGHT_DECAY = 1e-7   # 几乎不忘,corpus 是底子
MAX_WEIGHT = 1.0
RETRIEVE_TOPK = 5
# 全表 sims 的行分块并行度。scipy csr_matvec 放 GIL,线程池能吃满多核。
# 0 / 1 = 关闭并行(退回单线程 csr.dot)。
# 32 是这台 64 核机器上实测的最优点(N=71874, dim=16128, topk=960):
#   1线程 106.5ms / 4线程 25.7 / 16线程 15.0 / **32线程 11.0(9.7x)** / 48线程 13.4 / 64线程 19.3
# 超过 32 就因为超订 + 内存带宽饱和而退化,别往大了调。
SIMS_THREADS = int(os.environ.get("BRAIN_SIMS_THREADS", "32"))
# 稠密→CSR 的分块并行度(scipy 转换同样放 GIL):实测 1线程 18.6s → 16线程 1.7s
CSR_BUILD_THREADS = int(os.environ.get("BRAIN_CSR_BUILD_THREADS", "16"))
_SIMS_POOL = (ThreadPoolExecutor(max_workers=SIMS_THREADS,
                                 thread_name_prefix="brain-sims")
              if SIMS_THREADS > 1 else None)
# 全表 cos 缓存条数。一条 = N 个 float32(7 万条记忆约 287KB),
# 64 条约 18MB。主要是给「同一次请求里多条检索路复用」用,顺带命中重复提问。
COS_CACHE_SIZE = int(os.environ.get("BRAIN_COS_CACHE", "64"))
RERANK_POOL = 50          # SDR 召回池大小(两段式检索的第一段)
RERANK_LEX_BOOST = 1.5    # lexical(实义 bigram Jaccard)对最终分的加成系数
BRAIN_CREATIVE_MIN = 0.15      # 兜底召回时接受的最低 sim(对应 grpc 同名常量)
CRED_DECAY_PER_DAY = 0.001  # 闲置记忆每天衰减 0.1% 可信度

# ===== 双脑分工(基于内容,不是来源)=====
# 左脑:分析型 - 事实/科学/定义/公式
LEFT_KEYWORDS = (
    "定理", "定律", "方程", "常数", "公式", "函数", "矩阵", "粒子", "守恒",
    "物理学", "数学", "化学", "天文学", "工程", "算法", "理论", "悖论",
    "测量", "计算", "假说", "实验", "科学", "原理", "证明", "推导",
    "kg", "m/s", "J·s", "Hz", "Pa", "MHz", "GHz", "DNA", "RNA",
    "基因", "蛋白质", "细胞", "分子", "原子", "电子", "光子", "夸克",
)
# 右脑:联想型 - 人物/作品/叙事/情感
RIGHT_KEYWORDS = (
    "电影", "电视剧", "小说", "漫画", "动画", "歌曲", "专辑", "诗",
    "导演", "演员", "主演", "编剧", "歌手", "诗人", "作家", "画家",
    "讲述", "出生于", "诞生于", "成立于", "发表于", "出演",
    "故事", "情节", "剧情", "回忆", "传记", "传记片",
    "哲学家", "思想家", "心理学", "情感", "感人", "梦想",
    "爱情", "亲情", "友情", "失恋", "孤独", "悲伤", "幸福",
)
# 来源默认半球(无关键词时兜底)
SOURCE_DEFAULT_HEMISPHERE = {
    "vision_corpus": "right",
    "live": "right",
    "brain": "right",
    "qa_recall": "right",
    "chat_seq2seq": "right",
    "web_corpus": "left",
    "sentence_corpus": "left",
}


# ===== 自主特征抽取 =====
# jieba POS 标签:
#   n: 名词 / nr: 人名 / ns: 地名 / nt: 机构 / nz: 其他专名
#   v: 动词 / a: 形容词 / m: 数量 / t: 时间
_POS_ENTITY = {"nr", "nrt", "nrf", "ns", "nsf", "nt", "nz"}
_POS_NOUN = {"n", "nr", "nrt", "nrf", "ns", "nsf", "nt", "nz", "ng"}
_POS_VERB = {"v", "vd", "vn", "vi", "vshi", "vyou"}
_POS_ADJ = {"a", "ad", "an"}
_POS_NUM = {"m", "mq"}

# 极常见词不算特征(noise)
_FEATURE_STOPS = {
    "中", "上", "下", "里", "时", "前", "后", "之间", "一起", "一个", "一种",
    "什么", "怎么", "如何", "可能", "也许", "一些", "这种", "那种",
    "the", "and", "of", "to", "in", "is", "a",
}


def _extract_features(topic, content, max_per_class=15):
    """从 (topic, content) 抽各类特征。返回 dict:
       {entities, nouns, verbs, adjs, numbers, length, char_len}
    """
    text = ((topic or "") + " " + (content or "")[:600]).strip()
    if not text:
        return {"entities": [], "nouns": [], "verbs": [], "adjs": [], "numbers": [],
                "length": "empty", "char_len": 0}
    try:
        import jieba.posseg as psg
        import logging
        import jieba
        jieba.setLogLevel(logging.ERROR)
        pos_pairs = list(psg.cut(text))
    except Exception:
        return {"entities": [], "nouns": [], "verbs": [], "adjs": [], "numbers": [],
                "length": "unknown", "char_len": len(text)}

    entities, nouns, verbs, adjs, numbers = [], [], [], [], []
    seen = set()

    def _push(bucket, w, cap):
        if w in _FEATURE_STOPS or w in seen or len(w) < 2:
            return
        if len(bucket) >= cap:
            return
        bucket.append(w)
        seen.add(w)

    for w, flag in pos_pairs:
        if flag in _POS_ENTITY: _push(entities, w, max_per_class)
        elif flag in _POS_NOUN: _push(nouns, w, max_per_class)
        elif flag in _POS_VERB: _push(verbs, w, max_per_class)
        elif flag in _POS_ADJ:  _push(adjs, w, max_per_class)
        elif flag in _POS_NUM:  _push(numbers, w, max_per_class)

    n = len(text)
    length = "short" if n < 60 else "medium" if n < 300 else "long" if n < 1000 else "xlong"
    return {
        "entities": entities,
        "nouns": nouns,
        "verbs": verbs,
        "adjs": adjs,
        "numbers": numbers,
        "length": length,
        "char_len": n,
    }


_GATE_STRIP_RE = re.compile(r"[\s,。.!?!?~、:;\"'`()【】《》<>]+")

# gate 不算数的 2-gram:疑问/虚词/客套,常出现在 query 和 topic 两侧,
# 仅靠它们重叠不构成"实质相关",必须有其它实词 bigram 才算命中。
_STOP_BIGRAMS = frozenset({
    # 疑问/指代
    "什么", "怎么", "么样", "是什", "为什", "为何", "如何", "哪里", "哪个",
    "那个", "这个", "那些", "这些", "多少", "几个", "哪些", "怎样", "啥子",
    # 系动/连词/语气
    "么是", "是指", "指的", "的是", "也叫", "叫做", "称为", "所谓",
    "是不", "不是", "是吗", "是的", "是啊", "没有", "是否", "可以", "能不",
    "可不", "不可", "不能", "不会", "可能", "应该", "需要", "由于", "关于",
    "对于", "属于", "就是", "还是", "还有", "还能", "已经", "曾经", "也是",
    "都是", "一下", "一个", "一种", "一些", "一样", "一直", "或者", "或是",
    # 介绍/客套(常出现在 query)
    "介绍", "请问", "想问", "告诉", "知道",
    # 时间/方位的弱信号
    "今天", "明天", "昨天", "现在", "以前", "之后", "之前", "里面", "外面",
})


def _key_to_idx(key):
    """稠密二值 key → 激活列号(uint16)。assoc_dim 远小于 65536,够用。"""
    if key is None:
        return np.zeros(0, dtype=np.uint16)
    return np.flatnonzero(key).astype(np.uint16)


def _csr_from_dense_parallel(dense, threads=None):
    """稠密 → CSR,按行分块用线程池并行。

    scipy 的稠密转稀疏**会释放 GIL**,所以纯 Python 线程池就能吃满多核。
    生产规格(71874×16128, 4.79GB)实测:单线程 18.6s → 8 线程 2.4s → 16 线程 1.7s。
    这一步在每次 brain 热重载时都要做一遍,是 79s 重载里的第三大头。
    """
    if _sp is None:
        return None
    n = dense.shape[0]
    nt = max(1, min(threads or CSR_BUILD_THREADS, n))
    if nt <= 1 or n < 4096:
        return _sp.csr_matrix(dense)
    step = (n + nt - 1) // nt
    rng = [(s, min(s + step, n)) for s in range(0, n, step)]
    with ThreadPoolExecutor(max_workers=len(rng)) as pool:
        parts = list(pool.map(lambda r: _sp.csr_matrix(dense[r[0]:r[1]]), rng))
    return _sp.vstack(parts, format="csr")


def _char_bigrams(text):
    """提取字符 2-gram 集合,用于 brain 检索的实义重叠门控。
    不依赖 jieba — 对短文本、口语、混排、乱码都稳。
    去除空白/标点后,把每相邻两字符拼成 bigram。
    单字符 query 返回单字符集合;空串返回 set()。
    """
    if not text:
        return set()
    t = _GATE_STRIP_RE.sub("", text).lower()
    if not t:
        return set()
    if len(t) == 1:
        return {t}
    return {t[i:i + 2] for i in range(len(t) - 1)}


# 问题模板前缀/后缀:剥掉后剩下的是 query 的"实义核心"。
# 例:"原子是什么" → "原子";"介绍一下相对论" → "相对论"。
# 注意按长度降序,优先剥长模板。
_QUESTION_PREFIXES = sorted([
    "介绍一下", "介绍下", "介绍一个", "什么叫做", "什么是", "什么叫", "什么样的",
    "如何理解", "如何", "怎么理解", "怎么样的", "怎么", "为什么", "为何", "为啥",
    "请问", "想问", "想知道", "告诉我", "告诉", "解释一下", "解释下", "解释",
    "说一下", "说下", "说说", "聊一下", "聊聊", "讲一下", "讲讲", "讲下", "讲一讲",
    "给我讲讲", "给我说说", "给我聊聊", "给我讲", "给我说", "帮我介绍", "帮我看看",
    "哪个是", "哪里有", "哪些是", "哪里是",
], key=len, reverse=True)
_QUESTION_SUFFIXES = sorted([
    "是什么意思", "是什么东西", "是什么样", "是什么呀", "是什么的", "是什么",
    "是啥意思", "是啥东西", "是啥", "是多少", "是怎样", "是怎么回事", "是怎么样",
    "怎么样", "怎么回事", "怎么办", "怎么做", "怎么用", "怎么读", "怎么写", "怎么",
    "在哪里", "在哪儿", "在哪",
    "有什么用", "有什么", "有哪些", "有几种", "有多少",
    "是不是", "是吗", "对吗", "是嘛",
    "的意思", "的含义", "的作用", "的特点", "的原理",
], key=len, reverse=True)


def _strip_question_template(text):
    """剥掉 query 首尾的问题模板,返回核心实义短语(可能为空)。"""
    if not text:
        return ""
    t = _GATE_STRIP_RE.sub("", text).lower()
    for pre in _QUESTION_PREFIXES:
        if t.startswith(pre) and len(t) > len(pre):
            t = t[len(pre):]
            break
    for suf in _QUESTION_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            t = t[:-len(suf)]
            break
    return t


def _has_substantive_bigram(sub):
    """sub 里至少有一个非 stopword bigram 才算"实义"片段。
    用于拦截"是什么"/"为什么"这种纯疑问短语作为字符重叠通过 gate。
    """
    if len(sub) < 2:
        return False
    for i in range(len(sub) - 1):
        if sub[i:i + 2] not in _STOP_BIGRAMS:
            return True
    return False


def _substantive_overlap(query, topic, min_sub=3):
    """query 与 topic 是否有"实质"重叠(用于 brain 检索 gate)。

    判定:
      a) topic 整体出现在 query 里(长度 ≥ 2,且 topic 含非 stopword bigram)→ True
      b) query 整体出现在 topic 里(同上)→ True
      c) topic 或 query 的某个长度 ≥ min_sub 的子串完整出现在对方里,
         且该子串至少含一个非 stopword bigram → True

    例:
      "光速是多少" vs "光速"          → topic in query, "光速"实义     ✓
      "DNA是什么"  vs "DNA"           → topic in query, "DNA"实义      ✓
      "DNA是什么"  vs "RNA"           → 无 ≥3 子串重叠                  ✗
      "爱因斯坦"   vs "维特根斯坦"    → 仅共享 2 char "斯坦"            ✗
      "光速是多少" vs "徐速是什么"    → 无 ≥3 子串重叠                  ✗
      "量子纠缠是什么" vs "量子数是什么" → 共享 "是什么" 但全是 stopword ✗
      "你叫什么名字" vs "你叫什么名字" → 完整匹配,"你叫"实义           ✓
    """
    if not query or not topic:
        return False
    t = _GATE_STRIP_RE.sub("", topic).lower()
    q = _GATE_STRIP_RE.sub("", query).lower()
    if not t or not q:
        return False

    # 最强信号:剥掉问题模板后的 query 核心 in topic 核心(或反之)。
    # 例:"原子是什么" 核心"原子" → 拒 topic"关子是什么"(核心"关子"),
    #     但通过 topic"原子物理"(含"原子")。
    q_core = _strip_question_template(query)
    t_core = _strip_question_template(topic)
    # 单字核心("什么是熵" → "熵"):出现在短 topic 核心里就算相关。
    # 汉语单字概念(熵/光/力)不可能有 bigram,常规路径必漏。
    if q_core and len(q_core) == 1 and t_core and len(t_core) <= 6 \
       and q_core in t_core:
        return True
    if q_core and t_core and len(q_core) >= 2 and len(t_core) >= 2:
        if q_core in t_core or t_core in q_core:
            if _has_substantive_bigram(q_core if q_core in t_core else t_core):
                return True

    if len(t) >= 2 and t in q and _has_substantive_bigram(t):
        return True
    if len(q) >= 2 and q in t and _has_substantive_bigram(q):
        return True
    if min(len(t), len(q)) < min_sub:
        return False
    # 双向 ≥min_sub 子串 + 实义 bigram。topic 长子串优先(信号更强)。
    # 但子串必须在 query 核心 或 topic 核心 里,否则就是问题模板偶合。
    for size in range(min(len(t), 8), min_sub - 1, -1):
        for i in range(len(t) - size + 1):
            sub = t[i:i + size]
            if sub in q and _has_substantive_bigram(sub):
                # 子串必须有一端进入"核心"(非问题模板),否则视为模板偶合
                if (q_core and sub in q_core) or (t_core and sub in t_core):
                    return True
    for size in range(min(len(q), 8), min_sub - 1, -1):
        for i in range(len(q) - size + 1):
            sub = q[i:i + size]
            if sub in t and _has_substantive_bigram(sub):
                if (q_core and sub in q_core) or (t_core and sub in t_core):
                    return True
    return False


def _substantive_bigrams(text):
    """返回 query 的实义 2-gram(剔除 stopword bigram) — 兜底召回用。"""
    return _char_bigrams(text) - _STOP_BIGRAMS


def _query_significant_tokens(text):
    """(legacy)兼容旧调用:返回 query 的实义 token 集合。
    新 gate 已切到 _char_bigrams;此函数保留供其它路径继续用。
    """
    f = _extract_features(text, "")
    sig = set(f.get("entities") or []) | set(f.get("nouns") or []) | set(f.get("numbers") or [])
    return sig or None


def _memory_significant_tokens(features):
    if not features:
        return set()
    return (set(features.get("entities") or [])
            | set(features.get("nouns") or [])
            | set(features.get("numbers") or []))


def _content_hemisphere(topic, content, source=None):
    """根据内容特征决定半球。返回 'left' / 'right'。
    优先级:
      1. 命中右脑关键词(叙事/人物/作品)→ right
      2. 命中左脑关键词(科学/数学/逻辑)→ left
      3. source 默认值
    """
    text = (topic or "") + " " + (content or "")[:600]
    right_score = sum(1 for kw in RIGHT_KEYWORDS if kw in text)
    left_score = sum(1 for kw in LEFT_KEYWORDS if kw in text)
    if right_score > left_score:
        return "right"
    if left_score > right_score:
        return "left"
    return SOURCE_DEFAULT_HEMISPHERE.get(source, "left")


# === legacy(保留兼容旧 stats 输出)===
LEFT_SOURCES = {"web_corpus", "sentence_corpus"}
RIGHT_SOURCES = {"vision_corpus", "live", "brain", "qa_recall", "chat_seq2seq"}

# 路由启发式词集
QUESTION_WORDS = ("什么", "为什么", "如何", "怎么", "哪", "谁", "几", "是不是",
                  "what", "how", "why", "who", "which", "when", "where")
EMOTION_WORDS = ("爱", "恨", "想念", "难过", "开心", "怕", "孤独", "无聊",
                 "讨厌", "喜欢", "烦", "累", "饿", "困")
GREETING_WORDS = ("嗨", "hi", "你好", "您好", "hello", "拜拜", "再见", "晚安",
                  "早", "晚上好", "中午好", "谢谢", "感谢")


class CognitiveBrain:
    def __init__(self, path="brain/models/brain_v2.pkl"):
        self.path = path
        self.W1 = None  # perception -> association
        self.perc_sdr = {}    # char -> 512-d binary
        self.concept_sdr = {} # char -> 512-d binary (reserved)
        self.memory = []      # list of dicts
        self.steps = 0
        # 可成长的维度(初始 = 模块常量)
        self.assoc_dim = ASSOC_DIM
        self.assoc_topk = ASSOC_TOPK
        # 神经元成长事件历史
        self.growth_log = []  # [{ts, delta, new_dim, reason}]
        # RLock 保护:retrieve / answer / 兜底召回 与 hot-reload 互斥,
        # 防止 _key_matrix invalidate 撞上 numpy 大矩阵操作把 gRPC 线程池卡死。
        self._lock = threading.RLock()
        if os.path.exists(self.path):
            self._load()
        else:
            self._init_fresh()

    def _init_fresh(self):
        rng = np.random.default_rng(42)
        self.W1 = rng.uniform(0.0, 0.01, (PERC_DIM, self.assoc_dim)).astype(np.float32)
        self.encoder_version = ENCODER_VERSION

    def _invalidate_keys(self):
        """强制下次 _key_stack 全量重建(keys 整体变化时调用)"""
        self._key_matrix = None
        self._key_norms = None
        self._key_buf = None
        self._key_norm_buf = None
        self._key_filled = -1
        self._cred_buf = None
        self._hem_buf = None
        self._csr = None
        self._csr_version = -1
        self._csr_blocks = None
        self._csr_blocks_version = -1
        self._bigram_idx = None
        self._bigram_filled = -1
        self._topic_bi = None
        self._topic_index = None
        self._cos_cache = None
        self._keys_version = getattr(self, "_keys_version", 0) + 1

    def _load(self):
        try:
            with open(self.path, "rb") as f:
                d = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            # 主文件损坏,试 .bak
            bak = self.path + ".bak"
            print(f"[brain] load 主文件失败({e}),尝试 .bak ...")
            if os.path.exists(bak):
                with open(bak, "rb") as f:
                    d = pickle.load(f)
                print(f"[brain] 从 .bak 恢复成功")
            else:
                print(f"[brain] 没有 .bak 备份,全新初始化")
                self._init_fresh()
                return
        self.W1 = d["W1"]
        self.perc_sdr = d["perc_sdr"]
        self.concept_sdr = d.get("concept_sdr", {})
        self.memory = d["memory"]
        self.steps = d.get("steps", 0)
        self._pending_keys_idx = d.get("keys_idx")   # 稀疏格式:见下方 _restore_keys
        # 兼容老 pkl(没有 assoc_dim 字段)— 从 W1 形状推断
        self.assoc_dim = d.get("assoc_dim", self.W1.shape[1])
        self.assoc_topk = d.get("assoc_topk", ASSOC_TOPK)
        self.growth_log = d.get("growth_log", [])
        # 兼容老数据:回填 hemisphere + features
        backfilled = 0
        for m in self.memory:
            if "hemisphere" not in m:
                m["hemisphere"] = _content_hemisphere(
                    m.get("topic"), m.get("content"), m.get("source")
                )
            if "features" not in m:
                m["features"] = _extract_features(m.get("topic"), m.get("content"))
                backfilled += 1
        if backfilled > 0:
            print(f"[brain] 回填 {backfilled} 条记忆的特征字段")
        # 重建倒排索引
        self._rebuild_feature_index()
        # reconstruct stacked key matrix lazy(saves disk)
        self._invalidate_keys()
        # 稀疏格式:把列号展开回 key 缓冲(顺带把 _key_stack 那一步也省掉)
        self._restore_keys_from_idx()
        self._last_saved_n = len(self.memory)
        # ===== 编码器版本迁移 =====
        # pkl 里的 keys 是旧编码器生成的,与新 context_vector 失配 → 必须全量重编码。
        # encode_batch 在 64 核 BLAS 下 62k 条约 10-30s。
        # 这里只改内存不写盘(防多进程同时启动竞争写);由 scripts/migrate_encoder_v2.py
        # 或 consolidator 的下一次正常 save 落盘。
        loaded_ver = d.get("encoder_version", 1)
        if loaded_ver != ENCODER_VERSION and self.memory:
            print(f"[brain] encoder v{loaded_ver} -> v{ENCODER_VERSION},"
                  f" 全量重编码 {len(self.memory)} 条 keys ...")
            t0 = time.time()
            new_keys = self.encode_batch([m["topic"] for m in self.memory])
            for i, m in enumerate(self.memory):
                m["key"] = new_keys[i]
            print(f"[brain] 重编码完成 ({time.time() - t0:.1f}s)")
        self.encoder_version = ENCODER_VERSION

    def _restore_keys_from_idx(self):
        """稀疏格式(keys_idx = 每条 key 的激活列号 uint16)→ 直接铺进 key 缓冲。

        为什么值得:key 是 kWTA 二值,每条恰好 assoc_topk 个 1(实测恒定 960),
        按 16128 维稠密 float32 存 = **94% 的字节在存零**。
        稠密 4.79GB vs 稀疏列号 140MB,差 34 倍。

        这里不只是"解压",而是一步到位把 `_key_buf` 铺好、`m["key"]` 指成
        缓冲里的**行视图**,于是:
          - `_key_stack()` 的 `np.stack([m["key"] ...])`(实测 23.6s)整步跳过
          - 没有 73k 个独立小数组,内存碎片和拷贝都省了
        """
        idx_list = getattr(self, "_pending_keys_idx", None)
        self._pending_keys_idx = None
        if not idx_list:
            return False
        n = len(self.memory)
        if len(idx_list) != n:
            print(f"[brain] keys_idx 数量({len(idx_list)})与 memory({n})对不上,"
                  f"按旧格式处理")
            return False
        t0 = time.time()
        cap = max(1024, int(n * 1.25))
        buf = np.zeros((cap, self.assoc_dim), dtype=np.float32)
        norms = np.ones(cap, dtype=np.float32)
        creds = np.ones(cap, dtype=np.float32)
        hems = np.zeros(cap, dtype=np.uint8)
        for i, (m, idx) in enumerate(zip(self.memory, idx_list)):
            row = buf[i]
            row[idx] = 1.0
            m["key"] = row                     # 视图,不是副本
            m["_idx"] = i
            norms[i] = np.sqrt(len(idx)) + 1e-8   # 二值向量的模 = sqrt(nnz)
            creds[i] = m["credibility"]
            hems[i] = 1 if m.get("hemisphere", "left") == "right" else 0
        self._key_buf = buf
        self._key_norm_buf = norms
        self._cred_buf = creds
        self._hem_buf = hems
        self._key_filled = n
        self._key_matrix = buf[:n]
        self._key_norms = norms[:n]
        self._keys_version = getattr(self, "_keys_version", 0) + 1
        print(f"[brain] 稀疏 keys 展开完成 {n} 条 ({time.time() - t0:.1f}s)")
        return True

    def apply_delta(self, path=None):
        """吃 consolidator 写的增量 delta,把新记忆追加进来。

        取代「为了 13 条新记忆重新加载整个 pkl」:
          全量重载 = load 5.1GB(37.9s) + np.stack 4.79GB(23.6s) + 建 CSR ≈ 66s
          走 delta  = 读几十 MB + 往缓冲尾部写 13 行 + 重建 CSR ≈ 3s

        只在**能严格对上号**时才接受(条数连续 + 维度一致),否则返回 False
        让调用方退回全量重载。

        注意一个**有意的差异**:全量重载会用磁盘版覆盖本进程对 cred/hits 的
        在线微调,走 delta 则保留它们。这是好事(那些微调本来每次重载就丢一次)。

        返回 (是否应用, 新增条数)。
        """
        p = path or (self.path + ".delta")
        if not os.path.exists(p):
            return False, 0
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"[brain] delta 读取失败,退回全量: {e}")
            return False, 0
        n_now = len(self.memory)
        if d.get("from_n") != n_now:
            return False, 0                     # 对不上号(错过了好几轮 / 换过脑)
        if int(d.get("assoc_dim", -1)) != int(self.assoc_dim):
            return False, 0                     # grow_neurons 扩过维,必须全量
        new_mem = d.get("memory") or []
        new_idx = d.get("keys_idx") or []
        if len(new_mem) != len(new_idx) or not new_mem:
            return False, 0

        with self._lock:
            if len(self.memory) != n_now:       # 拿锁期间被别人动过
                return False, 0
            self._key_stack()                   # 确保缓冲已就绪
            need = n_now + len(new_mem)
            buf = self._key_buf
            if need > buf.shape[0]:             # 扩容 1.5×,连同平行缓冲
                new_cap = max(int(need * 1.5), buf.shape[0] * 2)
                for attr, dtype, fill in (("_key_buf", np.float32, 0.0),
                                          ("_key_norm_buf", np.float32, 1.0),
                                          ("_cred_buf", np.float32, 1.0),
                                          ("_hem_buf", np.uint8, 0)):
                    old = getattr(self, attr)
                    shape = ((new_cap, self.assoc_dim) if attr == "_key_buf"
                             else (new_cap,))
                    nb = np.full(shape, fill, dtype=dtype)
                    nb[:n_now] = old[:n_now]
                    setattr(self, attr, nb)
                buf = self._key_buf
            for j, (m, idx) in enumerate(zip(new_mem, new_idx)):
                i = n_now + j
                row = buf[i]
                row[:] = 0.0
                row[idx] = 1.0
                m["key"] = row
                m["_idx"] = i
                self._key_norm_buf[i] = np.sqrt(len(idx)) + 1e-8
                self._cred_buf[i] = m.get("credibility", 1.0)
                self._hem_buf[i] = 1 if m.get("hemisphere", "left") == "right" else 0
                self.memory.append(m)
                self._index_features(i, m.get("features") or {})
            self._key_filled = len(self.memory)
            self._key_matrix = buf[:self._key_filled]
            self._key_norms = self._key_norm_buf[:self._key_filled]
            if d.get("W1") is not None:
                self.W1 = d["W1"]               # 编码器被 hebbian 动过,跟上
            if d.get("steps") is not None:
                self.steps = d["steps"]
            self._keys_version = getattr(self, "_keys_version", 0) + 1
            self._cos_cache = {}                # 表变了,旧 cos 全作废
            self._last_saved_n = len(self.memory)
        return True, len(new_mem)

    def save(self):
        """原子写入 + 上轮快照备份。
        1) 先写 .tmp,fsync
        2) 把当前 .pkl 备份为 .bak(下轮如果 .pkl 损坏可恢复)
        3) os.replace(.tmp, .pkl)
        """
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # ⚠️ tmp 名必须带 pid:曾经是固定的 `.pkl.tmp`,gRPC 和 consolidator
        # 同时 save 时两个进程写同一个 tmp,再 os.replace 上去 =
        # 5.1GB 正式文件变成交错垃圾(实测 `invalid load key '\x00'`)。
        tmp = "%s.tmp.%d" % (self.path, os.getpid())
        bak = self.path + ".bak"
        # perc_sdr / memory 取快照再 pickle:encode() 会往 perc_sdr 里加新字符,
        # 边遍历边加会抛 `dictionary changed size during iteration`,
        # 异常一抛就留下半截 tmp。值都是共享 ndarray,浅拷贝很便宜。
        perc_snapshot = dict(self.perc_sdr)
        mem_snapshot = list(self.memory)
        # key 改稀疏落盘:只存激活列号(uint16 够用,assoc_dim < 65536)。
        # 稠密 4.79GB → 140MB。memory 里的 "key" / "_idx" 剥掉不落盘。
        keys_idx = [_key_to_idx(m.get("key")) for m in mem_snapshot]
        mem_out = [{k: v for k, v in m.items() if k not in ("key", "_idx")}
                   for m in mem_snapshot]
        payload = {
            "W1": self.W1, "perc_sdr": perc_snapshot,
            "concept_sdr": dict(self.concept_sdr),
            "memory": mem_out, "keys_idx": keys_idx,
            "key_format": "idx_u16",
            "steps": self.steps,
            "assoc_dim": self.assoc_dim,
            "assoc_topk": self.assoc_topk,
            "growth_log": self.growth_log,
            "encoder_version": getattr(self, "encoder_version", ENCODER_VERSION),
        }
        # 1) 写 tmp
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        # 2) 备份上一版为 .bak。
        # ⚠️ 原来是 `os.replace(self.path, bak)` 把正式文件**移走**,于是在第 3 步
        # 完成之前 `.pkl` 是不存在的;读者(BrainReloader / 任何脚本)正好撞进来就
        # 读不到或读半截。而且第 3 步 8 次全失败的话正式文件就直接没了。
        # 改成建**硬链接**:不拷 5GB 数据、瞬时完成,旧数据被 .bak 这个名字钉住,
        # 第 3 步的 os.replace 是原子的,读者要么看到旧的完整文件要么看到新的。
        linked = False
        if os.path.exists(self.path):
            try:
                if os.path.exists(bak):
                    os.remove(bak)
                os.link(self.path, bak)   # NTFS/ext4 都支持
                linked = True
            except OSError:
                try:                       # 不支持硬链接就退回老做法
                    if os.path.exists(bak):
                        os.remove(bak)
                    os.replace(self.path, bak)
                except OSError:
                    pass                   # 备份失败不致命
        # 3) tmp -> 正式(Windows 上若被其他进程读时占用,重试几次)
        last_err = None
        for attempt in range(8):
            try:
                os.replace(tmp, self.path)
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s, ... 渐增
        if last_err is not None:
            # 8 次都失败:正式文件仍是旧的完整版(硬链接方案下没被移走),清掉 tmp
            if linked:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise last_err
        # 3.5) 写增量 delta:让读者(gRPC 的 BrainReloader)不必为了几条新记忆
        #      去重新加载整个 pkl。consolidator 每轮只促入 13 条左右,
        #      而全量重载要 66s —— 这是「每 12 分钟卡一次」的根源。
        #      perc_sdr **故意不放进 delta**:它是 _hash_sdr 的确定性缓存,
        #      读者缺哪个字符自己算出来的结果完全一样,没必要传 177MB。
        try:
            last_n = getattr(self, "_last_saved_n", None)
            n_now = len(mem_snapshot)
            if last_n is not None and 0 <= last_n < n_now:
                dpath = self.path + ".delta"
                dtmp = "%s.tmp.%d" % (dpath, os.getpid())
                with open(dtmp, "wb") as f:
                    pickle.dump({
                        "from_n": last_n, "to_n": n_now,
                        "memory": mem_out[last_n:],
                        "keys_idx": keys_idx[last_n:],
                        "W1": self.W1,
                        "assoc_dim": self.assoc_dim,
                        "assoc_topk": self.assoc_topk,
                        "steps": self.steps,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                os.replace(dtmp, dpath)
            self._last_saved_n = n_now
        except Exception as e:
            print(f"[brain] delta 写入失败(不致命,读者会退回全量重载): {e}")

        # 4) 写轻量 stats.json,供 dashboard 快速读取(避免反复 load 1+ GB pkl)
        try:
            import json as _json
            stats_path = self.path + ".stats.json"
            tmp_stats = "%s.tmp.%d" % (stats_path, os.getpid())
            with open(tmp_stats, "w", encoding="utf-8") as f:
                _json.dump(self.stats(), f, ensure_ascii=False)
            os.replace(tmp_stats, stats_path)
        except Exception:
            pass

    @staticmethod
    def _hash_sdr(token, dim):
        h = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big") & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        idx = rng.choice(dim, size=ACTIVE_BITS, replace=False)
        sdr = np.zeros(dim, dtype=np.float32)
        sdr[idx] = 1.0
        return sdr

    def perc(self, ch):
        if ch not in self.perc_sdr:
            self.perc_sdr[ch] = self._hash_sdr("P|" + ch, PERC_DIM)
        return self.perc_sdr[ch]

    def context_vector(self, text):
        """工作记忆: 文本 -> 512-d 上下文向量(编码器 v2,双通道)。

        通道 1(unigram):字符袋 + 位置指数衰减 — 对错别字/换序鲁棒。
        通道 2(bigram):实义字符对(剔除 _STOP_BIGRAMS),每个 bigram 有
        独立 SDR — 提供词序/词组信息,把"量子纠缠"和"你会什么"从根上拉开。
        两通道各自归一化后按 BIGRAM_ALPHA 合成,再整体归一化。
        """
        chars = text[-CONTEXT_LEN:]
        # 通道 1: unigram
        uni = np.zeros(PERC_DIM, dtype=np.float32)
        for i, ch in enumerate(reversed(chars)):
            uni += (CONTEXT_DECAY ** i) * self.perc(ch)
        n_uni = np.linalg.norm(uni)
        if n_uni > 0:
            uni /= n_uni
        # 通道 2: 实义 bigram(无衰减,等权)
        bi = np.zeros(PERC_DIM, dtype=np.float32)
        n_bi_grams = 0
        for j in range(len(chars) - 1):
            bg = chars[j:j + 2]
            if bg in _STOP_BIGRAMS:
                continue
            bi += self.perc("B|" + bg)
            n_bi_grams += 1
        if n_bi_grams > 0:
            n_bi = np.linalg.norm(bi)
            if n_bi > 0:
                bi /= n_bi
            ctx = BIGRAM_ALPHA * uni + (1.0 - BIGRAM_ALPHA) * bi
        else:
            # 没有任何实义 bigram(纯虚词/单字)→ 退回 unigram 通道
            ctx = uni
        norm = np.linalg.norm(ctx)
        if norm > 0:
            ctx = ctx / norm
        return ctx

    @staticmethod
    def _kwta(x, k):
        if x.size <= k:
            return (x > 0).astype(np.float32)
        idx = np.argpartition(-x, k)[:k]
        out = np.zeros_like(x)
        out[idx] = 1.0
        return out

    def encode(self, text):
        """文本 -> assoc_dim-d 稀疏激活模式(检索 key)"""
        ctx = self.context_vector(text)
        a = ctx @ self.W1
        return self._kwta(a, self.assoc_topk)

    def encode_batch(self, texts, chunk=4096):
        """批量编码:把 ctx 堆成矩阵走一次大 matmul(BLAS 多线程),
        比逐条 encode 快 1-2 个数量级。用于全量重编码(迁移/扩神经元)。
        返回 [N, assoc_dim] float32。
        """
        if not texts:
            return np.zeros((0, self.assoc_dim), dtype=np.float32)
        k = self.assoc_topk
        out_chunks = []
        for s in range(0, len(texts), chunk):
            ctxs = np.stack([self.context_vector(t or "") for t in texts[s:s + chunk]])
            A = ctxs @ self.W1                    # [chunk, assoc_dim]
            if A.shape[1] <= k:
                keys = (A > 0).astype(np.float32)
            else:
                idx = np.argpartition(-A, k, axis=1)[:, :k]
                keys = np.zeros_like(A)
                np.put_along_axis(keys, idx, 1.0, axis=1)
            out_chunks.append(keys)
        return np.vstack(out_chunks) if len(out_chunks) > 1 else out_chunks[0]

    def grow_neurons(self, delta=256, reason="manual"):
        """新增 delta 个联想层神经元。
        - W1 加 delta 列(Xavier 风格的小随机)
        - assoc_topk 同比扩,保持稀疏比例
        - 重新编码所有 memory.key(维度变了,旧 key 失效)
        """
        if delta <= 0:
            return {"new_assoc_dim": self.assoc_dim, "added": 0}
        old_dim = self.assoc_dim
        rng = np.random.default_rng()
        new_cols = rng.uniform(0.0, 0.01, (PERC_DIM, delta)).astype(np.float32)
        self.W1 = np.hstack([self.W1, new_cols])
        self.assoc_dim = old_dim + delta
        # 保持稀疏度
        self.assoc_topk = max(1, int(round(self.assoc_topk * (self.assoc_dim / old_dim))))
        # 重编码所有 keys(批量 matmul,比逐条快 1-2 个数量级)
        if self.memory:
            new_keys = self.encode_batch([m["topic"] for m in self.memory])
            for i, m in enumerate(self.memory):
                m["key"] = new_keys[i]
        self._invalidate_keys()
        ev = {
            "ts": time.time(),
            "delta": delta,
            "old_assoc_dim": old_dim,
            "new_assoc_dim": self.assoc_dim,
            "new_topk": self.assoc_topk,
            "reason": reason,
        }
        self.growth_log.append(ev)
        return ev

    def maybe_auto_grow(self, threshold_ratio=3.0):
        """记忆数 / assoc_dim 超过阈值就自动扩 256。
        默认 3.0 = 每个联想神经元负担 3 条记忆时触发扩容。
        """
        if not self.memory:
            return None
        ratio = len(self.memory) / max(1, self.assoc_dim)
        if ratio >= threshold_ratio:
            ev = self.grow_neurons(delta=256, reason=f"auto:ratio={ratio:.2f}")
            return ev
        return None

    def _hebbian_update(self, ctx, a_assoc, lr=LEARN_RATE):
        """Hebbian: 一起激活的连接加强"""
        lr = np.float32(lr)
        decay = np.float32(1.0 - WEIGHT_DECAY)
        assoc_idx = np.where(a_assoc > 0)[0]
        if len(assoc_idx) > 0:
            self.W1[:, assoc_idx] += lr * ctx[:, None]
        self.W1 *= decay
        np.clip(self.W1, 0.0, MAX_WEIGHT, out=self.W1)
        self.steps += 1

    def add_memory(self, topic, content, source="live", credibility=1.0,
                   meta=None, dedup_threshold=0.95, consolidate=False):
        """加新记忆。
        consolidate=True 时同时做 Hebbian 巩固编码器(只在对话型在线学习时使用,
        批量灌库不该开,否则编码器会漂移坍缩)。
        dedup_threshold=None 时跳过 _best_match,适合灾后批量重灌(O(n²) → O(n))。"""
        key = self.encode(topic)
        if consolidate:
            ctx = self.context_vector(topic)
            self._hebbian_update(ctx, key)
            # 巩固后重新编码,与最新参数一致
            key = self.encode(topic)

        # 去重: 检查近似已有(dedup_threshold=None 跳过)
        if self.memory and dedup_threshold is not None:
            best_idx, best_score = self._best_match(key)
            if best_score >= dedup_threshold and self.memory[best_idx]["content"] == content:
                m = self.memory[best_idx]
                m["hits"] += 1
                m["last_seen"] = time.time()
                # 多次见到同样答案 -> 提高 credibility(上限 1.0)
                m["credibility"] = min(1.0, m["credibility"] + 0.01)
                self._sync_cred(m)
                return best_idx

        rec = {
            "topic": topic[:255],
            "content": content,
            "key": key,
            "source": source,
            "hemisphere": _content_hemisphere(topic, content, source),
            "features": _extract_features(topic, content),
            "credibility": float(credibility),
            "hits": 1,
            "created": time.time(),
            "last_seen": time.time(),
            "meta": meta or {},
        }
        self.memory.append(rec)
        idx = len(self.memory) - 1
        self._index_features(idx, rec["features"])
        # 不整表 invalidate:_key_stack 的增量逻辑会把新行追加进容量缓冲。
        # (之前每 add 一条就重建 3.5GB 矩阵,批量巩固时是最大 CPU 黑洞)
        return idx

    def _index_features(self, idx, features):
        """把这条 memory 的所有特征加到倒排索引。"""
        if not hasattr(self, "feature_index") or self.feature_index is None:
            self.feature_index = {}
        for cls in ("entities", "nouns", "verbs", "adjs", "numbers"):
            for w in features.get(cls, []):
                key = f"{cls}:{w}"
                self.feature_index.setdefault(key, set()).add(idx)

    def _rebuild_feature_index(self):
        """全量重建倒排索引(用于 load 老 brain.pkl)"""
        self.feature_index = {}
        for i, m in enumerate(self.memory):
            f = m.get("features")
            if not f: continue
            self._index_features(i, f)

    def feature_stats(self, top_n=30):
        """返回最高频的特征(按 memory 数量)"""
        if not hasattr(self, "feature_index") or not self.feature_index:
            return []
        items = [(k, len(v)) for k, v in self.feature_index.items()]
        items.sort(key=lambda x: -x[1])
        return items[:top_n]

    def search_by_feature(self, feature, top_k=20):
        """通过特征反查 memory(如 'entities:光速' 找所有提到光速的记忆)"""
        if not hasattr(self, "feature_index"):
            return []
        ids = list(self.feature_index.get(feature, []))[:top_k]
        return [self.memory[i] for i in ids if i < len(self.memory)]

    def _key_stack(self):
        """栈起所有 memory 的 key 成 [N, assoc_dim] 矩阵 + 行 norm 缓存。

        增量式:维护一个带预留容量的缓冲(_key_buf),新增 memory 只往
        缓冲尾部写行,不再整表 np.stack(之前批量巩固 200 条 = 200 次
        3.5GB 拷贝,是 consolidator 吃 CPU 的大头之一)。
        只有维度变化 / 行数回退(reload 换 memory)时才全量重建。
        """
        n = len(self.memory)
        buf = getattr(self, "_key_buf", None)
        filled = getattr(self, "_key_filled", -1)
        need_rebuild = (
            buf is None
            or buf.shape[1] != self.assoc_dim
            or filled > n          # memory 被整体替换/回退
            or filled < 0
            or getattr(self, "_cred_buf", None) is None  # 平行缓冲缺失(旧对象/部分拷贝)
        )
        if need_rebuild:
            cap = max(1024, int(n * 1.25))
            buf = np.zeros((cap, self.assoc_dim), dtype=np.float32)
            norms = np.ones(cap, dtype=np.float32)
            creds = np.ones(cap, dtype=np.float32)
            hems = np.zeros(cap, dtype=np.uint8)  # 0=left 1=right
            if n:
                buf[:n] = np.stack([m["key"] for m in self.memory])
                norms[:n] = np.linalg.norm(buf[:n], axis=1) + 1e-8
                creds[:n] = [m["credibility"] for m in self.memory]
                hems[:n] = [1 if m.get("hemisphere", "left") == "right" else 0
                            for m in self.memory]
            for i, m in enumerate(self.memory):
                m["_idx"] = i
            self._key_buf = buf
            self._key_norm_buf = norms
            self._cred_buf = creds
            self._hem_buf = hems
            self._key_filled = n
            self._keys_version = getattr(self, "_keys_version", 0) + 1
        elif filled < n:
            # 增量追加新行
            if n > buf.shape[0]:
                # 扩容 1.5×
                new_cap = max(int(n * 1.5), buf.shape[0] * 2)
                nbuf = np.zeros((new_cap, self.assoc_dim), dtype=np.float32)
                nbuf[:filled] = buf[:filled]
                nnorms = np.ones(new_cap, dtype=np.float32)
                nnorms[:filled] = self._key_norm_buf[:filled]
                ncreds = np.ones(new_cap, dtype=np.float32)
                ncreds[:filled] = self._cred_buf[:filled]
                nhems = np.zeros(new_cap, dtype=np.uint8)
                nhems[:filled] = self._hem_buf[:filled]
                buf = self._key_buf = nbuf
                self._key_norm_buf = nnorms
                self._cred_buf = ncreds
                self._hem_buf = nhems
            for i in range(filled, n):
                m = self.memory[i]
                buf[i] = m["key"]
                self._key_norm_buf[i] = np.linalg.norm(buf[i]) + 1e-8
                self._cred_buf[i] = m["credibility"]
                self._hem_buf[i] = 1 if m.get("hemisphere", "left") == "right" else 0
                m["_idx"] = i
            self._key_filled = n
            self._keys_version = getattr(self, "_keys_version", 0) + 1
        self._key_matrix = buf[:n]
        self._key_norms = self._key_norm_buf[:n]
        return self._key_matrix

    def _key_norms_stack(self):
        if not hasattr(self, "_key_norms") or self._key_norms is None \
           or self._key_norms.shape[0] != len(self.memory):
            self._key_stack()  # 顺便算 norms
        return self._key_norms

    def _ensure_csr(self):
        """key 矩阵的 CSR 稀疏镜像(keys 是 ~6% 稀疏的二值矩阵,
        CSR 点积的内存流量约为稠密 matmul 的 1/8,单查询 sims 快数倍)。
        版本号驱动重建:keys 变化(重建/追加/巩固原位改)才重建。"""
        if _sp is None:
            return None
        self._key_stack()
        ver = getattr(self, "_keys_version", 0)
        csr = getattr(self, "_csr", None)
        n = len(self.memory)
        if csr is None or getattr(self, "_csr_version", -1) != ver \
           or csr.shape[0] != n:
            csr = _csr_from_dense_parallel(self._key_matrix)
            self._csr = csr
            self._csr_version = ver
        return csr

    def _ensure_csr_blocks(self):
        """把全表 CSR 按行切成 SIMS_THREADS 块,块之间共享同一份 data/indices
        (只是 indptr 偏移,零拷贝),供线程池并行算 matvec。

        scipy 的 csr_matvec 是单线程的、且**会释放 GIL**,所以按行分块用
        Python 线程池就能吃满多核。这台机器 64 核,实测(N=71874,dim=16128,
        topk=945):单线程 91.4ms → 8 线程 24.0ms → 16 线程 15.9ms,
        结果逐元素相同(纯行切分,不改任何算术)。

        ⚠️ 走过的弯路:先试过「转置倒排,只把 query 激活的列加起来」,
        合成数据上 93→24ms,真实数据上反而 364ms。真因是 kWTA 的激活列
        高度集中在 hub 列上 —— query 激活列的命中记忆数中位数 = 71874
        (等于全库),按列取等于把整个矩阵又扫一遍。别再试这条。
        """
        if _sp is None:
            return None
        csr = self._ensure_csr()
        ver = getattr(self, "_keys_version", 0)
        blocks = getattr(self, "_csr_blocks", None)
        if blocks is not None and getattr(self, "_csr_blocks_version", -1) == ver:
            return blocks
        n = csr.shape[0]
        nt = max(1, SIMS_THREADS)
        step = max(1, (n + nt - 1) // nt)
        data, indices, indptr = csr.data, csr.indices, csr.indptr
        blocks = []
        for s in range(0, n, step):
            e = min(s + step, n)
            d0, d1 = int(indptr[s]), int(indptr[e])
            blocks.append((s, e, _sp.csr_matrix(
                (data[d0:d1], indices[d0:d1], indptr[s:e + 1] - d0),
                shape=(e - s, csr.shape[1]), copy=False)))
        self._csr_blocks = blocks
        self._csr_blocks_version = ver
        return blocks

    def _sims_all(self, key):
        """全表 sims 向量。大库走「CSR 行分块 + 线程池」,
        scipy 不可用或小库时退回稠密 matmul。"""
        K = self._key_stack()
        if _sp is None or K.shape[0] < 20000:
            return K @ key
        blocks = self._ensure_csr_blocks()
        if blocks is None or len(blocks) <= 1 or _SIMS_POOL is None:
            return self._ensure_csr().dot(key)
        out = np.empty(K.shape[0], dtype=np.float32)

        def _run(b):
            s, e, m = b
            out[s:e] = m.dot(key)

        list(_SIMS_POOL.map(_run, blocks))
        return out

    def _cos_all(self, prompt):
        """(prompt, keys_version) -> 全表 cosine,带小缓存。

        一次 /api/chat 里「左脑 → 右脑 fallback → 兜底召回 → creative」
        问的是同一个 prompt,以前每条路各跑一遍全表点乘(93ms×3);
        现在只算一次。cred 会在命中后被改写,所以缓存只存 cos,
        score = cos*cred 仍每次重算(7万次乘法,~0.05ms)。
        """
        ver = getattr(self, "_keys_version", 0)
        n = len(self.memory)
        cache = getattr(self, "_cos_cache", None)
        if cache is None:
            cache = self._cos_cache = {}
        hit = cache.get(prompt)
        if hit is not None and hit[0] == ver and hit[1] == n:
            return hit[2]
        key = self.encode(prompt)
        norms = self._key_norms_stack()
        sims = self._sims_all(key)
        cos = sims / (norms * (np.linalg.norm(key) + 1e-8))
        if len(cache) >= COS_CACHE_SIZE:
            cache.clear()
        cache[prompt] = (ver, n, cos)
        return cos

    def _ensure_bigram_index(self):
        """topic 实义 bigram -> memory 下标 的倒排索引 + 短 topic 下标表。
        替代 _topic_bigram_recall 的全表扫(68k 次 set 构造/查询)。
        增量:只补 filled..n 的新 memory;整体替换/回退时全量重建。"""
        n = len(self.memory)
        idxmap = getattr(self, "_bigram_idx", None)
        filled = getattr(self, "_bigram_filled", -1)
        if idxmap is None or filled < 0 or filled > n:
            idxmap = {}
            self._short_topic_idx = []
            self._topic_bi = []
            self._topic_index = {}
            filled = 0
        if getattr(self, "_topic_bi", None) is None:
            self._topic_bi = []
        if getattr(self, "_topic_index", None) is None:
            self._topic_index = {}
        for i in range(filled, n):
            raw = self.memory[i].get("topic") or ""
            self._topic_index.setdefault(raw, []).append(i)
            t = _GATE_STRIP_RE.sub("", raw).lower()
            if not t:
                self._topic_bi.append(frozenset())
                continue
            # 与 _substantive_bigrams(topic) 等价,顺手缓存:rerank 循环里
            # 每次请求要对上千候选算一遍正则+set,是第二大热点。
            if len(t) == 1:
                self._topic_bi.append(frozenset({t} - _STOP_BIGRAMS))
            else:
                self._topic_bi.append(frozenset(
                    {t[j:j + 2] for j in range(len(t) - 1)} - _STOP_BIGRAMS))
            if len(t) <= 6:
                self._short_topic_idx.append(i)
            for j in range(len(t) - 1):
                bg = t[j:j + 2]
                if bg in _STOP_BIGRAMS:
                    continue
                idxmap.setdefault(bg, []).append(i)
        self._bigram_idx = idxmap
        self._bigram_filled = n
        return idxmap

    def _topic_bigrams_at(self, i):
        """第 i 条 memory 的 topic 实义 bigram(缓存版,等价 _substantive_bigrams)。"""
        cache = getattr(self, "_topic_bi", None)
        if cache is None or i >= len(cache):
            self._ensure_bigram_index()
            cache = self._topic_bi
        if i < len(cache):
            return cache[i]
        return _substantive_bigrams(self.memory[i].get("topic") or "")

    def _lexical_candidates(self, prompt, cap=3000):
        """词法召回通道:query 实义 bigram 命中的 memory 下标集合。
        单字核心(如"什么是熵"→"熵")走短 topic 字符包含兜底。"""
        idxmap = self._ensure_bigram_index()
        cand = set()
        for bg in _substantive_bigrams(prompt):
            lst = idxmap.get(bg)
            if lst:
                cand.update(lst)
                if len(cand) >= cap:
                    break
        q_core = _strip_question_template(prompt)
        if len(q_core) == 1:
            for i in getattr(self, "_short_topic_idx", []):
                t = self.memory[i].get("topic") or ""
                if q_core in t.lower():
                    cand.add(i)
        return cand

    def _sync_cred(self, m):
        """单条 memory 的 credibility 变化后同步进缓存向量。"""
        i = m.get("_idx")
        buf = getattr(self, "_cred_buf", None)
        if buf is not None and i is not None and 0 <= i < getattr(self, "_key_filled", 0):
            buf[i] = m["credibility"]

    def _best_match(self, key):
        K = self._key_stack()
        if K.shape[0] == 0:
            return -1, -1.0
        sims = K @ key
        # 稀疏二值,cosine 等价于 dot/sqrt(k1*k2),分母对所有 key 都一样,直接 argmax
        idx = int(np.argmax(sims))
        # 实际 cosine
        norm_q = np.linalg.norm(key)
        norm_k = np.linalg.norm(K[idx])
        score = float(sims[idx] / (norm_q * norm_k + 1e-8))
        return idx, score

    # =================================================================
    # 双脑路由
    # =================================================================
    def hemisphere_route(self, prompt):
        """根据 query 文本特征决定走哪个半球。返回 'left' / 'right' / 'both'
        注意:大多语料目前在左脑(web/sentence),右脑只有视觉,所以 'both' 是默认安全
        路径 — 只对明确情感/招呼词强制路由到右脑。
        """
        if not prompt:
            return "both"
        p = prompt.lower()
        # 招呼语 / 情感词 -> 右脑(联想 + 个性化回答)
        if any(g in prompt for g in GREETING_WORDS):
            return "right"
        if any(e in prompt for e in EMOTION_WORDS):
            return "right"
        # 含疑问词 + 包含命名实体 -> 左脑(分析回答)
        has_q = any(w in p for w in QUESTION_WORDS)
        has_entity = bool(_re_entity.search(prompt))
        if has_q and has_entity:
            return "left"
        # 默认:双脑会诊
        return "both"

    def _topic_bigram_recall(self, prompt, hem=None, top_k=20):
        """兜底召回:凡 topic 含 query 任一实义 bigram(去 stopword)的 memory
        都计算 SDR cosine,返回按 sim*cred 排序的 top_k。
        候选来自倒排索引(_ensure_bigram_index),不再全表扫。
        """
        with self._lock:
            if not self.memory:
                return []
            # 倒排索引召回(bigram 命中 + 单字核心短 topic 包含)
            cand_idx = sorted(self._lexical_candidates(prompt))
            if not cand_idx:
                return []
            if hem in ("left", "right"):
                cand_idx = [i for i in cand_idx
                            if self.memory[i].get("hemisphere", "left") == hem]
                if not cand_idx:
                    return []
            cand_idx_arr = np.array(cand_idx)
            # 复用整表 cos(同一 prompt 一次请求内只算一次);
            # 旧写法 K[cand_idx] 会拷一份 3000×16128 的稠密子矩阵(~190MB)。
            cos = self._cos_all(prompt)[cand_idx_arr]
            creds = self._cred_buf[cand_idx_arr]
            score = cos * creds
            order = np.argsort(-score)[:top_k]
            return [(self.memory[cand_idx[k]], float(cos[k]), float(creds[k]))
                    for k in order]

    def retrieve(self, prompt, top_k=RETRIEVE_TOPK, hemisphere=None):
        """检索 top-k。hemisphere=None|'left'|'right'|'both'
        返回 [(memory, sim, cred), ...] 按 credibility * sim 排序

        持锁:与 brain hot-reload 互斥,防止 self.memory / self._key_matrix
        被替换到一半时撞上 numpy 大矩阵操作。
        """
        with self._lock:
            return self._retrieve_locked(prompt, top_k, hemisphere)

    def _retrieve_locked(self, prompt, top_k=RETRIEVE_TOPK, hemisphere=None):
        if not self.memory:
            return []

        n = len(self.memory)
        cos = self._cos_all(prompt)
        creds = self._cred_buf[:n]
        score = cos * creds
        # 半球过滤(缓存的 uint8 向量,0=left 1=right)
        if hemisphere == "left":
            score = np.where(self._hem_buf[:n] == 0, score, -1.0)
        elif hemisphere == "right":
            score = np.where(self._hem_buf[:n] == 1, score, -1.0)

        # ===== 三通道:SDR 召回池 ∪ 词法倒排候选 → lexical rerank =====
        # SDR cosine 在短中文文本上区分度有限(大库上普遍饱和到 0.99+),
        # 纯按 SDR 分取池时真命中常被短句噪音挤出池外(结构性漏召)。
        # 词法通道用倒排索引把"topic 与 query 有实义 bigram 重叠"的候选
        # 强制拉进池,再用 bigram Jaccard 融合重排收口。
        pool = min(max(top_k * 10, RERANK_POOL), n)
        top_idx = np.argpartition(-score, pool - 1)[:pool]
        q_bi = _substantive_bigrams(prompt)
        cand = {int(i) for i in top_idx}
        if q_bi:
            cand.update(self._lexical_candidates(prompt, cap=2000))
        rescored = []
        for i in cand:
            if score[i] < 0:
                continue
            lex = 0.0
            if q_bi:
                t_bi = self._topic_bigrams_at(i)
                if t_bi:
                    inter = len(q_bi & t_bi)
                    if inter:
                        lex = inter / len(q_bi | t_bi)
            final = score[i] * (1.0 + RERANK_LEX_BOOST * lex)
            rescored.append((final, i))
        rescored.sort(key=lambda x: -x[0])
        out = []
        for _, i in rescored[:top_k]:
            out.append((self.memory[i], float(cos[i]), float(creds[i])))
        return out

    def answer(self, prompt, sim_threshold=0.35, sampling=None):
        """检索式回答(双脑版)。整段在 _lock 内,与 brain hot-reload 互斥,
        保证 _key_matrix / memory 不会在中途被替换。
        sampling=None 自动决定:
          - 右脑(招呼/情感)→ 总是采样(回答多变)
          - 否则:top-3 有多个候选分数接近 top-1(差 < 0.05)→ 采样(明显是同 topic 多变体)
          - 否则:确定性 top-1
        """
        with self._lock:
            return self._answer_locked(prompt, sim_threshold, sampling)

    def _answer_locked(self, prompt, sim_threshold=0.35, sampling=None):
        hem = self.hemisphere_route(prompt)
        # 真正的 sampling 决定:稍后看到 results 再判
        force_sampling = (hem == "right") if sampling is None else sampling

        # 左/右半球 OR both
        results = self.retrieve(prompt, top_k=RETRIEVE_TOPK, hemisphere=hem)

        def _strong_hit(rs):
            return rs and rs[0][1] >= sim_threshold

        # 单脑空 / 弱 -> fallback 到另一半球
        # (hem="both" 时 retrieve 本就不过滤半球,再查一次 hemisphere=None
        #  是完全相同的重复计算,已去掉)
        if hem in ("left", "right") and not _strong_hit(results):
            other = "right" if hem == "left" else "left"
            other_results = self.retrieve(prompt, top_k=RETRIEVE_TOPK, hemisphere=other)
            if _strong_hit(other_results):
                results = other_results
                hem = f"{other}_fallback"

        if not results:
            try:
                self.agi.observe_query_fast(prompt, None, 0.0, None, hem)
            except Exception:
                pass
            return None, {"reason": "empty_memory", "hemisphere": hem}

        # 字符 2-gram gate(排除疑问/虚词 bigram):
        # 防稀疏 SDR 在不相关 query 上虚高 cosine 误命中。
        # 例:"量子纠缠是什么" vs topic"你会什么" 仅共享"什么"(stopword)→ 拒;
        #     "光速是多少"     vs topic"光速"     共享实义 "光速"        → 通过。
        gate_active = bool(_char_bigrams(prompt))
        used_recall_fallback = False
        if gate_active:
            filtered = [(m, s, c) for m, s, c in results
                        if _substantive_overlap(prompt, m.get("topic", ""))]
            if filtered:
                results = filtered
            else:
                # SDR top-K 太小,真命中常被短句噪音挤掉。
                # 用 query 的实义 bigram 全表扫 topic 兜底召回,再用 substring gate 精排。
                recall = self._topic_bigram_recall(prompt, hem)
                recall = [(m, s, c) for m, s, c in recall
                          if _substantive_overlap(prompt, m.get("topic", ""))]
                if not recall:
                    top_mem_orig, top_sim_orig, _ = results[0]
                    try:
                        self.agi.observe_query_fast(prompt, top_mem_orig.get("key"), float(top_sim_orig) * 0.4, None, hem)
                    except Exception:
                        pass
                    return None, {"reason": "no_topic_overlap",
                                  "top_sim": top_sim_orig,
                                  "top_topic": top_mem_orig["topic"],
                                  "hemisphere": hem}
                results = recall
                used_recall_fallback = True

        top_mem, top_sim, top_cred = results[0]
        # 兜底召回的 sim 可能低于 sim_threshold,但既然 topic 实义重叠 + 走过 gate,
        # 接受较低 sim(BRAIN 创造性下限);主路径 SDR 命中仍要满足主阈值。
        min_sim = BRAIN_CREATIVE_MIN if used_recall_fallback else sim_threshold
        if top_sim < min_sim:
            try:
                self.agi.observe_query_fast(prompt, top_mem.get("key"), float(top_sim), top_mem.get("topic"), hem)
            except Exception:
                pass
            return None, {"reason": "below_threshold", "top_sim": top_sim,
                          "top_topic": top_mem["topic"], "hemisphere": hem}

        # 自动判采样:多个候选 sim 接近 top → 必然有多变体 → 随机选
        do_sample = force_sampling
        if not do_sample and len(results) >= 2:
            close = sum(1 for _, s, _ in results[:5] if s >= top_sim - 0.05)
            if close >= 2:
                do_sample = True

        if do_sample and len(results) >= 2:
            # 取前 5 个高 sim 候选,按 sim*cred 加权采样
            cand = [r for r in results[:5]
                    if r[1] >= max(sim_threshold, top_sim - 0.10)]
            if len(cand) >= 2:
                weights = np.array([s * c for _, s, c in cand], dtype=np.float64)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    pick = int(np.random.choice(len(cand), p=weights))
                    top_mem, top_sim, top_cred = cand[pick]

        # 隐式 RL: 命中 -> hits++、cred 微涨
        top_mem["hits"] = top_mem.get("hits", 0) + 1
        top_mem["last_seen"] = time.time()
        top_mem["credibility"] = min(1.0, top_mem["credibility"] + 0.005)
        self._sync_cred(top_mem)  # 同步进 cred 缓存向量

        agi_meta = {}
        try:
            agi_meta = self.agi.observe_query_fast(
                prompt, top_mem.get("key"), float(top_sim), top_mem.get("topic"), hem
            )
        except Exception:
            pass

        return top_mem["content"], {
            "topic": top_mem["topic"],
            "source": top_mem["source"],
            "credibility": top_cred,
            "similarity": top_sim,
            "hemisphere": hem,
            "sampled": do_sample,
            "alternatives": [(m["topic"], s) for m, s, _ in results[1:3]],
            "agi": agi_meta,
        }


    def deep_think(self, prompt, depth=3, sim_threshold=0.30):
        """链式联想思考: prompt -> top topic -> 跳到不同主题再查 -> ... 共 depth 步。
        每轮排除已访问的主题,让链路真正扩散。
        返回 {chain: [...], final: (content, dbg)}
        """
        visited = set()
        chain = [{"step": 0, "query": prompt}]
        cur = prompt
        for i in range(depth):
            results = self.retrieve(cur, top_k=5)
            # 跳过自匹配 / 已访问
            picked = None
            for m, sim, cred in results:
                if m["topic"] == cur or m["topic"] in visited:
                    continue
                if sim < sim_threshold:
                    continue
                picked = (m, sim, cred)
                break
            if not picked:
                break
            m, sim, cred = picked
            visited.add(m["topic"])
            chain.append({
                "step": i + 1,
                "query": cur,
                "matched_topic": m["topic"],
                "similarity": round(sim, 3),
                "source": m["source"],
                "preview": (m["content"] or "")[:60],
            })
            cur = m["topic"]
        if len(chain) <= 1:
            return {"chain": chain, "final": (None, {"reason": "no_chain"})}
        ans, dbg = self.answer(prompt)
        dbg["thought_chain"] = chain
        return {"chain": chain, "final": (ans, dbg)}

    def feedback(self, topic_or_idx, delta=0.05):
        """显式 RL 反馈: +delta(👍) 或 -delta(👎)。返回更新后的 cred"""
        if isinstance(topic_or_idx, int):
            if 0 <= topic_or_idx < len(self.memory):
                m = self.memory[topic_or_idx]
            else:
                return None
        else:
            m = next((x for x in self.memory if x["topic"] == topic_or_idx), None)
            if m is None:
                return None
        m["credibility"] = max(0.05, min(1.0, m["credibility"] + delta))
        m["last_seen"] = time.time()
        self._sync_cred(m)
        return m["credibility"]

    def learn_dialogue(self, prompt, response, source="live", credibility=0.7):
        """对话不直接污染 brain.memory(避免回声学习):
          - 写 PG sentence_corpus pending(由 dialogue_logger 完成)
          - daemon 审核 credibility 后才促入脑
        这里只 bump 已有完全匹配条目的 hits/cred(隐式 RL)
        """
        # 走 topic 倒排,别再对 7 万条记忆做 Python 线性扫(每次命中都白烧 ~20ms)
        self._ensure_bigram_index()
        for i in (self._topic_index.get(prompt) or ()):
            m = self.memory[i]
            if m["content"] == response:
                m["hits"] = m.get("hits", 0) + 1
                m["last_seen"] = time.time()
                m["credibility"] = min(1.0, m["credibility"] + 0.005)
                self._sync_cred(m)
                return None
        return None  # 不新增,避免污染

    def consolidate_batch(self, n=50, lr=0.001):
        """后台慢巩固:用最近 n 条记忆做小步 Hebbian。
        比 online 学习温和得多,不会让编码器漂移。

        CPU 优化:lr<=0.001、decay=1e-7 时 W1 的单轮变化量级 ~1e-4,
        旧 keys 与新编码器的 kWTA top-k 集合几乎不变 — 之前每轮全量
        重编码 62k 条(~1 TFLOP)纯属浪费。现在只重编码本批 n 条,
        并原位更新 key 缓冲,无任何全表操作。
        """
        if not self.memory:
            return 0
        start = max(0, len(self.memory) - n)
        batch = self.memory[start:]
        for m in batch:
            ctx = self.context_vector(m["topic"])
            key = self.encode(m["topic"])
            self._hebbian_update(ctx, key, lr=lr)
        # 只刷新本批的 key,并同步进增量缓冲(若已建)
        new_keys = self.encode_batch([m["topic"] for m in batch])
        buf = getattr(self, "_key_buf", None)
        filled = getattr(self, "_key_filled", -1)
        for j, m in enumerate(batch):
            m["key"] = new_keys[j]
            i = start + j
            if buf is not None and i < filled and buf.shape[1] == self.assoc_dim:
                buf[i] = new_keys[j]
                self._key_norm_buf[i] = np.linalg.norm(new_keys[j]) + 1e-8
            try:
                self.agi.dao_core.step(
                    new_keys[j],
                    prediction_error=max(0.0, 1.0 - float(m.get("credibility", 0.7))),
                    learn=True,
                    topic_label=m.get("topic", ""),
                )
                self.agi.graph.ingest_knowledge_passage(
                    topic=m.get("topic", ""),
                    content=m.get("content", ""),
                    source=m.get("source", "consolidate"),
                    credibility=float(m.get("credibility", 0.7)),
                    meta=m.get("meta"),
                )
            except Exception:
                pass
        try:
            self.agi.run_sleep_consolidation(n_replay=min(16, len(batch)), lr=0.02)
        except Exception:
            pass
        # keys 原位变更,CSR 镜像失效
        self._keys_version = getattr(self, "_keys_version", 0) + 1
        return len(batch)

    @property
    def agi(self):
        eng = getattr(self, "_agi_engine", None)
        if eng is None:
            from brain.daxy_agi_core import get_agi_engine
            eng = get_agi_engine(brain=self)
            self._agi_engine = eng
        return eng

    def decay_unused(self, max_seconds=86400 * 30):
        """长时间未命中的记忆可信度缓慢降低(模拟遗忘)"""
        now = time.time()
        for m in self.memory:
            days_idle = max(0, (now - m["last_seen"]) / 86400)
            m["credibility"] = max(0.05, m["credibility"] - CRED_DECAY_PER_DAY * days_idle)
            self._sync_cred(m)

    def stats(self):
        sources = {}
        left = right = 0
        for m in self.memory:
            sources[m["source"]] = sources.get(m["source"], 0) + 1
            hem = m.get("hemisphere", "left")
            if hem == "left": left += 1
            elif hem == "right": right += 1
        # 特征统计
        feature_count = 0
        if hasattr(self, "feature_index") and self.feature_index:
            feature_count = len(self.feature_index)
        dao_info = {}
        try:
            dao_info = {
                "dao_prototypes": self.agi.dao_core.cfg.n_prototypes,
                "dao_usage_entropy": round(self.agi.dao_core.usage_entropy(), 4),
                "multimodal_graph_nodes": len(self.agi.graph.nodes),
                "multimodal_graph_edges": len(self.agi.graph.edges),
                "procedures_count": len(self.agi.procedures),
            }
        except Exception:
            pass
        return {
            "neurons": PERC_DIM + self.assoc_dim + CONCEPT_DIM,
            "perception_dim": PERC_DIM,
            "association_dim": self.assoc_dim,
            "concept_dim": CONCEPT_DIM,
            "assoc_topk": self.assoc_topk,
            "growth_events": len(self.growth_log),
            "params_M": self.W1.size / 1e6,
            "memories": len(self.memory),
            "left_brain": left,
            "right_brain": right,
            "feature_count": feature_count,
            "by_source": sources,
            "vocab": len(self.perc_sdr),
            "steps": self.steps,
            "W1_density": float((self.W1 > 0.01).mean()),
            **dao_info,
        }

