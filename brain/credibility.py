"""
内容置信度评分 - 综合多个信号判断一条 corpus 是否可信。

输出 [0, 1]:
  >= 0.7 高置信(可直接 promote)
  0.3 - 0.7 中等(看 ON CONFLICT 加成)
  < 0.3 可疑(自动 quarantine,不入大脑)

信号:
  1. 来源可信度(baike > dialog > seq2seq generated)
  2. 内容长度(过短可疑)
  3. 字符分布(中文主题应该主要含 CJK)
  4. 乱码检测(� / GBK→UTF8 残留 / Latin1 残留)
  5. 重复率(单字反复)
  6. 主题相关性(内容是否提到主题字)
  7. 控制字符比例
"""
from collections import Counter

# 来源基准分(经验值)
SOURCE_BASELINE = {
    "web_corpus": 0.75,       # 百科类
    "baike": 0.75,
    "wikipedia": 0.85,
    "sentence_corpus": 0.50,  # 对话来源,需验证
    "vision_corpus": 0.55,
    "qa_recall": 0.50,
    "brain": 0.55,
    "live": 0.45,
    "chat_seq2seq": 0.25,     # Seq2Seq 自由生成,低置信
    "user_verified": 0.95,
}

# Latin-1 解码 UTF-8 后常见乱码字符
MOJIBAKE_HINTS = {"Ã", "â", "ã", "ä", "Ã©", "Ã¨", "Ã ", "â\x80", "ï¿½"}


def _char_classes(text):
    if not text:
        return 0.0, 0.0, 0.0, 0.0
    cjk = ascii_p = ctrl = 0
    for c in text:
        if "一" <= c <= "鿿" or "㐀" <= c <= "䶿":
            cjk += 1
        elif c.isprintable() and ord(c) < 128:
            ascii_p += 1
        elif (not c.isprintable() and c not in "\n\r\t") or c == "�":
            ctrl += 1
    n = len(text)
    other = n - cjk - ascii_p - ctrl
    return cjk / n, ascii_p / n, other / n, ctrl / n


def _has_mojibake(text):
    if "�" in text:
        return True
    # Latin-1 解码 UTF-8 残留
    sample = text[:500]
    bad = sum(sample.count(p) for p in MOJIBAKE_HINTS)
    return bad >= 3


def _repetition_penalty(text):
    """单字反复程度。返回 [0, 1] 越高越正常。"""
    if len(text) < 20:
        return 1.0
    counts = Counter(text)
    if not counts:
        return 1.0
    top_char, top_count = counts.most_common(1)[0]
    rep = top_count / len(text)
    # 标点符号常见,允许到 0.25
    if rep <= 0.25:
        return 1.0
    if rep >= 0.6:
        return 0.0
    return 1.0 - (rep - 0.25) / 0.35


def _topic_relevance(topic, content):
    """内容里是否提到主题字。"""
    if not topic or not content:
        return 0.0
    sample = content[:1500]
    # 子串完整匹配
    if topic in sample:
        return 1.0
    # 按字字符重叠
    topic_chars = set(topic)
    content_chars = set(sample)
    if not topic_chars:
        return 0.0
    overlap = len(topic_chars & content_chars) / len(topic_chars)
    return overlap


def score(topic, content, source="unknown"):
    """主入口。返回 0..1 置信度。"""
    if not content or not isinstance(content, str):
        return 0.0
    L = len(content)
    if L < 10:
        return 0.0
    if _has_mojibake(content):
        return 0.0

    cjk_r, ascii_r, _, ctrl_r = _char_classes(content)
    if ctrl_r > 0.05:
        return 0.05

    base = SOURCE_BASELINE.get(source, 0.45)

    # 长度因子
    if L < 30:
        len_f = 0.45
    elif L < 80:
        len_f = 0.75
    elif L < 5000:
        len_f = 1.0
    else:
        len_f = 0.9  # 太长可能掺水

    # 字符分布:如果主题是中文,内容应主要是中文
    topic_is_cjk = topic and any("一" <= c <= "鿿" for c in topic)
    if topic_is_cjk and cjk_r < 0.30:
        # 中文主题但内容主要不是中文 -> 严重不匹配
        return 0.10

    # 重复 / 相关性
    rep = _repetition_penalty(content)
    rel = _topic_relevance(topic or "", content)

    # 加权融合
    quality = len_f * rep * (0.6 + 0.4 * rel)
    final = base * 0.55 + quality * 0.45
    return max(0.0, min(1.0, final))


def explain(topic, content, source="unknown"):
    """调试用:返回各项分数和最终值"""
    if not content:
        return {"final": 0.0, "reason": "empty"}
    L = len(content)
    if _has_mojibake(content):
        return {"final": 0.0, "reason": "mojibake_detected"}
    cjk_r, ascii_r, _, ctrl_r = _char_classes(content)
    return {
        "length": L,
        "cjk_ratio": round(cjk_r, 3),
        "ascii_ratio": round(ascii_r, 3),
        "control_ratio": round(ctrl_r, 3),
        "repetition": round(_repetition_penalty(content), 3),
        "topic_relevance": round(_topic_relevance(topic or "", content), 3),
        "source_baseline": SOURCE_BASELINE.get(source, 0.45),
        "final": round(score(topic, content, source), 3),
    }
