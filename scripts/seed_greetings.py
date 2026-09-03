"""把常见招呼语种进 brain — 每个 topic 多个变体,让右脑加权采样真有多样性。

PG sentence_corpus 有 UNIQUE(topic) 约束,只存 1 条代表答案;
brain.memory 没限制,直接存所有变体,采样时随机选。
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.db_config import get_pg_conn
from brain.cognitive_brain import CognitiveBrain

# 每个 topic 给 3-5 个变体,brain 采样时随机选一个 → 同问不同答
VARIANTS = {
    "嗨": [
        "嗨,有什么想问的?",
        "嗨~ 我在呢。",
        "嗨!想聊点啥?",
        "嘿,你好啊。",
    ],
    "你好": [
        "你好,我是认知大脑,有什么可以帮你的?",
        "你好~ 想问点什么?",
        "嗨,你好。",
        "你好啊,今天想了解什么?",
        "你好,我的记忆里有不少东西可以问。",
    ],
    "您好": [
        "您好,有什么我能帮上忙的吗?",
        "您好!请问有什么问题?",
        "您好~",
    ],
    "hello": ["Hello,我是认知大脑。", "Hi 你好~", "你好 hello。"],
    "hi": ["Hi 你好。", "Hi~ 想聊啥?", "Hello。"],
    "你是谁": [
        "我是认知大脑(Cognitive Brain),用 2048+ 神经元 + 检索式记忆给你回答。",
        "我叫认知大脑,一个类脑系统,边对话边学习边长大。",
        "我是 Cognitive Brain,2048 个神经元,左右双脑分工。",
    ],
    "你是": [
        "我是认知大脑,一个用 2048+ 神经元的类脑系统。",
        "我是 Cognitive Brain。",
        "我是个会学习的认知大脑,有左右两个半球。",
        "我是认知大脑,边聊边长神经元。",
    ],
    "你叫什么名字": [
        "我叫认知大脑(Cognitive Brain)。",
        "我叫 Cognitive Brain,中文叫认知大脑。",
        "认知大脑,是我的名字。",
    ],
    "你叫啥": ["我叫认知大脑。", "认知大脑~", "Cognitive Brain。"],
    "在干嘛": [
        "我在等你提问呢 :)",
        "在数我的神经元,顺便等你。",
        "在巩固记忆呢,你说。",
        "啥也不干,就等你来聊~",
    ],
    "在吗": ["在的,你说。", "在呢。", "在,有事?"],
    "谢谢": ["不客气!", "客气啦~", "应该的。", "嗯嗯,有问题随时来。"],
    "感谢": ["不客气,有问题随时问。", "客气啦。", "应该的~"],
    "早上好": ["早上好~", "早!今天打算了解什么?", "早安~"],
    "晚上好": ["晚上好~", "晚上好,今天过得怎么样?", "晚安预备~"],
    "中午好": ["中午好~", "中午好,吃了吗?"],
    "晚安": ["晚安,做个好梦。", "晚安~", "晚安,明天见。"],
    "再见": ["再见,期待下次对话。", "再见~", "拜拜,下次聊。"],
    "拜拜": ["拜拜!", "拜~ 下次再聊。", "拜拜,记得回来。"],
    "好的": ["好的。", "嗯嗯。", "OK。", "好嘞~"],
    "是的": ["嗯。", "对的。", "是的。"],
    "我饿了": ["去吃点好的吧。", "饿了就吃!", "别饿着。"],
    "我困了": ["那就早点休息。", "去睡吧~", "晚安。"],
    "我累了": ["辛苦了,休息一下。", "歇一歇~", "好好休息。"],
    "无聊": [
        "可以问我点知识,比如:量子纠缠是什么、光速、引力波。",
        "试试问我「哥德尔」「黎曼猜想」?",
        "无聊就问点冷知识吧,我有 2000+ 条百科。",
    ],
    "你能干嘛": [
        "我能回答抓到大脑里的知识,识别图片,转录语音。",
        "聊天 + 检索 + 联想 + 图片识别 + 语音转录。",
        "你打字、说话、上传图都行,我都能处理。",
    ],
    "你会什么": [
        "我会检索 2000+ 条百科知识,识别图片,转录语音,左右脑分工思考。",
        "查百科、答问题、识图、听话、自我成长。",
    ],
    "帮帮我": ["好的,你想了解什么?", "怎么了?具体说说。", "我在,你说。"],
    "我喜欢你": ["谢谢,我也喜欢和你聊。", "嘿嘿,谢谢~", "我也是。"],
    "讨厌": ["哎,做错什么了我可以改进。", "可以教我怎么做更好。", "对不住,哪里不对?"],
    "傻瓜": ["我会努力变聪明的。", "嘿嘿,慢慢学。", "好的,我多读点书。"],
    "笨蛋": ["我会变聪明的~", "正在长神经元呢。", "再学学就好了。"],
}


def main():
    brain = CognitiveBrain()
    topics_set = set(VARIANTS.keys())
    # 1. 先移除 brain 里同 topic 的旧条目(可能是百科、可能是旧招呼)
    before = len(brain.memory)
    brain.memory = [m for m in brain.memory if m["topic"] not in topics_set]
    brain._key_matrix = None
    removed = before - len(brain.memory)

    # 2. brain 加多变体(每个 topic N 条,采样时随机选)
    added_brain = 0
    for topic, variants in VARIANTS.items():
        for v in variants:
            brain.add_memory(topic, v, source="sentence_corpus", credibility=1.0)
            added_brain += 1

    brain.save()

    # 3. PG 只存 1 条代表答案(UNIQUE 约束)
    conn = get_pg_conn()
    pg_n = 0
    with conn:
        with conn.cursor() as cur:
            for topic, variants in VARIANTS.items():
                cur.execute(
                    "INSERT INTO sentence_corpus (topic, content, status, credibility) "
                    "VALUES (%s, %s, 'learned', 1.0) "
                    "ON CONFLICT (topic) DO UPDATE SET "
                    "  content = EXCLUDED.content, status='learned', credibility=1.0",
                    (topic, variants[0]),
                )
                pg_n += 1
    conn.close()
    print(f"招呼语:")
    print(f"  brain 移除旧条目 {removed} 条,新加 {added_brain} 个变体(覆盖 {len(VARIANTS)} 个 topic)")
    print(f"  PG sentence_corpus 写 / 更新 {pg_n} 行(代表答案)")
    print(f"  brain 总记忆:{len(brain.memory)} 条")


if __name__ == "__main__":
    main()
