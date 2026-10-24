"""
持续抓取 daemon - 自我发现新主题 + 周期抓取。

工作循环:
  1. 队列空时:从已抓内容提取新名词作种子(jieba 词性标注 + 排除已存在)
  2. 队列里取一个主题 -> 抓 baike.baidu.com -> 写 web_corpus pending
  3. 限速(每 N 秒一次),避免被反爬
  4. 失败重试 3 次后跳过

启动后无限运行,直到 Ctrl+C。
"""
import argparse
import io
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jieba
import jieba.posseg as psg
import logging
jieba.setLogLevel(logging.ERROR)

from utils.db_config import get_pg_conn
from utils.log_setup import setup_daemon_logger
from scripts.web_crawler import (
    fetch_baike, insert_pending, download_image, get_last_image_url,
)
import scripts.web_crawler as _web_crawler

IMAGES_DIR = ROOT / "corpus" / "images"
LOG = setup_daemon_logger("crawler")


def safe_filename(topic):
    """topic -> sample_<safe>.jpg 文件名(过滤路径字符)"""
    bad = '<>:"/\\|?*'
    safe = "".join("_" if c in bad else c for c in topic)
    return f"sample_{safe}.jpg"


def fetch_and_save_image(topic, content):
    """下图 + 写 vision_corpus pending。
    优先用文字源回带的 og:image;没有则去 Wikimedia / Bing / 百度图片搜一张。
    返回 (saved_path or None, vision_id or None)
    """
    img_url = get_last_image_url(topic)
    img_source = "og:image"
    if not img_url:
        # 兜底:专门去图片搜索引擎找
        try:
            from skills.image_search import search_image
            img_url, img_source = search_image(topic, timeout=8)
        except Exception as e:
            LOG.debug(f"  image_search error for {topic}: {e}")
            img_url = None
    if not img_url:
        return None, None
    fname = safe_filename(topic)
    dest = IMAGES_DIR / fname
    if dest.exists():
        return str(dest), None  # 已下过(不重复入 vision_corpus)
    saved = download_image(img_url, str(dest))
    if not saved:
        return None, None
    rel = f"corpus/images/{fname}"
    try:
        conn = get_pg_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO vision_corpus (image_path, question, answer, status, surprise_score) "
                        "VALUES (%s, %s, %s, 'pending', %s) RETURNING id",
                        (rel, f"{topic}是什么", content, 0.7),
                    )
                    rid = cur.fetchone()[0]
        finally:
            conn.close()
        LOG.info(f"     IMG src={img_source} -> {fname}")
        return saved, rid
    except Exception as e:
        LOG.warning(f"  ! vision insert error for {topic}: {e}")
        return saved, None


SEED_TOPICS = [
    # ── 学科主干 ──
    "物理学", "化学", "生物学", "计算机科学", "数学",
    "哲学", "心理学", "经济学", "天文学", "语言学", "人类学",
    "社会学", "政治学", "法学", "考古学", "地质学", "气象学",
    "海洋学", "地理学", "材料科学", "工程学", "医学", "药学",
    # ── 学科细分 ──
    "量子力学", "相对论", "广义相对论", "狭义相对论", "热力学", "电动力学",
    "固体物理", "原子物理", "粒子物理", "天体物理", "凝聚态物理",
    "有机化学", "无机化学", "物理化学", "分析化学", "高分子化学",
    "分子生物学", "细胞生物学", "遗传学", "微生物学", "生态学", "神经科学",
    "代数", "几何", "概率论", "数理逻辑", "组合数学",
    "操作系统", "编译原理", "数据结构", "算法", "数据库", "计算机网络",
    "机器学习", "神经网络", "卷积神经网络", "循环神经网络", "注意力机制",
    "Transformer", "区块链", "云计算", "分布式系统", "信息论",
    # ── 物理概念 ──
    "牛顿三大定律", "麦克斯韦方程", "波粒二象性", "暗物质", "暗能量", "弦理论",
    "黑洞", "白洞", "中子星", "脉冲星", "超新星", "宇宙微波背景辐射",
    "引力波", "量子纠缠", "薛定谔方程", "海森堡不确定性原理", "测不准原理",
    "热力学第二定律", "熵", "玻尔兹曼常数", "普朗克常数", "光速",
    "电磁场", "电磁感应", "洛伦兹力", "麦克斯韦妖", "卡西米尔效应",
    "夸克", "中微子", "希格斯玻色子", "标准模型", "超对称", "霍金辐射",
    # ── 数学概念 ──
    "欧拉公式", "微积分", "希尔伯特空间", "拓扑学", "数论", "群论",
    "范畴论", "傅里叶变换", "拉普拉斯变换", "矩阵", "行列式", "特征值",
    "黎曼猜想", "费马大定理", "哥德巴赫猜想", "四色定理", "庞加莱猜想",
    "黄金分割", "斐波那契数列", "贝叶斯定理", "马尔可夫链", "蒙特卡洛方法",
    "图论", "动态规划", "线性代数", "复分析", "实分析",
    # ── 哲学家 ──
    "苏格拉底", "柏拉图", "亚里士多德", "康德", "尼采", "黑格尔",
    "笛卡尔", "斯宾诺莎", "莱布尼茨", "休谟", "洛克", "卢梭", "伏尔泰",
    "马克思", "恩格斯", "海德格尔", "维特根斯坦", "罗素", "萨特",
    "孔子", "老子", "孟子", "庄子", "墨子", "荀子", "韩非子", "朱熹", "王阳明",
    # ── 物理学家 / 数学家 / 计算机 ──
    "图灵奖", "图灵", "冯·诺依曼", "P与NP问题", "深度学习", "强化学习", "递归",
    "牛顿", "伽利略", "开普勒", "霍金", "麦克斯韦", "玻尔", "薛定谔",
    "海森堡", "费曼", "狄拉克", "杨振宁", "钱学森", "邓稼先",
    "高斯", "欧拉", "黎曼", "庞加莱", "希尔伯特", "哥德尔", "拉马努金",
    "毕达哥拉斯", "欧几里得", "阿基米德", "笛卡儿",
    "达尔文", "孟德尔", "巴斯德", "弗洛伊德", "巴甫洛夫", "居里夫人",
    "特斯拉", "爱迪生", "贝尔", "诺贝尔", "瓦特",
    "香农", "巴贝奇", "高德纳", "丹尼斯·里奇", "林纳斯·托瓦兹",
    # ── 自然 / 生物 ──
    "光合作用", "DNA", "RNA", "进化论", "细胞分裂", "免疫系统",
    "基因", "染色体", "蛋白质", "氨基酸", "酶", "病毒", "细菌",
    "新陈代谢", "呼吸作用", "神经元", "突触", "激素",
    # ── 化学元素 ──
    "氢", "氦", "锂", "铍", "硼", "碳", "氮", "氧", "氟", "氖",
    "钠", "镁", "铝", "硅", "磷", "硫", "氯", "钾", "钙", "铁",
    "铜", "锌", "银", "金", "汞", "铅", "铀", "镭",
    # ── 中国朝代 / 历史 ──
    "夏朝", "商朝", "西周", "东周", "春秋", "战国", "秦朝", "汉朝",
    "三国", "晋朝", "南北朝", "隋朝", "唐朝", "宋朝", "元朝", "明朝",
    "清朝", "中华民国", "新中国",
    # ── 中国历史人物 ──
    "屈原", "司马迁", "班固", "王羲之", "李白", "杜甫", "白居易",
    "苏轼", "辛弃疾", "李清照", "陶渊明", "韩愈", "柳宗元",
    "岳飞", "文天祥", "戚继光", "郑成功", "林则徐",
    "康有为", "梁启超", "孙中山", "鲁迅", "毛泽东", "周恩来", "邓小平",
    "华罗庚", "陈景润", "袁隆平", "屠呦呦",
    "秦始皇", "汉武帝", "唐太宗", "宋太祖", "成吉思汗", "康熙帝", "乾隆帝",
    "诸葛亮", "曹操", "刘备", "关羽", "李世民",
    # ── 西方人物 / 艺术 ──
    "贝多芬", "莫扎特", "巴赫", "肖邦", "舒伯特", "勃拉姆斯",
    "毕加索", "达·芬奇", "米开朗基罗", "拉斐尔", "梵高", "莫奈",
    "莎士比亚", "歌德", "雨果", "巴尔扎克", "托尔斯泰", "陀思妥耶夫斯基",
    "海明威", "马克·吐温", "卡夫卡", "加缪",
    "拿破仑", "丘吉尔", "罗斯福", "林肯", "华盛顿", "甘地",
    # ── 国家 ──
    "中国", "美国", "英国", "法国", "德国", "俄罗斯", "日本", "印度",
    "巴西", "加拿大", "澳大利亚", "意大利", "西班牙", "葡萄牙", "希腊",
    "埃及", "土耳其", "伊朗", "韩国", "越南", "泰国", "新加坡",
    "马来西亚", "印度尼西亚", "墨西哥", "阿根廷", "南非", "瑞士", "瑞典",
    # ── 城市 ──
    "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "西安",
    "重庆", "武汉", "天津", "香港", "澳门", "台北",
    "东京", "首尔", "莫斯科", "伦敦", "巴黎", "柏林", "罗马",
    "纽约", "洛杉矶", "华盛顿", "悉尼", "迪拜",
    # ── 动物 ──
    "大熊猫", "金丝猴", "东北虎", "藏羚羊", "丹顶鹤", "朱鹮",
    "大象", "长颈鹿", "狮子", "老虎", "猎豹", "北极熊", "企鹅",
    "鲨鱼", "鲸鱼", "海豚", "章鱼", "蝴蝶", "蜜蜂", "蚂蚁", "蜘蛛",
    # ── 植物 ──
    "银杏", "水杉", "红松", "樟树", "桂花", "牡丹", "荷花",
    "梅花", "菊花", "兰花", "松树", "竹子", "柳树", "桃花", "海棠",
    "玫瑰", "郁金香", "百合", "向日葵",
    # ── 重要书籍 / 文化经典 ──
    "论语", "道德经", "孙子兵法", "史记", "资治通鉴", "周易",
    "诗经", "楚辞", "红楼梦", "西游记", "水浒传", "三国演义", "聊斋志异",
    "金瓶梅", "儒林外史", "本草纲目", "天工开物", "永乐大典", "四库全书",
    # ── 重大科技 / 事件 ──
    "互联网", "万维网", "人工智能", "图灵测试", "摩尔定律",
    "工业革命", "信息革命", "蒸汽机", "电报", "电话", "电视", "计算机",
    "登月", "阿波罗计划", "国际空间站", "中国空间站", "嫦娥工程",
]


# 词条碎片黑名单 - 这些通常是 jieba 切出来的语义残渣,百科里不会有同名词条
_TOPIC_STOPWORDS = {
    # 常见碎片前后缀
    "原指", "又名", "原名", "引申为", "首字", "被誉为", "合理安排", "代表作品",
    "性格特点", "基本特征", "基本要素", "卓越贡献", "通过审查", "广泛传播",
    "需经", "时需", "因病", "驻原", "兰原", "诗文", "言行", "事迹", "胜迹",
    "院团", "院士", "首演", "首歌曲", "申请人", "持有人", "采购人", "投资者",
    "委会", "民委", "教练员", "运动员", "委员长", "副主任", "总书记",
    # 单字组合(jieba 误切产物)
    "电视网", "广播公司", "购车", "买车", "养车", "县名",
    # 描述性虚词
    "卓越", "适应力", "对抗性", "高大挺拔", "现行标准", "技术指标",
    "基本要求", "评选活动", "重大事件", "新年贺词", "重要性",
    # 数量/时间词
    "总面积", "高可达", "首次", "本次", "本届", "全国", "国家级",
    # 方位/范围短语(实测高频垃圾,前缀过滤会误伤"中国"/"亚洲"等条目)
    "中西部", "大部地区", "东欧地区", "中文版", "知识分子", "短时间",
    "十分迅速", "极微小", "亚显微", "类节目", "专业型",
    # 形容词性 / 描述性碎片
    "灰褐色", "红褐色", "淡褐色", "圆柱形", "圆柱状", "锯齿状", "鳞片状",
    "压迫感", "灼热感", "敏锐度", "故事性", "表现形式",
    # 动作 / 状态短语
    "重新命名", "任教于", "公司出品", "法定代表", "中心地段",
    "雍正年间", "古文化", "成形术", "戏剧学院表演系",
    # "X+通用词"(看似实体实则歧义)
    "造船公司", "造船厂", "电视艺术家协会", "理论研究室", "研究局",
    "经济系", "戏剧系", "文学系", "数码音",
    # 古汉语 / 古职官残片(jieba 在历史人物文里高频切出)
    "汉魏至", "司户参", "曹参军", "天宝末", "南西川", "宋处州", "左奉议",
    "谋略家", "越王勾", "型飞机", "架飞机", "躯干部",
    # 音译错切(常规过滤抓不住,在线观察补)
    "一定量", "木科刚", "马萨诸萨州", "伯克利分校",
}

# 必须排除的前缀(以这些开头通常是修饰语)
_BAD_PREFIX = ("驻", "原", "首", "被", "副", "总", "时", "本", "全", "再",
               "本届", "又", "新", "和氏", "卓越", "高",
               # 新增:数量/方位/否定/重做 等修饰语开头
               "重新", "广泛", "合理", "十分", "高度", "极其")
# 必须排除的后缀
_BAD_SUFFIX = ("作品", "特点", "要素", "贡献", "传播", "审查", "事迹", "因素",
               "事件", "活动", "比赛", "标准", "指标", "要求", "性质", "性格",
               "时需", "需经", "者", "员", "长", "委", "原",
               # 新增:形态/范围/年代/机构 等"伪实体"高频尾字
               "状", "形", "系", "感", "度", "地区", "地段", "年间",
               "命名", "节目", "代表")


def _good_candidate(word: str, existing: set) -> bool:
    """挑词条候选:必须看起来像真实概念名,不是 jieba 切出的语义碎片。"""
    if not word or word in existing or word in _TOPIC_STOPWORDS:
        return False
    n = len(word)
    # 2 字候选歧义太多(常匹配错词条),只收 3 字以上
    if n < 3 or n > 8:
        return False
    if not all("一" <= ch <= "鿿" for ch in word):
        return False
    if word.startswith(_BAD_PREFIX):
        return False
    if word.endswith(_BAD_SUFFIX):
        return False
    return True


def ensure_blacklist_table(cur):
    """失败自愈黑名单。被记录的 topic 下次 discovery 直接跳过。
    仅写入百度百科返回 404 且其余源也都无内容的 topic(明确"实体不存在"的强信号),
    不写入 no_content / http_error / 限流类临时失败。"""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS topic_failed_blacklist (
            topic VARCHAR(255) PRIMARY KEY,
            last_status TEXT,
            fail_count INT DEFAULT 1,
            failed_at TIMESTAMP DEFAULT NOW()
        )
    """)


def record_failed_topic(topic, status):
    """仅对"实体不存在"型失败永久记忆。其他失败(no_content/http/限流)不记。"""
    if not status or "status_404" not in status:
        return
    try:
        conn = get_pg_conn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO topic_failed_blacklist (topic, last_status) "
                    "VALUES (%s, %s) "
                    "ON CONFLICT (topic) DO UPDATE SET "
                    "  fail_count = topic_failed_blacklist.fail_count + 1, "
                    "  last_status = EXCLUDED.last_status, "
                    "  failed_at = NOW()",
                    (topic[:255], status),
                )
        finally:
            conn.close()
    except Exception as e:
        LOG.debug(f"  ! blacklist write error for {topic}: {e}")


def discover_new_topics(cur, max_new=20):
    """从 thinking_gaps (认知惊奇缺口) 与 web_corpus 已学内容里发现新主题:
    1) 优先消费高惊奇度的未知概念 (thinking_gaps status='pending')
    2) 其次从已学百科文本通过 jieba 切出 3+ 字高质量实体名词
    """
    cur.execute(
        "SELECT topic FROM web_corpus "
        "UNION SELECT topic FROM sentence_corpus "
        "UNION SELECT topic FROM topic_failed_blacklist"
    )
    existing = {r[0] for r in cur.fetchall() if r[0]}

    candidates = []
    seen = set()

    # 0) 优先提取认知惊奇缺口 (Surprise-Driven Curiosity Queue)
    try:
        cur.execute(
            "SELECT query FROM thinking_gaps WHERE status='pending' "
            "ORDER BY hits DESC, last_seen DESC LIMIT 15"
        )
        for (q_txt,) in cur.fetchall():
            if not q_txt:
                continue
            cleaned = q_txt.replace("是什么", "").replace("？", "").replace("?", "").strip()
            if _good_candidate(cleaned, existing) and cleaned not in seen:
                seen.add(cleaned)
                candidates.append(cleaned)
                if len(candidates) >= max(4, max_new // 3):
                    break
    except Exception:
        pass

    cur.execute(
        "SELECT content, topic FROM web_corpus WHERE status='learned' "
        "ORDER BY RANDOM() LIMIT 50"
    )
    rows = cur.fetchall()
    contents = [r[0] for r in rows]
    learned_topics = [r[1] for r in rows if r[1]]

    # 1) jieba 切词 - 仅高质量候选
    for c in contents:
        for word, flag in psg.cut(c[:400]):
            if not flag.startswith("n"):
                continue
            if word in seen or not _good_candidate(word, existing):
                continue
            seen.add(word)
            candidates.append(word)
            if len(candidates) >= max_new:
                return candidates

    # 2) 候选不够 -> 用 jieba 在已 learned 的 topic 上再切一次(更可能产生真实词条)
    for t in learned_topics:
        if len(candidates) >= max_new:
            break
        for word, flag in psg.cut(t):
            if not flag.startswith("n"):
                continue
            if word in seen or not _good_candidate(word, existing):
                continue
            seen.add(word)
            candidates.append(word)
            if len(candidates) >= max_new:
                break
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval_min", type=float, default=2.0,
                    help="单次抓取最小间隔(秒,加随机)")
    ap.add_argument("--interval_max", type=float, default=4.0,
                    help="单次抓取最大间隔")
    ap.add_argument("--credibility", type=float, default=0.7)
    ap.add_argument("--max_per_run", type=int, default=0,
                    help="本次会话最多抓多少条;0 = 无限")
    ap.add_argument("--seed_only", action="store_true",
                    help="只用 SEED_TOPICS,不自动发现")
    ap.add_argument("--allow_browser", action="store_true",
                    help="百度百科 + 其余 HTTP 源都死掉时,启动 Playwright 浏览器兜底")
    args = ap.parse_args()

    # 浏览器兜底开关传到 web_crawler 模块的全局
    _web_crawler.ENABLE_BROWSER_FALLBACK = bool(args.allow_browser)

    LOG.info(f"[crawler] starting. interval={args.interval_min}-{args.interval_max}s "
             f"max_per_run={'inf' if args.max_per_run==0 else args.max_per_run} "
             f"browser_fallback={args.allow_browser}")
    print(f"[crawler] starting. interval={args.interval_min}-{args.interval_max}s "
          f"max_per_run={'inf' if args.max_per_run==0 else args.max_per_run}")

    # 确保失败黑名单表存在
    conn_b = get_pg_conn()
    try:
        with conn_b, conn_b.cursor() as cur:
            ensure_blacklist_table(cur)
    finally:
        conn_b.close()

    queue = deque(SEED_TOPICS)
    crawled = 0
    skipped = 0
    failed = 0

    while True:
        if args.max_per_run and crawled >= args.max_per_run:
            print(f"[crawler] reached max_per_run={args.max_per_run}, stopping.")
            break

        if not queue:
            LOG.info("[crawler] queue empty, discovering ...")
            conn_d = get_pg_conn()
            try:
                with conn_d, conn_d.cursor() as cur:
                    new = discover_new_topics(cur, max_new=20) if not args.seed_only else []
            finally:
                conn_d.close()
            if not new:
                LOG.info("[crawler] no new topics found, sleeping 60s and re-seeding")
                time.sleep(60)
                queue.extend(SEED_TOPICS)
                continue
            queue.extend(new)
            LOG.info(f"[crawler] discovered {len(new)} new topics: "
                     f"{', '.join(new[:5])}{'...' if len(new) > 5 else ''}")

        topic = queue.popleft()
        # 限速
        time.sleep(random.uniform(args.interval_min, args.interval_max))

        try:
            txt, status = fetch_baike(topic)
        except Exception as e:
            LOG.warning(f"  ! exception fetching {topic}: {e}")
            failed += 1
            continue

        if not txt:
            LOG.info(f"  - {topic}: {status}")
            failed += 1
            record_failed_topic(topic, status)
            continue

        # status 现在可能是源标签 (baike.baidu / wikipedia.zh / browser:bing ...)
        # 把它存为元数据,起码先写日志
        try:
            conn = get_pg_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        rid, was_new = insert_pending(cur, topic, txt, args.credibility)
            finally:
                conn.close()
        except Exception as e:
            LOG.warning(f"  ! db error {topic}: {e}")
            failed += 1
            continue

        if was_new:
            crawled += 1
            LOG.info(f"  + [{crawled}] id={rid} {topic} ({len(txt)} chars) <{status}>")
            # 同时尝试下载主图,若有 -> 写 vision_corpus pending
            img_path, vid = fetch_and_save_image(topic, txt)
            if img_path and vid:
                LOG.info(f"     IMG vision id={vid}  img={os.path.basename(img_path)}")
            elif img_path:
                LOG.info(f"     IMG already on disk: {os.path.basename(img_path)}")
        else:
            skipped += 1
            LOG.info(f"  = [{skipped}] {topic} (already, cred bumped)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[crawler] interrupted by user.")
