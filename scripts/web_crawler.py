"""
网页爬虫 - 从百度百科抓取主题摘要,写入 web_corpus(status=pending)。

用法:
  python scripts/web_crawler.py 量子物理 哥德尔 黎曼猜想
  python scripts/web_crawler.py --file seeds.txt
  python scripts/web_crawler.py --auto-suggest 20    (从 brain 未匹配的查询自动挖掘种子)

抓到的内容由 consolidator daemon 后续按可信度规则提升为 learned。
"""
import argparse
import os
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests
from bs4 import BeautifulSoup
from utils.db_config import get_pg_conn
from skills.multi_source_crawler import fetch_topic as _multi_fetch

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

BAIKE_URL = "https://baike.baidu.com/item/{topic}"
DEFAULT_CREDIBILITY = 0.7  # 百科:可信但非满分
DELAY_RANGE = (1.2, 2.5)   # 限速,避免被封
MIN_LEN = 30
MAX_LEN = 3000

# 单源百度百科被封 (status_403) 时,自动切到 wikipedia / moegirl / bing 等
ENABLE_MULTI_SOURCE = True
# 默认开:Playwright 装了就自动调,没装 multi_source 内部会优雅降级返回 None,
# 不影响其它 HTTP 源。daemon 不再需要显式 --allow_browser。
ENABLE_BROWSER_FALLBACK = True

# 创建独立 session,**不读** HTTP_PROXY/HTTPS_PROXY 环境变量。
# baike.baidu.com 国内直连,系统代理(Clash 7890 等)反而经常 SSL fail。
_session = requests.Session()
_session.trust_env = False   # 关键:忽略环境代理
_session.headers.update({"User-Agent": UA})

import re as _re

OG_IMAGE_RE = _re.compile(
    r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', _re.IGNORECASE
)


def _extract_image_url(html):
    """从 baike 页面 HTML 抽出主词条图 URL(og:image)。"""
    m = OG_IMAGE_RE.search(html)
    if not m:
        return None
    url = m.group(1).replace("&amp;", "&")
    # 过滤掉默认占位图(空白 / logo)
    if not url or "logo" in url or "default" in url:
        return None
    return url


def fetch_baike(topic, timeout=10, want_image=True):
    """抓"一条主题摘要"。返回 (content, status)

    旧行为是只抓百度百科;现在若 ENABLE_MULTI_SOURCE=True,百度百科失败/403
    会自动接 wikipedia.zh/baike.so/baike.sogou/moegirl/bing/...
    status 字段记录数据源,方便排查("baike.baidu" / "wikipedia.zh" / "browser:bing"...)
    """
    # ── 1. 先按原行为抓百度百科 ──
    url = BAIKE_URL.format(topic=requests.utils.quote(topic))
    txt = None
    baike_err = None
    try:
        r = _session.get(url, timeout=timeout)
        if r.status_code != 200:
            baike_err = f"status_{r.status_code}"
        else:
            soup = BeautifulSoup(r.text, "lxml")
            summary = soup.find(class_="lemma-summary") or soup.find(class_="J-summary")
            if summary:
                t = summary.get_text(strip=True, separator=" ")
            else:
                ps = [p.get_text(strip=True) for p in soup.find_all("p")
                      if len(p.get_text(strip=True)) > 50]
                t = " ".join(ps[:3])
            t = t.replace("​", "").strip()
            if t and len(t) >= MIN_LEN:
                txt = t[:MAX_LEN] if len(t) > MAX_LEN else t
                if want_image:
                    _last_image_urls[topic] = _extract_image_url(r.text)
                return txt, "baike.baidu"
            baike_err = "no_content"
    except Exception as e:
        baike_err = f"http_error: {e}"

    # ── 2. 百度百科撞墙 -> 多源兜底 ──
    if not ENABLE_MULTI_SOURCE:
        return None, baike_err or "no_content"

    try:
        # 百度百科已尝试过,排除掉避免重复请求
        res = _multi_fetch(topic, strategy="first_ok", timeout=timeout,
                           allow_browser=ENABLE_BROWSER_FALLBACK,
                           exclude_sources=["baike.baidu"])
    except Exception as e:
        return None, f"multi_error: {e}"

    if not res:
        return None, f"all_sources_failed (baike={baike_err})"

    if want_image:
        _last_image_urls[topic] = res.get("image_url")
    return res["content"], res["source"]


# 临时存放最近一次抓的 image_url(避免改 fetch_baike 的返回签名)
_last_image_urls = {}


def get_last_image_url(topic):
    return _last_image_urls.get(topic)


def download_image(url, dest_path, timeout=15):
    """下载图片到磁盘。失败返回 None,成功返回路径。"""
    try:
        r = _session.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "")
        if "image" not in ctype:
            return None
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return dest_path
    except Exception:
        return None


def insert_pending(cur, topic, content, credibility):
    """写 web_corpus,topic 唯一 -> 已存在则只 bump credibility"""
    cur.execute(
        "INSERT INTO web_corpus (topic, content, status, credibility) "
        "VALUES (%s, %s, 'pending', %s) "
        "ON CONFLICT (topic) DO UPDATE SET "
        "  credibility = LEAST(1.0, web_corpus.credibility + 0.05) "
        "RETURNING id, (xmax = 0) AS inserted",
        (topic[:255], content, credibility),
    )
    row = cur.fetchone()
    return row[0], row[1]  # id, was_new_insert


def auto_suggest_topics(cur, limit=20):
    """从过去命中失败/低分的对话中提取潜在主题"""
    # 简单策略:取最近的低分 sentence_corpus topic 中最常见的关键词候选
    cur.execute(
        "SELECT topic FROM sentence_corpus "
        "WHERE status='pending' AND credibility < 0.55 "
        "ORDER BY created_at DESC LIMIT %s", (limit,)
    )
    return [r[0] for r in cur.fetchall() if r[0] and len(r[0]) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topics", nargs="*", help="主题列表(支持多个)")
    ap.add_argument("--file", help="从文件读种子(每行一个)")
    ap.add_argument("--auto-suggest", type=int, default=0,
                    help="从 PG pending 低分主题取 N 个作为种子")
    ap.add_argument("--credibility", type=float, default=DEFAULT_CREDIBILITY)
    args = ap.parse_args()

    seeds = list(args.topics)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            seeds += [l.strip() for l in f if l.strip()]

    conn = get_pg_conn()
    if args.auto_suggest > 0:
        with conn.cursor() as cur:
            seeds += auto_suggest_topics(cur, args.auto_suggest)

    seeds = [s for s in dict.fromkeys(seeds) if s]  # dedup, keep order
    if not seeds:
        print("no seeds. usage: python web_crawler.py 主题1 主题2 ...")
        return

    print(f"[crawl] {len(seeds)} seeds")
    n_ok, n_fail, n_skip = 0, 0, 0
    for i, topic in enumerate(seeds, 1):
        time.sleep(random.uniform(*DELAY_RANGE))
        txt, status = fetch_baike(topic)
        if txt:
            try:
                with conn:
                    with conn.cursor() as cur:
                        rid, was_new = insert_pending(cur, topic, txt, args.credibility)
                if was_new:
                    print(f"  [{i}/{len(seeds)}] + id={rid} {topic} ({len(txt)} chars)")
                    n_ok += 1
                else:
                    print(f"  [{i}/{len(seeds)}] = id={rid} {topic} (already, cred bumped)")
                    n_skip += 1
            except Exception as e:
                print(f"  [{i}/{len(seeds)}] DB ERR {topic}: {e}")
                n_fail += 1
        else:
            print(f"  [{i}/{len(seeds)}] - {topic} ({status})")
            n_fail += 1

    conn.close()
    print(f"\n[done] new={n_ok} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
