"""
多源百科 / 搜索爬虫 - 把"模型自主上网学习"的能力从单一百度百科扩展到全网

每个 fetch_xxx() 返回 dict:
    {"content": str, "source": str, "credibility": float, "image_url": str | None}
失败返回 None。

顶层入口:
    fetch_topic(topic, strategy='vote', allow_browser=False)
        strategy='first_ok'  : 顺序尝试,第一个 ok 就返回
        strategy='vote'      : 全部尝试,挑可信度*长度 最高的一条
        strategy='multi'     : 返回 list[dict],调用方自己融合

可信度参考(主观先验,可被 immune_system 二次评估覆盖):
    维基百科 (官方 REST API)       0.90
    arXiv 论文摘要                  0.92
    百度百科                        0.70
    360百科 / 搜狗百科              0.65
    萌娘百科 (二次元/流行文化)      0.65
    Bing / DuckDuckGo 搜索摘要      0.55
    Playwright 渲染的任意网页        0.50
"""
import json
import os
import re
import time
import random
import urllib.parse
from typing import Optional, List, Dict

import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 国内站点直连(忽略系统代理 Clash 等,否则常 SSL fail)
_session = requests.Session()
_session.trust_env = False
_session.headers.update({"User-Agent": UA})

# 国外站点走本机代理(墙)。
# 不能用 trust_env:Windows 会把注册表 IE 代理解析成 https://127.0.0.1:7890,
# 对 Clash 的明文 HTTP 端口发 TLS 握手直接 SSL 失败,所有国际源全灭。
# 显式给 http:// scheme 才走得通;CRAWLER_INTL_PROXY 置空可改直连。
_INTL_PROXY = os.getenv("CRAWLER_INTL_PROXY", "http://127.0.0.1:7890")
_session_intl = requests.Session()
_session_intl.trust_env = False
_session_intl.headers.update({"User-Agent": UA})
if _INTL_PROXY:
    _session_intl.proxies = {"http": _INTL_PROXY, "https": _INTL_PROXY}

MIN_LEN = 30
MAX_LEN = 3000

OG_IMAGE_RE = re.compile(
    r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', re.IGNORECASE)

# 限流冷却:某个源遇到 429/403 后,接下来 N 秒内不再调它
_COOLDOWN_SECONDS = 300
_cooldown_until: dict = {}  # source_name -> epoch_ts


def _on_cooldown(name: str) -> bool:
    until = _cooldown_until.get(name)
    return until is not None and time.time() < until


def _trigger_cooldown(name: str, seconds: int = _COOLDOWN_SECONDS):
    _cooldown_until[name] = time.time() + seconds


def _clean(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"\[\d+\]", "", txt)           # [12] 引注
    txt = re.sub(r"\[[^\]]{1,30}\]", "", txt)   # [编辑] [来源请求]
    txt = txt.replace("​", "").replace("\xa0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _trim(txt: str) -> str:
    if len(txt) > MAX_LEN:
        return txt[:MAX_LEN]
    return txt


def _extract_og_image(html: str) -> Optional[str]:
    m = OG_IMAGE_RE.search(html)
    if not m:
        return None
    url = m.group(1).replace("&amp;", "&")
    if not url or "logo" in url or "default" in url:
        return None
    return url


# ──────────────────────────────────────────────────────────────
# Source 1: 百度百科
# ──────────────────────────────────────────────────────────────
def fetch_baike_baidu(topic: str, timeout: int = 8) -> Optional[Dict]:
    if _on_cooldown("baike.baidu"):
        return None
    url = f"https://baike.baidu.com/item/{urllib.parse.quote(topic)}"
    try:
        r = _session.get(url, timeout=timeout)
        if r.status_code == 403:
            # 触发 IP 反爬 - 冷却 5 分钟,期间走其他源(尤其 browser)
            _trigger_cooldown("baike.baidu")
            return None
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        node = soup.find(class_="lemma-summary") or soup.find(class_="J-summary")
        if node:
            txt = node.get_text(strip=True, separator=" ")
        else:
            ps = [p.get_text(strip=True) for p in soup.find_all("p")
                  if len(p.get_text(strip=True)) > 50]
            txt = " ".join(ps[:3])
        txt = _clean(txt)
        if len(txt) < MIN_LEN:
            return None
        return {
            "content": _trim(txt),
            "source": "baike.baidu",
            "credibility": 0.70,
            "image_url": _extract_og_image(r.text),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 2: 中文维基百科 (官方 REST API,稳定且无需 HTML 解析)
# ──────────────────────────────────────────────────────────────
def fetch_wikipedia_zh(topic: str, timeout: int = 8) -> Optional[Dict]:
    if _on_cooldown("wikipedia.zh"):
        return None
    try:
        # 用 REST 摘要接口
        url = ("https://zh.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(topic))
        r = _session_intl.get(url, timeout=timeout)
        if r.status_code == 429:
            # 限流 -> 冷却 5 分钟
            _trigger_cooldown("wikipedia.zh")
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        extract = _clean(data.get("extract") or "")
        if len(extract) < MIN_LEN:
            return None
        img = (data.get("thumbnail") or {}).get("source") \
              or (data.get("originalimage") or {}).get("source")
        return {
            "content": _trim(extract),
            "source": "wikipedia.zh",
            "credibility": 0.90,
            "image_url": img,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 3: 英文维基百科 (摘要 + 简单翻译降级,科学/技术词条覆盖最全)
# ──────────────────────────────────────────────────────────────
def fetch_wikipedia_en(topic: str, timeout: int = 8) -> Optional[Dict]:
    if _on_cooldown("wikipedia.en"):
        return None
    try:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(topic))
        r = _session_intl.get(url, timeout=timeout)
        if r.status_code == 429:
            _trigger_cooldown("wikipedia.en")
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        extract = _clean(data.get("extract") or "")
        if len(extract) < MIN_LEN:
            return None
        img = (data.get("thumbnail") or {}).get("source") \
              or (data.get("originalimage") or {}).get("source")
        return {
            "content": _trim(extract),
            "source": "wikipedia.en",
            # 英文内容对中文模型价值打折
            "credibility": 0.80,
            "image_url": img,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 4: 萌娘百科 (zh.moegirl.org.cn) - ACGN / 二次元 / 流行文化
# 使用 MediaWiki API,跟 Wikipedia 一样
# ──────────────────────────────────────────────────────────────
def fetch_moegirl(topic: str, timeout: int = 8) -> Optional[Dict]:
    try:
        url = ("https://zh.moegirl.org.cn/api.php?action=query&format=json"
               "&prop=extracts&exintro=1&explaintext=1&redirects=1"
               "&titles=" + urllib.parse.quote(topic))
        r = _session_intl.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _pid, page in pages.items():
            extract = _clean(page.get("extract") or "")
            if len(extract) < MIN_LEN:
                continue
            return {
                "content": _trim(extract),
                "source": "moegirl",
                "credibility": 0.65,
                "image_url": None,
            }
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
# Source 5: 360百科 (baike.so.com)
# ──────────────────────────────────────────────────────────────
_BAIKE_SO_HOMEPAGE_FINGERPRINT = "360百科提供百科知识的搜索"


def fetch_baike_so(topic: str, timeout: int = 8) -> Optional[Dict]:
    """360 百科。三个坑要躲:
    1) topic 不存在时返 homepage (HTTP 200),内容是通用 SEO 文案 —— 用指纹拒绝
    2) meta description 通常是 SEO 短标语,信息密度低 —— 不再 fallback 到它
    3) 标题里必须出现 topic 关键字,否则是错配页面
    """
    try:
        url = f"https://baike.so.com/doc/{urllib.parse.quote(topic)}"
        r = _session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        # 软 404 指纹
        if _BAIKE_SO_HOMEPAGE_FINGERPRINT in r.text[:5000]:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        # 标题必须包含 topic
        title = (soup.title.get_text(strip=True) if soup.title else "")
        if topic not in title:
            return None
        # 优先按结构化摘要找,找不到拼段落
        node = (soup.find(class_="summary")
                or soup.find(id="js-card-content")
                or soup.find(class_="abstract")
                or soup.find(class_="card-content"))
        if node is not None:
            txt = node.get_text(" ", strip=True)
        else:
            ps = [p.get_text(strip=True) for p in soup.find_all("p")
                  if len(p.get_text(strip=True)) > 40]
            txt = " ".join(ps[:3])
        txt = _clean(txt)
        # 真词条通常 > 80 字;低于该值就是页头噪声
        if len(txt) < 80:
            return None
        return {
            "content": _trim(txt),
            "source": "baike.so",
            "credibility": 0.65,
            "image_url": _extract_og_image(r.text),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 6: 搜狗百科 (baike.sogou.com)
# ──────────────────────────────────────────────────────────────
_BAIKE_SOGOU_HOMEPAGE_FINGERPRINT = "搜狗百科是一部全面"


def fetch_baike_sogou(topic: str, timeout: int = 8) -> Optional[Dict]:
    """搜狗百科。同样过滤软 404 / 标题错配 / SEO 短文。"""
    try:
        search = ("https://baike.sogou.com/Search.e?sp=S&sp=126&query="
                  + urllib.parse.quote(topic))
        r = _session.get(search, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return None
        if _BAIKE_SOGOU_HOMEPAGE_FINGERPRINT in r.text[:5000]:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        title = (soup.title.get_text(strip=True) if soup.title else "")
        if topic not in title:
            return None
        node = (soup.find(class_="abstract_content")
                or soup.find(class_="summary"))
        if node is None:
            ps = [p.get_text(strip=True) for p in soup.find_all("p")
                  if len(p.get_text(strip=True)) > 40]
            txt = " ".join(ps[:3])
        else:
            txt = node.get_text(" ", strip=True)
        txt = _clean(txt)
        if len(txt) < 80:
            return None
        return {
            "content": _trim(txt),
            "source": "baike.sogou",
            "credibility": 0.65,
            "image_url": _extract_og_image(r.text),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 7: Bing 网页搜索 - 抽第一条非广告摘要
# ──────────────────────────────────────────────────────────────
def fetch_bing(topic: str, timeout: int = 8) -> Optional[Dict]:
    try:
        url = ("https://cn.bing.com/search?q=" + urllib.parse.quote(topic)
               + "&ensearch=0")
        r = _session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        # 跳过广告,挑第一条 organic 摘要
        for li in soup.select("li.b_algo"):
            cap = li.find("p") or li.find("div", class_="b_caption")
            if cap is None:
                continue
            txt = _clean(cap.get_text(" ", strip=True))
            if len(txt) >= MIN_LEN:
                return {
                    "content": _trim(txt),
                    "source": "bing",
                    "credibility": 0.55,
                    "image_url": None,
                }
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
# Source 8: arXiv 论文摘要 - 给"前沿/学术"类查询用
# 仅在 topic 看起来像英文学术词时调用
# ──────────────────────────────────────────────────────────────
def fetch_arxiv(topic: str, timeout: int = 8) -> Optional[Dict]:
    try:
        url = ("http://export.arxiv.org/api/query?search_query=all:"
               + urllib.parse.quote(topic)
               + "&start=0&max_results=1")
        r = _session_intl.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "lxml-xml")
        entry = soup.find("entry")
        if entry is None:
            return None
        title = _clean(entry.title.get_text() if entry.title else "")
        summary = _clean(entry.summary.get_text() if entry.summary else "")
        if len(summary) < MIN_LEN:
            return None
        # 给 brain 看到的内容: 标题 + 摘要
        content = (f"{title} - {summary}" if title else summary)
        return {
            "content": _trim(content),
            "source": "arxiv",
            "credibility": 0.92,
            "image_url": None,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 9: DuckDuckGo Instant Answer API - 免费 / 无 key
# ──────────────────────────────────────────────────────────────
def fetch_ddg_instant(topic: str, timeout: int = 8) -> Optional[Dict]:
    try:
        url = ("https://api.duckduckgo.com/?format=json&no_html=1"
               "&skip_disambig=1&q=" + urllib.parse.quote(topic))
        r = _session_intl.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        abstract = _clean(data.get("AbstractText") or "")
        if len(abstract) < MIN_LEN:
            # fallback: definition
            abstract = _clean(data.get("Definition") or "")
            if len(abstract) < MIN_LEN:
                return None
        img = data.get("Image")
        if img and not img.startswith("http"):
            img = "https://duckduckgo.com" + img
        return {
            "content": _trim(abstract),
            "source": "duckduckgo",
            "credibility": 0.60,
            "image_url": img,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Source 10: Playwright 浏览器渲染 - 兜底,可关
# ──────────────────────────────────────────────────────────────
def fetch_via_browser(topic: str, timeout: int = 15) -> Optional[Dict]:
    """让模型真正"开个浏览器"上网。Playwright 未装则直接返回 None。"""
    try:
        from skills.browser_fetcher import browser_fetch
    except Exception:
        return None
    return browser_fetch(topic, timeout=timeout)


# ──────────────────────────────────────────────────────────────
# 顶层分发器
# ──────────────────────────────────────────────────────────────

# 按"先国内 / 先高可信" 排序;越靠前越优先
DEFAULT_ORDER = [
    fetch_baike_baidu,
    fetch_wikipedia_zh,
    fetch_baike_so,
    fetch_baike_sogou,
    fetch_moegirl,
    fetch_wikipedia_en,
    fetch_ddg_instant,
    fetch_bing,
]


def _looks_english(topic: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9 \-_:,\.]+", topic or ""))


def _build_order(topic: str) -> List:
    order = list(DEFAULT_ORDER)
    if _looks_english(topic):
        # 英文词条优先英文 Wiki / arXiv
        order = [fetch_wikipedia_en, fetch_arxiv, fetch_ddg_instant,
                 fetch_wikipedia_zh, fetch_baike_baidu, fetch_bing]
    return order


def fetch_topic(topic: str,
                strategy: str = "first_ok",
                allow_browser: bool = False,
                timeout: int = 8,
                delay_between: float = 0.4,
                exclude_sources=None) -> Optional[Dict]:
    """统一抓取入口。

    strategy:
        first_ok : 顺序找,首个返回即用 (生产用,快)
        vote     : 取多个,挑 credibility * log(len) 最高的
        multi    : 返回所有非空结果 (list)

    allow_browser: True 时,常规源全空才会启动 Playwright 浏览器。
    exclude_sources: 跳过的 source 标签(如 ["baike.baidu"]),用于上层已试过的源。
    """
    excl = set(exclude_sources or [])
    order = _build_order(topic)
    results: List[Dict] = []

    for fn in order:
        # source 标签由函数自己写入返回 dict;粗略按函数名匹配跳过
        if "baike.baidu" in excl and fn is fetch_baike_baidu:
            continue
        if "wikipedia.zh" in excl and fn is fetch_wikipedia_zh:
            continue
        if "wikipedia.en" in excl and fn is fetch_wikipedia_en:
            continue
        try:
            res = fn(topic, timeout=timeout)
        except Exception:
            res = None
        if res:
            results.append(res)
            if strategy == "first_ok":
                return res
        # 限速,避免连撞同 ISP 出口
        time.sleep(random.uniform(0, delay_between))

    if not results and allow_browser:
        br = fetch_via_browser(topic, timeout=max(timeout, 15))
        if br:
            results.append(br)
            if strategy == "first_ok":
                return br

    if strategy == "multi":
        return results or None
    if strategy == "vote" and results:
        import math
        return max(results,
                   key=lambda d: d["credibility"] * math.log(len(d["content"]) + 1))
    return None


def fetch_multi(topic: str, **kw) -> List[Dict]:
    """便利封装: 永远返回 list,空则 []"""
    r = fetch_topic(topic, strategy="multi", **kw)
    return r or []


def download_image(url: str, dest_path: str, timeout: int = 15) -> Optional[str]:
    """通用图片下载,与 web_crawler.download_image 行为一致。"""
    try:
        sess = _session_intl if url.startswith(("https://upload.wikimedia",
                                                 "https://en.", "https://zh."))\
                              else _session
        r = sess.get(url, timeout=timeout, stream=True)
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


# ──────────────────────────────────────────────────────────────
# CLI: 调试单个 topic 用
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--strategy", default="multi",
                    choices=["first_ok", "vote", "multi"])
    ap.add_argument("--browser", action="store_true",
                    help="兜底启动 Playwright 浏览器")
    args = ap.parse_args()

    res = fetch_topic(args.topic, strategy=args.strategy,
                      allow_browser=args.browser)
    if not res:
        print("[no result]"); sys.exit(1)
    if isinstance(res, list):
        for d in res:
            print(f"\n=== {d['source']}  cred={d['credibility']}  "
                  f"len={len(d['content'])}  img={bool(d['image_url'])} ===")
            print(d['content'][:400])
    else:
        print(f"=== {res['source']}  cred={res['credibility']} ===")
        print(res['content'])
