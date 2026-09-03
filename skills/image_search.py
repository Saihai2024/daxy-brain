"""
图片搜索兜底 - 文字源(baike/wiki/bing summary)没返回 og:image 时,
专门去图片搜索引擎找一张代表图,让 vision_corpus 跟上 web_corpus 的增长。

API 优先级(均无需 key):
    1. Wikimedia Commons         免费, CC 协议, 拿主图
    2. Bing 图片搜索 (HTML 抓)    国内可达, 数量大
    3. 百度图片 search/json API   兜底

返回 (url, source) 或 (None, None)。
"""
import json
import os
import re
import urllib.parse
from typing import Optional, Tuple

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

_session = requests.Session()
_session.trust_env = False
_session.headers.update({"User-Agent": UA})

_session_intl = requests.Session()
_session_intl.headers.update({"User-Agent": UA})


# ── Wikimedia Commons ──────────────────────────────────────────────────
def _wikimedia(topic: str, timeout: int = 8) -> Optional[str]:
    """从 Commons 找一张 topic 相关的 free-use 图片"""
    try:
        api = ("https://commons.wikimedia.org/w/api.php?action=query"
               "&format=json&generator=search&gsrnamespace=6&gsrlimit=3"
               "&prop=imageinfo&iiprop=url&iiurlwidth=600"
               "&gsrsearch=" + urllib.parse.quote(topic))
        r = _session_intl.get(api, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        pages = (data.get("query") or {}).get("pages") or {}
        for _pid, page in pages.items():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url and url.lower().endswith(("jpg", "jpeg", "png", "gif", "webp", "svg")):
                return url
    except Exception:
        pass
    return None


# ── Bing 图片 ──────────────────────────────────────────────────────────
_BING_IMG_RE = re.compile(r'"murl":"([^"]+\.(?:jpg|jpeg|png|webp))"', re.IGNORECASE)


def _bing_images(topic: str, timeout: int = 8) -> Optional[str]:
    """抓 Bing 图片搜索结果第一个直链"""
    try:
        url = ("https://cn.bing.com/images/search?q="
               + urllib.parse.quote(topic) + "&form=HDRSC2&first=1")
        r = _session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        # 优先从 m="..." 的 JSON 数据里抽
        m = _BING_IMG_RE.search(r.text)
        if m:
            return m.group(1).encode("utf-8").decode("unicode_escape")
        # fallback: <img src=... > 第一个外站图
        for sm in re.finditer(r'<img[^>]+src="(https?://[^"]+\.(?:jpg|jpeg|png|webp))"',
                              r.text, re.IGNORECASE):
            u = sm.group(1)
            if "bing.com" in u or "rms-prod" in u:
                continue
            return u
    except Exception:
        pass
    return None


# ── 百度图片 ──────────────────────────────────────────────────────────
def _baidu_images(topic: str, timeout: int = 8) -> Optional[str]:
    """百度图片 acjson 接口拿第一张"""
    try:
        url = ("https://image.baidu.com/search/acjson?tn=resultjson_com"
               "&logid=11&ipn=rj&ct=201326592&fp=result&queryWord=" + urllib.parse.quote(topic)
               + "&word=" + urllib.parse.quote(topic) + "&pn=0&rn=10")
        r = _session.get(url, timeout=timeout,
                         headers={"Referer": "https://image.baidu.com/"})
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except Exception:
            # 有时是非严格 JSON,自己取关键字段
            m = re.search(r'"thumbURL":"(https?://[^"]+)"', r.text)
            return m.group(1) if m else None
        for item in data.get("data", []):
            for k in ("hoverURL", "middleURL", "thumbURL", "objURL"):
                u = item.get(k)
                if u and u.startswith("http"):
                    return u
    except Exception:
        pass
    return None


# ── 顶层入口 ──────────────────────────────────────────────────────────
def search_image(topic: str, timeout: int = 8) -> Tuple[Optional[str], Optional[str]]:
    """按优先级试三个图源,返回 (url, source) 或 (None, None)"""
    for fn, name in (
        (_wikimedia, "wikimedia"),
        (_bing_images, "bing_images"),
        (_baidu_images, "baidu_images"),
    ):
        try:
            u = fn(topic, timeout=timeout)
        except Exception:
            u = None
        if u:
            return u, name
    return None, None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python image_search.py <topic>")
        sys.exit(1)
    u, src = search_image(sys.argv[1])
    print(f"{src or '[fail]'}: {u or '[none]'}")
