"""
Playwright + Stealth 浏览器抓取 - 反指纹 + 反反爬。

技术栈:
  - playwright_stealth.Stealth      抹掉 navigator.webdriver / 自动化痕迹
  - Chromium launch args            关闭 GPU / 默认浏览器检查 / first-run
  - 真实 viewport + UA + 时区 + 地理 真实环境模拟
  - _human_warmup()                 随机鼠标移动 + 滚动,绕过行为指纹
  - cookies 持久化(可选 cookie_file)
  - 字符级 type 模拟(本模块抓取不输入,故略)

依赖:
    pip install playwright playwright-stealth
    python -m playwright install chromium

未装 -> browser_fetch() 返回 None,multi_source_crawler 优雅降级。
"""
import json
import os
import random
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional, Dict

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    _PW_OK = True
except Exception:
    _PW_OK = False

# playwright-stealth v2(Stealth class)和 v1(stealth_sync) 都兼容
_STEALTH_OK = False
_STEALTH_V2 = None  # callable: apply(page)
try:
    from playwright_stealth import Stealth as _StealthV2  # v2 API
    _STEALTH_V2 = lambda page: _StealthV2().apply_stealth_sync(page)
    _STEALTH_OK = True
except Exception:
    try:
        from playwright_stealth import stealth_sync as _stealth_sync  # v1 API
        _STEALTH_V2 = _stealth_sync
        _STEALTH_OK = True
    except Exception:
        pass

MIN_LEN = 30
MAX_LEN = 3000

# 真实 Chrome 131 UA(豆包测过能过头条系指纹)
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/131.0.0.0 Safari/537.36")

_LAUNCH_ARGS = [
    "--disable-infobars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-gpu",
    "--lang=zh-CN",
    "--window-size=1920,1080",
    # 进一步减少自动化指纹
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
]

# 北京 经纬度 / 时区,中文站会按地区返回内容
_CONTEXT_OPTS = dict(
    viewport={"width": 1920, "height": 1080},
    user_agent=_UA,
    geolocation={"latitude": 39.9042, "longitude": 116.4074},
    permissions=["geolocation"],
    timezone_id="Asia/Shanghai",
    locale="zh-CN",
)

# cookies 持久化目录(让二次抓取继承会话,跳过验证码概率高)
_COOKIE_DIR = Path(__file__).resolve().parents[1] / "logs" / "browser_cookies"
_COOKIE_DIR.mkdir(parents=True, exist_ok=True)


def _clean(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\[\d+\]", "", t)
    return t


def _trim(t: str) -> str:
    return t[:MAX_LEN] if len(t) > MAX_LEN else t


def _cookie_path(domain: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", domain)
    return _COOKIE_DIR / f"{safe}.json"


def _load_cookies(context, domain: str):
    p = _cookie_path(domain)
    if not p.exists():
        return
    try:
        cookies = json.loads(p.read_text("utf-8"))
        if cookies:
            context.add_cookies(cookies)
    except Exception:
        pass


def _save_cookies(context, domain: str):
    try:
        cookies = context.cookies()
        if not cookies:
            return
        # 过滤出该 domain 的
        keep = [c for c in cookies if domain in (c.get("domain") or "")]
        _cookie_path(domain).write_text(
            json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _human_warmup(page):
    """随机鼠标移动 + 上下小幅滚动,绕过行为指纹"""
    try:
        for _ in range(random.randint(3, 5)):
            page.mouse.move(random.randint(200, 1200), random.randint(150, 600))
            time.sleep(random.uniform(0.08, 0.18))
        page.evaluate(f"window.scrollBy(0, {random.randint(50, 200)})")
        time.sleep(random.uniform(0.1, 0.25))
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(random.uniform(0.08, 0.15))
    except Exception:
        pass


def _apply_stealth(page):
    """有 playwright-stealth 就用,没有时手动抹掉一些常见指纹"""
    if _STEALTH_OK:
        try:
            Stealth().apply_stealth_sync(page)
            return
        except Exception:
            pass
    # 手动 fallback(playwright-stealth 没装时聊胜于无)
    try:
        page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','zh','en-US','en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            window.chrome = window.chrome || { runtime: {} };
            const orig = navigator.permissions && navigator.permissions.query;
            if (orig) {
                navigator.permissions.query = (p) =>
                    p.name === 'notifications'
                        ? Promise.resolve({state: Notification.permission})
                        : orig(p);
            }
            """
        )
    except Exception:
        pass


# ── 站点抓取器 ────────────────────────────────────────────────────────────

def _try_baike_baidu(page, topic: str) -> Optional[Dict]:
    """百度百科 - http 通道 403 时这里用浏览器再试一次"""
    url = "https://baike.baidu.com/item/" + urllib.parse.quote(topic)
    try:
        page.goto(url, timeout=20000, wait_until="domcontentloaded")
        _human_warmup(page)
        node = page.query_selector(".lemma-summary, .J-summary")
        if not node:
            return None
        txt = _clean(node.inner_text())
        if len(txt) < MIN_LEN:
            return None
        img_h = page.query_selector('meta[property="og:image"]')
        img = img_h.get_attribute("content") if img_h else None
        return {
            "content": _trim(txt),
            "source": "browser:baike.baidu",
            "credibility": 0.78,
            "image_url": img,
        }
    except Exception:
        return None


def _try_wiki(page, topic: str) -> Optional[Dict]:
    url = "https://zh.wikipedia.org/wiki/" + urllib.parse.quote(topic)
    try:
        page.goto(url, timeout=20000, wait_until="domcontentloaded")
        _human_warmup(page)
        p = page.query_selector("#mw-content-text p")
        if not p:
            return None
        txt = _clean(p.inner_text())
        if len(txt) < MIN_LEN:
            return None
        img_h = page.query_selector('meta[property="og:image"]')
        img = img_h.get_attribute("content") if img_h else None
        return {
            "content": _trim(txt),
            "source": "browser:wikipedia",
            "credibility": 0.88,
            "image_url": img,
        }
    except Exception:
        return None


def _try_bing(page, topic: str) -> Optional[Dict]:
    url = "https://cn.bing.com/search?q=" + urllib.parse.quote(topic)
    try:
        page.goto(url, timeout=20000, wait_until="domcontentloaded")
        _human_warmup(page)
        page.wait_for_selector("li.b_algo", timeout=5000)
        cap = page.query_selector("li.b_algo p, li.b_algo div.b_caption")
        if not cap:
            return None
        txt = _clean(cap.inner_text())
        if len(txt) < MIN_LEN:
            return None
        return {
            "content": _trim(txt),
            "source": "browser:bing",
            "credibility": 0.55,
            "image_url": None,
        }
    except Exception:
        return None


def _try_ddg(page, topic: str) -> Optional[Dict]:
    url = "https://duckduckgo.com/?q=" + urllib.parse.quote(topic) + "&t=h_&ia=web"
    try:
        page.goto(url, timeout=20000, wait_until="domcontentloaded")
        _human_warmup(page)
        page.wait_for_selector("a[data-testid=result-title-a]", timeout=5000)
        snip = page.query_selector("div[data-result=snippet]")
        if not snip:
            return None
        txt = _clean(snip.inner_text())
        if len(txt) < MIN_LEN:
            return None
        return {
            "content": _trim(txt),
            "source": "browser:ddg",
            "credibility": 0.50,
            "image_url": None,
        }
    except Exception:
        return None


# 顺序: 百度百科(被 http 403 的最常见目标) -> wiki -> bing -> ddg
_BROWSER_ORDER = [
    (_try_baike_baidu, "baike.baidu.com"),
    (_try_wiki, "zh.wikipedia.org"),
    (_try_bing, "cn.bing.com"),
    (_try_ddg, "duckduckgo.com"),
]


def browser_fetch(topic: str, timeout: int = 20,
                  headless: bool = True) -> Optional[Dict]:
    """对外接口。Playwright 未装时返回 None。"""
    if not _PW_OK:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
            ctx = browser.new_context(**_CONTEXT_OPTS)
            page = ctx.new_page()

            # 装 stealth + 反指纹脚本 - 必须在 goto 之前
            _apply_stealth(page)

            try:
                for fn, domain in _BROWSER_ORDER:
                    _load_cookies(ctx, domain)
                    try:
                        r = fn(page, topic)
                    except Exception:
                        r = None
                    if r:
                        _save_cookies(ctx, domain)
                        return r
                    # 切站点前短停,装得像人
                    time.sleep(random.uniform(0.4, 0.9))
            finally:
                try: ctx.close()
                except Exception: pass
                try: browser.close()
                except Exception: pass
    except Exception:
        return None
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python browser_fetcher.py <topic> [--headed]")
        sys.exit(1)
    headed = "--headed" in sys.argv
    print(f"playwright={_PW_OK}  stealth={_STEALTH_OK}")
    r = browser_fetch(sys.argv[1], headless=not headed)
    if r:
        print(f"=== {r['source']}  cred={r['credibility']}  "
              f"len={len(r['content'])}  img={bool(r['image_url'])} ===")
        print(r["content"])
    else:
        print("[no result]")
