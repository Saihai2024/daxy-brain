"""
仪表盘登录鉴权:
  - 账号密码:环境变量 DASH_USER / DASH_PASS
  - 会话:HMAC-SHA256 签名的 Cookie token(user|expiry|sig),无需额外依赖
  - 签名密钥持久化在 webapp/.session_secret,重启服务不掉登录态

⚠️ 没配 DASH_PASS 时会随机生成一个一次性口令并打印到启动日志。
   这是刻意的:仪表盘能读写整个语料库和大脑,绝不能有"人尽皆知的默认密码"。
"""
import hmac
import hashlib
import os
import secrets
import time
from pathlib import Path
from utils.db_config import load_dotenv_if_present

load_dotenv_if_present()

AUTH_USER = os.environ.get("DASH_USER", "root")
AUTH_PASS = os.environ.get("DASH_PASS", "")
if not AUTH_PASS:
    AUTH_PASS = secrets.token_urlsafe(18)
    print(f"[auth] 未设置 DASH_PASS,本次启动的随机口令:{AUTH_USER} / {AUTH_PASS}")

COOKIE_NAME = "brain_session"
SESSION_TTL = 7 * 24 * 3600  # 7 天

_SECRET_PATH = Path(__file__).parent / ".session_secret"


def _get_secret() -> bytes:
    if _SECRET_PATH.is_file():
        data = _SECRET_PATH.read_bytes().strip()
        if len(data) >= 32:
            return data
    data = secrets.token_hex(32).encode()
    _SECRET_PATH.write_bytes(data)
    return data


_SECRET = _get_secret()


def _sign(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    ok_user = hmac.compare_digest(username.encode(), AUTH_USER.encode())
    ok_pass = hmac.compare_digest(password.encode(), AUTH_PASS.encode())
    return ok_user and ok_pass


def make_token(username: str) -> str:
    expiry = int(time.time()) + SESSION_TTL
    payload = f"{username}|{expiry}"
    return f"{payload}|{_sign(payload)}"


def verify_token(token) -> bool:
    if not token or not isinstance(token, str):
        return False
    parts = token.rsplit("|", 1)
    if len(parts) != 2:
        return False
    payload, sig = parts
    if not hmac.compare_digest(_sign(payload), sig):
        return False
    try:
        _, expiry = payload.split("|", 1)
        return int(expiry) > time.time()
    except ValueError:
        return False
