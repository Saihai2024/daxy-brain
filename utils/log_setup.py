"""
统一 daemon 日志器 - 保证每个 daemon 在自己的 logs/<name>.log 里都有产出。

历史问题: run.py 用 subprocess.Popen 把 stdout 重定向到文件,
fasttext 等 C++ 库不写 Python 的 stdout(只写到原始 fd 2 / fd 1 的 C 层缓冲),
若 daemon 在首条 print 前 import 较慢或卡住,日志文件就是 0 字节。

解决: 每个 daemon 在 main() 开头调 setup_daemon_logger("crawler"),
拿到一个会同时写文件 + 控制台的 logger,内部任何 logger.info(...) 都会落盘。
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_daemon_logger(name: str, level=logging.INFO) -> logging.Logger:
    """返回 logger,默认输出到 logs/<name>.log 与 stdout 两边。"""
    log_path = LOG_DIR / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 已配置过避免重复 add handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")

    # 文件 handler - append 模式,run.py 覆盖式重定向不影响这里
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler - 让 run.py 的 stdout 重定向也能抓到
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 启动即写一行,保证文件非空
    logger.info(f"[{name}] logger ready -> {log_path}")
    return logger
