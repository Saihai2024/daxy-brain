"""
认知大脑系统一键启动器
======================

启动顺序:
  1. PostgreSQL(先 ping 端口 5432,挂了就 pg_ctl start,需配 PGSQL_BIN/PGSQL_DATA)
  2. gRPC 大脑服务 (:9999)
  3. FastAPI 仪表盘 (:8000)
  4. consolidator daemon(每 30s 巩固 pending → learned)
  5. crawler daemon(持续抓 baike + 图片)
  6. training daemon(每 120s 重训 fasttext)

用法:
  python run.py                  # 启动全部(爬虫默认多源:baike+wiki+360+sogou+moegirl+bing+arxiv+ddg)
  python run.py --no-crawl       # 不启爬虫
  python run.py --no-train       # 不启训练
  python run.py --browser-crawl  # 爬虫额外启用 Playwright 浏览器兜底(需先装 playwright)
  python run.py status           # 看当前进程状态
  python run.py stop             # 停所有(只能停本脚本启的)

按 Ctrl+C 优雅停止所有进程。
"""
import argparse
import io
import os
import signal
import socket
import subprocess
import sys
import time
import builtins
from pathlib import Path

# ---- 强制 UTF-8 stdout(兼容 PyCharm / cmd / PowerShell)----
def _setup_stdout():
    for stream_name in ("stdout", "stderr"):
        s = getattr(sys, stream_name, None)
        if s is None: continue
        # 1) Python 3.7+ TextIOWrapper.reconfigure(优先)
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
            continue
        except (AttributeError, ValueError, OSError):
            pass
        # 2) buffer-based 重新包装(老 Python / 普通 stdout)
        try:
            buf = s.buffer
            setattr(sys, stream_name,
                    io.TextIOWrapper(buf, encoding="utf-8", errors="replace",
                                     line_buffering=True))
        except (AttributeError, ValueError):
            # PyCharm 的 stdout 没 buffer — 不动它,但下面 patch print 强制 flush
            pass
_setup_stdout()

# 让所有 print 默认 flush(PyCharm 控制台靠这个出文字)
_orig_print = builtins.print
def _print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _orig_print(*args, **kwargs)
builtins.print = _print_flush

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 全部可用环境变量覆盖(见 .env.example);默认取当前解释器 / 交给 PATH。
PYTHON = os.environ.get("BRAIN_PYTHON") or sys.executable
# 思维链守护跑在另一套依赖里(DrissionPage 等),脚本也在本仓库之外,
# 所以默认不启用 —— 只有同时配了 THINKING_PYTHON 和 THINKING_DAEMON 才会拉起。
THINKING_PYTHON = os.environ.get("THINKING_PYTHON", "")
THINKING_DAEMON = os.environ.get("THINKING_DAEMON", "")
PGSQL_BIN = os.environ.get("PGSQL_BIN", "")
PGSQL_DATA = os.environ.get("PGSQL_DATA", "")
FFMPEG_DIR = os.environ.get("FFMPEG_DIR", "")


# 服务清单
SERVICES = [
    {
        "name": "grpc",
        "label": "gRPC 大脑",
        "cmd": [PYTHON, "-u", "grpc_predict.py"],
        "port": 9999,
        "ready_marker": "listening on :9999",
    },
    {
        "name": "fastapi",
        "label": "FastAPI 仪表盘",
        "cmd": [PYTHON, "-m", "uvicorn", "webapp.app:app",
                "--host", "127.0.0.1", "--port", "8000",
                "--log-level", "info"],
        "port": 8000,
        "ready_marker": "Application startup complete",
    },
    {
        "name": "consolidator",
        "label": "巩固守护",
        # 30s 太勤,每次都重 load brain.pkl 写盘,跟 grpc reload 串成颠簸。
        # 10 分钟一次足够把 pending 推进 brain。
        "cmd": [PYTHON, "-u", "scripts/consolidator_daemon.py", "--interval", "600"],
        "port": None,
        "ready_marker": "[daemon] starting",
    },
    {
        "name": "crawler",
        "label": "网页 + 图片爬虫",
        # 1.5-3s 抓一次太密,CPU/网络都占。30-60s 仍能持续学习。
        "cmd": [PYTHON, "-u", "scripts/crawler_daemon.py",
                "--interval_min", "30", "--interval_max", "60"],
        "port": None,
        "ready_marker": "[crawler] starting",
        "skip_if": "no_crawl",
    },
    {
        "name": "training",
        "label": "fasttext 重训",
        # 30 分钟一轮;线程封顶后 epoch 25 也只要 ~30s,精度比 10 轮更稳。
        "cmd": [PYTHON, "-u", "scripts/training_daemon.py",
                "--interval", "1800", "--epoch", "25"],
        "port": None,
        "ready_marker": "[training] starting",
        "skip_if": "no_train",
    },
    {
        "name": "thinking",
        "label": "思维链持续学习",
        # 让大脑「知道自己不会什么就去学」:把思维逻辑路答不上的任务类提问
        # (PG thinking_gaps)拿去问 Qwen,解析成拆解步骤+工具计划入 thinking_corpus。
        # ⚠️ 它跑在另一套 Python 环境、脚本也在本仓库之外(两个项目都有 utils 包,
        #    不能混 sys.path),所以要配 THINKING_PYTHON + THINKING_DAEMON 才会启用。
        # 2 分钟一轮 × 9 条 × 3 路并发。
        # ★ 并发 3 是实测出来的天花板,别再往上加:
        #     并发3 → 失败率 17-33%,~27 条/小时
        #     并发6 → 失败率 50%(日志实锤"3 次尝试里 2 次撞验证码"),仅 ~34 条/小时
        #   多花一倍请求只换 1.26 倍产出,不划算。单 IP 的量在这了,
        #   真要再快只能换百炼 API(官方接口没网页版风控)。
        # 守护每轮打印失败率,≥40% 会告警。
        # ★ 题源只有「真实提问的缺口」+「Qwen 自生成题目」,
        #   **不碰我们自己的对话记录**(那是评测集,见 thinking_daemon 顶部注释)。
        "cmd": [THINKING_PYTHON, "-u", THINKING_DAEMON,
                "--interval", "120", "--per-round", "9", "--concurrency", "3"],
        "port": None,
        "ready_marker": "[daemon] starting",
        "skip_if": "no_thinking",
        # 没配外部环境就直接跳过,不当成故障
        "require": lambda: bool(THINKING_PYTHON and THINKING_DAEMON),
    },
]


def is_port_open(port, host="127.0.0.1", timeout=0.5):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def find_pid_on_port(port):
    """Windows-only: 查 port 的占用 PID(只看 LISTENING)"""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        # TCP  127.0.0.1:8000  0.0.0.0:0  LISTENING  PID
        if len(parts) >= 5 and "LISTENING" in line and (f":{port}" in parts[1]):
            try:
                return int(parts[-1])
            except ValueError:
                continue
    return None


def kill_pid(pid):
    if not pid: return False
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True,
                       text=True, timeout=5)
        return True
    except Exception:
        return False


def precheck_ports(reset=False):
    """检查 8000 / 9999 是否被占。如 reset=True 杀掉旧进程。"""
    conflicts = []
    for port in (8000, 9999):
        if is_port_open(port):
            pid = find_pid_on_port(port)
            conflicts.append((port, pid))
    if not conflicts:
        return True
    print("\n⚠ 检测到端口已被占用:")
    for port, pid in conflicts:
        print(f"    :{port}  pid={pid}")
    if reset:
        print("\n  → --reset 模式:杀掉旧进程")
        for port, pid in conflicts:
            if pid and kill_pid(pid):
                print(f"    ✓ killed pid={pid}")
        time.sleep(2)
        # 重新检查
        still = [p for p in (8000, 9999) if is_port_open(p)]
        if still:
            print(f"  ✗ 仍占用: {still}")
            return False
        return True
    print("\n  解决方案:")
    print("    1) 重跑 run.py 加 --reset 自动杀旧进程")
    print("    2) 手动 taskkill /F /PID <pid>")
    return False


def ensure_postgres():
    """确认 PG 在 :5432 上跑;不在就 pg_ctl start"""
    if is_port_open(5432):
        print("  ✓ PostgreSQL 已在 :5432")
        return True
    if not (PGSQL_BIN and PGSQL_DATA):
        print("  ✗ PG 没在跑,且未配齐 PGSQL_BIN / PGSQL_DATA —— 请先自行启动 PostgreSQL")
        return False
    print("  → PostgreSQL 未启动,执行 pg_ctl start ...")
    pg_ctl = Path(PGSQL_BIN) / "pg_ctl.exe"
    if not pg_ctl.is_file():
        print(f"  ✗ 找不到 {pg_ctl}")
        return False
    log = LOG_DIR / "postgres.log"
    try:
        subprocess.run(
            [str(pg_ctl), "-D", PGSQL_DATA, "-l", str(log), "start"],
            check=True, timeout=20,
        )
    except Exception as e:
        print(f"  ✗ pg_ctl 启动失败: {e}")
        return False
    # 等端口起
    for _ in range(20):
        if is_port_open(5432):
            print("  ✓ PostgreSQL 已就绪")
            return True
        time.sleep(0.5)
    print("  ✗ PostgreSQL 启动 10s 超时")
    return False


def make_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # 关键:直接用 env 里的 python.exe(未 activate)时,conda 环境的
    # Library\bin 不在 PATH 上,numpy 的 BLAS DLL 延迟加载会失败,
    # 矩阵乘法直接段错误(Windows 0xC06D007F = ERROR_MOD_NOT_FOUND)。
    # 这里手动把 conda env 的 DLL 目录注入 PATH,等价于 activate。
    conda_env_root = Path(PYTHON).parent
    conda_dll_dirs = [
        conda_env_root,
        conda_env_root / "Library" / "mingw-w64" / "bin",
        conda_env_root / "Library" / "usr" / "bin",
        conda_env_root / "Library" / "bin",
        conda_env_root / "Scripts",
        conda_env_root / "bin",
    ]
    prefix = os.pathsep.join(str(p) for p in conda_dll_dirs if p.is_dir())
    if prefix:
        env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
    if FFMPEG_DIR and Path(FFMPEG_DIR).is_dir():
        env["PATH"] = FFMPEG_DIR + os.pathsep + env.get("PATH", "")
    return env


# 每个服务的 BLAS/OMP 线程上限。64 核机器上 numpy/MKL/fasttext 默认
# 每进程开 64 线程,5 个进程同时矩阵运算 = 300+ 线程互踩(系统卡顿主因)。
# grpc 是前台延迟敏感路径给最多;daemon 给少,靠低优先级 + 长间隔慢慢跑。
SERVICE_THREADS = {
    "grpc": 8,
    "fastapi": 4,
    "consolidator": 4,
    "crawler": 2,
    "training": 8,   # fasttext 在 daemon 内部另有 thread=8 cap
}
# 后台 daemon 用低优先级,绝不和前台抢 CPU
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
DAEMON_SERVICES = {"consolidator", "crawler", "training"}


def start_service(svc, env):
    log_path = LOG_DIR / f"{svc['name']}.log"
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    print(f"  → 启动 {svc['label']:20s}  log={log_path.name}")
    env = dict(env)
    n_threads = str(SERVICE_THREADS.get(svc["name"], 4))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env[var] = n_threads
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    if os.name == "nt" and svc["name"] in DAEMON_SERVICES:
        flags |= BELOW_NORMAL_PRIORITY_CLASS
    p = subprocess.Popen(
        svc["cmd"], cwd=str(ROOT), env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    svc["proc"] = p
    svc["log_f"] = log_f
    svc["log_path"] = log_path
    return p


def wait_for_ready(svc, timeout=120):
    """等服务就绪(看 log marker 或端口)"""
    if not svc.get("ready_marker") and not svc.get("port"):
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        # 端口 open 算 OK
        if svc.get("port") and is_port_open(svc["port"]):
            return True
        # log marker
        if svc.get("ready_marker"):
            try:
                txt = svc["log_path"].read_text(encoding="utf-8", errors="replace")
                if svc["ready_marker"] in txt:
                    return True
            except Exception:
                pass
        # 进程死了
        if svc["proc"].poll() is not None:
            print(f"  ✗ {svc['label']} 启动失败,退出码 {svc['proc'].returncode}")
            print(f"    日志: {svc['log_path']}")
            return False
        time.sleep(0.5)
    print(f"  ⚠ {svc['label']} 等待就绪超时(已启动,但 marker 没出现)")
    return True  # 不阻塞后续启动


def stop_all(services, prefix="  →"):
    for svc in services:
        p = svc.get("proc")
        if not p or p.poll() is not None:
            continue
        print(f"{prefix} 停止 {svc['label']}(pid={p.pid})")
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                p.terminate()
        except Exception as e:
            print(f"    signal 失败: {e}")
    # 等 5s
    deadline = time.time() + 5
    while time.time() < deadline:
        if all((s.get("proc") is None or s["proc"].poll() is not None) for s in services):
            break
        time.sleep(0.2)
    # 残留的强杀
    for svc in services:
        p = svc.get("proc")
        if p and p.poll() is None:
            print(f"  ✗ 强杀 {svc['label']}(pid={p.pid})")
            try: p.kill()
            except Exception: pass
        if svc.get("log_f"):
            try: svc["log_f"].close()
            except Exception: pass


def cmd_status():
    print("\n=== 服务状态 ===")
    for svc in SERVICES:
        port = svc.get("port")
        if port:
            ok = is_port_open(port)
            print(f"  {svc['label']:20s} :{port} {'✓ 在线' if ok else '✗ 离线'}")
        else:
            print(f"  {svc['label']:20s} (后台 daemon,无端口)")
    print(f"  PostgreSQL          :5432 {'✓ 在线' if is_port_open(5432) else '✗ 离线'}")


def cmd_stop():
    """通过端口标识尝试杀(无脚本上下文时不可靠,推荐直接 Ctrl+C 启动脚本)"""
    print("提示:run.py 启动的进程请用 Ctrl+C 在前台脚本里停止;")
    print("如果脚本已退出,请用任务管理器结束 python.exe / postgres.exe。")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("action", nargs="?", default="run",
                    choices=["run", "status", "stop"])
    ap.add_argument("--no-crawl", action="store_true", help="不启动爬虫")
    ap.add_argument("--no-train", action="store_true", help="不启动 fasttext 重训")
    ap.add_argument("--no-thinking", action="store_true",
                    help="不启动思维链持续学习守护(它会调 qwen,想省资源时可关)")
    ap.add_argument("--no-pg", action="store_true", help="不检查 PostgreSQL")
    ap.add_argument("--reset", action="store_true", help="启动前杀掉占用 8000/9999 的旧进程")
    ap.add_argument("--browser-crawl", action="store_true",
                    help="爬虫多源全失败时启动 Playwright 浏览器兜底(需 pip install playwright)")
    args = ap.parse_args()

    # 把 --browser-crawl 透传到 crawler daemon 的命令行
    if args.browser_crawl:
        for svc in SERVICES:
            if svc["name"] == "crawler" and "--allow_browser" not in svc["cmd"]:
                svc["cmd"].append("--allow_browser")

    if args.action == "status":
        return cmd_status()
    if args.action == "stop":
        return cmd_stop()

    print("=" * 60)
    print(" 认知大脑系统启动器")
    print("=" * 60)
    print(f" 项目目录: {ROOT}")
    print(f" Python:   {PYTHON}")
    print(f" 日志:     {LOG_DIR}/")
    print()

    # 0. 端口冲突检查
    print("[0] 检查端口")
    if not precheck_ports(reset=args.reset):
        return 1

    # 1. PostgreSQL
    if not args.no_pg:
        print("[1] 检查 PostgreSQL")
        if not ensure_postgres():
            print("无法启动 PG,中断。")
            return 1

    # 2. 各 Python 服务
    print("\n[2] 启动 Python 服务")
    env = make_env()
    skip_set = set()
    if args.no_crawl: skip_set.add("no_crawl")
    if args.no_train: skip_set.add("no_train")
    if args.no_thinking: skip_set.add("no_thinking")

    services_to_start = [
        s for s in SERVICES
        if s.get("skip_if") not in skip_set and s.get("require", lambda: True)()
    ]

    started = []
    try:
        for svc in services_to_start:
            start_service(svc, env)
            started.append(svc)
            ok = wait_for_ready(svc)
            if not ok and svc["name"] in ("grpc", "fastapi"):
                # 主服务起不来直接停
                print(f"主服务 {svc['label']} 启动失败,中止。")
                stop_all(started)
                return 1
    except Exception as e:
        print(f"\n启动期间出错: {e}")
        stop_all(started)
        return 1

    # 3. 报告
    print("\n" + "=" * 60)
    print("  ✓ 全部就绪")
    print("=" * 60)
    print()
    print("  📊 仪表盘: http://127.0.0.1:8000")
    print("  🔌 gRPC:   127.0.0.1:9999")
    print("  🗄  PG:     127.0.0.1:5432")
    print()
    print("  日志文件:")
    for svc in started:
        print(f"    {svc['label']:20s}  {svc['log_path']}")
    print()
    print("  按 Ctrl+C 停止所有服务")
    print()

    # 4. 守护循环
    try:
        while True:
            time.sleep(2)
            # 检查每个服务是否还活着
            for svc in started:
                p = svc.get("proc")
                if p and p.poll() is not None:
                    print(f"\n  ⚠ {svc['label']} 异常退出(code={p.returncode}),"
                          f"详情见 {svc['log_path']}")
                    # 标记不再监控避免重复打印
                    svc["proc"] = None
    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C,优雅停止全部服务...")
        stop_all(started)
        print("\n已全部停止。")
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
