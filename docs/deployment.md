# 部署

## 服务拓扑

`run.py` 是唯一入口,它按顺序拉起下面这些进程并在前台守着:

| 服务 | 命令 | 端口 | 说明 |
| --- | --- | --- | --- |
| PostgreSQL | 外部 | 5432 | 不在就尝试 `pg_ctl start`(需配 `PGSQL_BIN` / `PGSQL_DATA`) |
| gRPC 大脑 | `grpc_predict.py` | 9999 | 主服务,加载大脑权重 |
| FastAPI 仪表盘 | `uvicorn webapp.app:app` | 8000 | 带登录 |
| 巩固守护 | `scripts/consolidator_daemon.py --interval 600` | — | pending → learned → 写进大脑 |
| 爬虫守护 | `scripts/crawler_daemon.py` | — | 多源抓取,自己找新主题 |
| 重训守护 | `scripts/training_daemon.py --interval 1800` | — | fasttext 意图分类器 |
| 思维链守护 | 外部脚本 | — | 同时配了 `THINKING_PYTHON` + `THINKING_DAEMON` 才启用 |

`Ctrl+C` 会逐个优雅停止。

## 启动前

```bash
pip install -r requirements.txt
cp .env.example .env          # 至少要配 PG_DSN
psql "$PG_DSN" -f db/schema.sql
```

## 常驻运行

### Linux / systemd

```ini
[Unit]
Description=daxy-brain
After=network.target postgresql.service

[Service]
WorkingDirectory=/opt/daxy-brain
EnvironmentFile=/opt/daxy-brain/.env
ExecStart=/opt/daxy-brain/.venv/bin/python run.py
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
```

### Windows / 计划任务

`deploy/start_service.bat` 是个看门循环(退出后 15 秒自动拉起),
`deploy/start_hidden.vbs` 负责无窗口启动它 —— 用 VBS 包一层是为了不留可见终端,
否则那个窗口一旦被手滑关掉,整个服务就跟着停了。

```bat
schtasks /Create /TN daxy-brain /SC ONLOGON /RU <用户> /RP <密码> ^
  /TR "wscript.exe C:\path\to\daxy-brain\deploy\start_hidden.vbs" /F
schtasks /Run /TN daxy-brain
```

两点注意:

- **`ONLOGON` 计划任务会莫名消失。** 遇到"重启后服务再也不回来",
  先 `schtasks /Query /TN daxy-brain` 确认任务还在不在,别一上来就查代码。
- **gRPC 起来要时间。** 大脑权重文件大的时候,从进程启动到端口开始监听可能要一两分钟
  (加载权重 + 建 key 矩阵),这期间连不上是正常的,不是启动失败。

## 停服

Windows 上那个看门 bat 会 15 秒把服务复活,所以**光杀 python 进程停不干净**。
真要停:先 `schtasks /End`,再杀掉 CommandLine 里含 `start_service` 的 `cmd.exe`,
最后才杀 python。

**改大脑权重文件之前必须真停干净**,否则 gRPC 的自动保存和巩固守护会把旧的内存状态
写回去,改动就白做了 —— 而且这种"改了没生效"极难察觉,容易一路误判下去。

## 冒烟验证

```bash
python tests/grpc_client_test.py     # 打 gRPC
curl -I http://127.0.0.1:8000/       # 仪表盘;没登录被打回是正常的
```

## 重新生成 protobuf

`chatbot_grpc/` 里是已生成好的代码,一般不用动。要改接口定义:

```bash
python -m grpc_tools.protoc -I chatbot_grpc \
  --python_out=chatbot_grpc --grpc_python_out=chatbot_grpc \
  chatbot_grpc/chatbot.proto
```

生成的 `chatbot_pb2.py` 里嵌着 `chatbot.proto` 这个 descriptor 名,
**改文件名就对不上**,所以这个目录和 proto 文件名保持原样,没跟着项目一起改名。
