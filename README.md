# daxy-brain

一套跑在 CPU 上的**类脑稀疏关联记忆系统**:不做梯度下降式的"训练—部署"两段论,
而是把每一次对话、每一条抓来的语料当场写进记忆,边跑边长。

核心不是又一个 Transformer。`brain/cognitive_brain.py` 是纯 numpy + scipy 稀疏矩阵实现的
**SDR(稀疏分布表示)关联网络**:字符经感知层编码成稀疏激活模式,Hebbian 规则在线巩固,
检索时走稀疏点积召回。没有 GPU 也能毫秒级返回,记忆条数上到十万量级仍然是稀疏运算。

外面套着一条完整的自学习闭环:爬虫抓 → 置信度打分 → 巩固进脑 → 对话命中 → 再巩固。

---

## 它由哪些部分组成

```
                 ┌─────────────────────────────────────────┐
   用户提问 ────▶ │  gRPC 服务  grpc_predict.py  :9999      │
                 │   ├─ 双脑检索 CognitiveBrain.answer     │
                 │   ├─ 思维逻辑路 ThinkingIndex(BM25)    │
                 │   └─ 意图分类 fasttext → recall+dnnsort │
                 └───────────────┬─────────────────────────┘
                                 │ 每轮对话异步落库
                 ┌───────────────▼─────────────────────────┐
                 │  PostgreSQL :5432                       │
                 │   sentence_corpus / web_corpus          │
                 │   vision_corpus  / training_log         │
                 └───────────────┬─────────────────────────┘
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
 ┌──────▼───────┐      ┌─────────▼────────┐     ┌─────────▼────────┐
 │ crawler      │      │ consolidator     │     │ training         │
 │ 多源抓取     │      │ pending→learned  │     │ fasttext 重训    │
 │ 自己找新词   │      │ 写进大脑记忆     │     │ 热重载分类器     │
 └──────────────┘      └──────────────────┘     └──────────────────┘

                 FastAPI 仪表盘 webapp/  :8000
                 浏览语料库 / 看大脑状态 / 在线问答测试
```

| 目录 | 干什么的 |
| --- | --- |
| `brain/` | 认知大脑本体:SDR 编码、情景记忆、Hebbian 巩固、置信度评分、思维逻辑 BM25 索引 |
| `recall/` | 词法召回层(BM25 / fasttext 向量) |
| `classify/` | fasttext 意图分类(问答 vs 闲聊) |
| `dnnsort/` | 候选答案的深度排序模型 |
| `skills/` | 多源爬虫、浏览器兜底抓取、图片搜索、内容免疫(反污染) |
| `scripts/` | 三个常驻守护 + 灌库、迁移、评测脚本 |
| `webapp/` | FastAPI 仪表盘(带登录) |
| `chatbot_grpc/` | protobuf 生成代码(原样保留,改名要重新 protoc) |
| `benchmarks/` | 性能与精度基准 |
| `db/schema.sql` | 建表语句 |
| `deploy/` | Windows 计划任务用的看门启动脚本 |

---

## 快速开始

```bash
git clone https://github.com/Saihai2024/daxy-brain.git
cd daxy-brain
pip install -r requirements.txt

cp .env.example .env      # 按自己的环境改,尤其是 PG_DSN
psql "$PG_DSN" -f db/schema.sql   # 四张主表;thinking_* 两张表由代码首次运行时自建

python run.py             # 一键拉起 gRPC + 仪表盘 + 三个守护
```

常用参数:

```bash
python run.py --no-crawl      # 不启爬虫
python run.py --no-train      # 不启重训
python run.py status          # 看各服务状态
python run.py stop            # 停掉本脚本启的进程
```

仪表盘在 http://127.0.0.1:8000 。**没配 `DASH_PASS` 时会随机生成一次性口令并打印在启动日志里**
—— 仓库里刻意不留任何默认密码,因为这个面板能读写整个语料库和大脑。

冒烟验证:

```bash
python tests/grpc_client_test.py
```

---

## 仓库里没有什么

模型权重(`brain_v2.pkl` 这类)、语料库、日志、浏览器 cookie **都不在仓库里**,
体量在几百 MB 到几 GB 之间,已经写进 `.gitignore`。

冷启动一个空大脑:建好表 → 准备一份问答对语料(两个平行文本文件,一行一条)→

```bash
python scripts/bootstrap_brain.py --input 你的/input.txt --target 你的/target.txt --limit 5000
```

给记忆打个底,然后让 crawler 跑起来自己长。也可以完全跳过 bootstrap,
直接让爬虫从零积累,只是前期问什么都答不上来。

---

## 几个值得先知道的坑

这些都是实际运行中踩出来的,不是理论风险:

**巩固守护会被陈旧锁卡死。** `logs/BRAIN_RELOAD_IN_PROGRESS` 是 reload 期间的互斥锁,
但判据只有"文件在不在",没有 PID 存活检查。一次 reload 崩溃留下的孤儿锁,能让整合
**永久空转**——爬虫照抓、日志照刷,就是一条也进不了大脑。发现"抓了很多但大脑没长",
先去 `logs/` 看有没有这个文件(注意写锁的四个脚本里只有一部分会把 pid 写进去,
所以文件里没 pid 不代表锁是活的)。

**自我污染闭环。** 早期版本把机器人自己的回答也写回语料库,巩固后又提升成记忆,
于是**答错一次就永久固化**(相似度 1.0,以后永远答这个错的)。现在只有高置信主路径
才回写,`creative` / `qa_recall` 这两条不再自我强化。改这块逻辑时务必想清楚这一点。

**Windows 上直接调 conda 环境里的 python.exe 会崩在 numpy。** 没 activate 时
`Library\bin` 不在 PATH 上,BLAS 的 DLL 延迟加载失败,矩阵乘法直接段错误
(`0xC06D007F`),而且**零输出**。`run.py` 已经手动把这些目录注入子进程 PATH,
但你要自己单独跑脚本时得注意。

**国际源整批静默失灭,多半是代理协议写错了。** 走 `trust_env` 时,系统代理设置
可能被解析成 `https://127.0.0.1:7890`,对着一个明文 HTTP 端口发 TLS 握手必然失败,
维基/arXiv/DDG 一起完蛋,而异常被吞掉不留日志。用 `CRAWLER_INTL_PROXY` 显式指定,
注意写 `http://`。

**fasttext 0.9.2 的 `test_label()` recall 恒为 nan。** 如果重训守护的验收门写成
`min_recall >= 0.9`,会被 nan 永远卡死,每轮都 SKIP,模型再也不更新——而进程还活着、
日志还在刷,看起来一切正常。本仓库的 `training_daemon.py` 已改成手工批量 predict 算
per-label 精确率/召回率。

**清理"脏记忆"别用单一字面判据。** 用过"主题与内容零 bigram 交集"来判脏,结果误伤了
一批手工教的人格词条("你叫什么名字"→"我叫认知大脑",字面确实零交集)。繁简差异、
中文词条配英文正文,也都会被误伤。`credibility` 才是更可靠的分界线。

**⚠️ gRPC 服务是裸奔的。** `grpc_predict.py` 用 `add_insecure_port("[::]:9999")`
绑的是**所有网卡**,而且没有任何鉴权。更麻烦的是它命中后会 `learn_dialogue` 写进大脑
并落库 —— 也就是说同网段的任何人不仅能查,还能往你的大脑里投喂。
仪表盘那一侧有登录,这一侧没有。**放到内网以外之前请自己加防火墙规则或改成绑
`127.0.0.1`。**

---

## 已知的能力边界

`brain/thinking_index.py` 顶部记着一次证伪结论:SDR 编码器在**长文本**上不成立
——实测编码器只有效利用文本末尾很短一段,长任务简报的 recall@1 是 0%,
全库余弦相似度均值 0.998(向量空间基本坍缩)。所以任务类提问单独走了一条 BM25 路,
没有硬套 SDR。

换句话说:这套 SDR 记忆适合**短问答、实体、概念**这类粒度;
长文档检索请用 `thinking_index` 那条词法路,或者自己接向量库。

---

## 环境要求

- Python 3.8 – 3.11(实际验证环境是 3.8)
- PostgreSQL 14+
- 语音转写功能需要 ffmpeg(可选,配 `FFMPEG_DIR`)
- 浏览器兜底抓取需要 Playwright(可选,`python run.py --browser-crawl`)

完整依赖见 `requirements.txt`。

---

由 [大兮科技](https://github.com/Saihai2024) 开发与维护。
