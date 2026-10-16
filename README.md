# 智能文档问答助手（docqa）

一个基于 **FastAPI + RAG + 多 Agent 编排** 的文档问答后端：先把文档切块、向量化入库，再用「路由 → 检索 → 作答 → 质检」的多 Agent 流程回答关于这些文档的问题，并返回引用来源。

> 本地嵌入模型（`BAAI/bge-small-zh-v1.5`）+ 云端大模型（默认 DeepSeek），无需显卡也能跑。

---

## 功能特性

| 能力 | 说明 |
| --- | --- |
| 文档管理 | 增 / 查 / 删文档，自动切块并建立向量索引 |
| 向量检索 | ChromaDB 持久化向量库，余弦相似度取 Top-K 片段 |
| 纯对话 | `/api/chat/` 直连大模型，支持流式输出 |
| 单轮 RAG 问答 | `/api/ask/` 检索 + 拼接资料 + 作答，返回 `sources` |
| 多 Agent 编排 | `/api/agent/` Router → Retriever → Answerer → Critic，质检不通过时自动重答 |
| 健壮性 | 统一超时、关闭思维链、上游异常翻译成 502/503/504、写库与建索引保持事务一致 |

---

## 技术栈

- **Web 框架**：FastAPI 0.141 + Uvicorn
- **数据层**：SQLAlchemy 2.0 + SQLite
- **向量库**：ChromaDB 1.5（持久化到本地目录）
- **嵌入模型**：sentence-transformers 6.1 + `BAAI/bge-small-zh-v1.5`（本地推理，CPU 可跑）
- **大模型**：openai SDK 3.19，默认指向 DeepSeek（OpenAI 兼容协议，可换任意兼容服务）
- **测试**：pytest 9

---

## 目录结构

```
docqa/
├── app/
│   ├── main.py               # FastAPI 入口，建表 + 挂载 4 个路由
│   ├── config.py             # Settings（pydantic-settings，读 .env）
│   ├── database.py           # engine / SessionLocal / declarative Base / get_db
│   ├── models.py             # Document ORM 模型
│   ├── schemas.py            # Pydantic 出入参模型
│   ├── crud.py               # 文档增删查
│   ├── llm.py                # 大模型统一入口：惰性 client + 关思考 + 超时 + 异常翻译
│   ├── rag/
│   │   ├── chunking.py       # split_text()：定长滑窗切块（默认 400 字 / 重叠 80）
│   │   └── pipeline.py       # embed / index_document / delete_document / retrieve
│   ├── agents/               # 多 Agent：每个文件一个角色
│   │   ├── router.py         # 意图分类 rag / chat / unknown
│   │   ├── retriever.py      # 检索封装
│   │   ├── answerer.py       # 结合资料作答（含闲聊模式与重答模式）
│   │   ├── critic.py         # 答案质检 pass / fail
│   │   └── orchestrator.py   # 手写编排 handle()
│   └── routers/
│       ├── documents.py      # /api/documents
│       ├── chat.py           # /api/chat
│       ├── ask.py            # /api/ask
│       └── agent.py          # /api/agent
├── tests/                    # pytest 用例
├── requirements.txt          # 运行时直接依赖
├── requirements-dev.txt      # 开发依赖（= 运行时 + pytest）
├── requirements.lock.txt     # 完整锁定版本
└── pytest.ini
```

---

## 快速开始

### 1. 环境要求

- Python **3.11**（在 3.11.9 上验证通过）
- 无显卡即可运行

### 2. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 没有 NVIDIA 显卡时，务必加 CPU 索引，否则会拉 CUDA 版 torch（好几个 GB）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 开发/测试再装这个（已包含运行时依赖）
pip install -r requirements-dev.txt
```

### 3. 配置 `.env`

在项目根目录创建 `.env`（**该文件已在 `.gitignore` 中，不会入库**）：

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-flash
LLM_TIMEOUT=60
# 可选：自定义库位置
# DATABASE_URL=sqlite:///C:/data/docqa.db
# CHROMA_DIR=C:/data/chroma_db
```

### 4. 启动

```bash
uvicorn app.main:app --reload
# 或使用 FastAPI CLI
fastapi dev app/main.py
```

启动后：

- 健康检查 <http://127.0.0.1:8000/> → `{"status":"ok","msg":"ok"}`
- 交互式接口文档 <http://127.0.0.1:8000/docs>

> 首次启动会自动下载嵌入模型 `BAAI/bge-small-zh-v1.5` 到 HuggingFace 缓存。若模型已在本地缓存、或处在无外网环境，程序默认走离线模式（见下方 `EMBEDDING_OFFLINE`）。

---

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 健康检查 |
| POST | `/api/documents/` | 新建文档（201），自动切块并建索引 |
| GET | `/api/documents/` | 分页列出文档（`skip` / `limit`） |
| GET | `/api/documents/{doc_id}` | 查询单篇文档，不存在返回 404 |
| DELETE | `/api/documents/{doc_id}` | 删除文档及其全部向量切片（204） |
| POST | `/api/chat/` | 纯对话，返回完整回答 |
| POST | `/api/chat/stream` | 纯对话，`text/plain` 流式返回 |
| POST | `/api/ask/` | 单轮 RAG 问答，返回 `answer` + `sources` |
| POST | `/api/agent/` | 多 Agent 问答，返回 `intent` + `answer` + `sources` + `critic_passed` |

### 请求示例

```bash
# 1) 上传一篇文档
curl -X POST http://127.0.0.1:8000/api/documents/ \
  -H "Content-Type: application/json" \
  -d '{"title":"考勤制度","content":"公司上班时间是早上九点，下班时间是晚上六点，午休一小时。"}'

# 2) 单轮 RAG 问答
curl -X POST http://127.0.0.1:8000/api/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question":"几点上班？","top_k":3}'
# → {"answer":"九点。","sources":[{"..."}]}

# 3) 多 Agent 问答
curl -X POST http://127.0.0.1:8000/api/agent/ \
  -H "Content-Type: application/json" \
  -d '{"question":"年假有几天？","top_k":3}'
# → {"intent":"rag","answer":"...","sources":[...],"critic_passed":true}
```

---

## 多 Agent 编排

`app/agents/orchestrator.py` 里的 `handle()` 是一段可读性优先的**手写编排**，不是黑盒框架：

```
                 ┌──────────┐
   问题 ────────►│  Router  │  意图分类：rag / chat / unknown
                 └────┬─────┘
          rag ────────┴──────── chat / unknown
           │                         │
           ▼                         ▼
    ┌────────────┐            ┌─────────────┐
    │ Retriever  │  检索 TopK │  chat_only  │  不检索，直接对话
    └─────┬──────┘            └─────────────┘
          ▼
    ┌────────────┐
    │  Answerer  │  仅依据资料作答
    └─────┬──────┘
          ▼
    ┌────────────┐  fail
    │   Critic   │────────► Answerer(retry=True) 打回重答一次
    └─────┬──────┘
          │ pass
          ▼
      返回结果
```

各角色职责：

- **Router** — 只输出 `rag` / `chat` / `unknown` 一个词；模型若多说废话，兜底成 `unknown`。
- **Retriever** — 复用 `app/rag/pipeline.py` 的 `retrieve()`，返回文档块文本。
- **Answerer** — 严格「只根据资料回答，资料没有就说『资料中未提及』」；`retry=True` 时追加更严谨的指令重答。
- **Critic** — 只输出 `pass` / `fail`，检查是否真正回答问题、是否编造资料外内容。

> 提示词全部内置在代码中，不需要额外配置。

---

## 配置项

均通过 `.env` 或环境变量注入（字段名大小写不敏感）。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | 大模型密钥。为空时只在真正调用大模型时报 503，不影响文档接口 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | 兼容 OpenAI 协议的服务地址 |
| `MODEL_NAME` | `deepseek-flash` | 模型名 |
| `LLM_TIMEOUT` | `60` | 单次大模型调用超时（秒） |
| `DATABASE_URL` | `sqlite:///<项目根>/.docqa.db` | 数据库地址（绝对路径，不随启动目录漂移） |
| `CHROMA_DIR` | `<项目根>/chroma_db` | 向量库持久化目录 |
| `EMBEDDING_PROVIDER` | `local` | `local` 用本地模型；`openai` 走 OpenAI Embeddings |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 仅 `EMBEDDING_PROVIDER=openai` 时生效 |
| `EMBEDDING_OFFLINE` | `1` | 默认离线。设为 `0` 才会联网校验/下载模型 |

---

## 运行测试

```bash
# 全部用例（不联网、不消耗 token）
pytest

# 真实调用大模型的联调用例（会消耗 token）
RUN_LIVE_LLM=1 pytest -m live
```

- 用例通过 `tests/conftest.py` 在导入 `app` **之前**就把数据库、向量库切到临时目录，跑完不留垃圾，也不污染本地数据。
- 大模型调用用假客户端（ScriptedClient）按脚本应答，覆盖了意图路由、重答、异常翻译等路径。
- 当前状态：**51 passed, 3 skipped**（跳过的 3 个是需 `RUN_LIVE_LLM=1` 的联调用例）。

---

## 设计要点

这些都是踩过坑之后固化下来的行为，改动时请留意：

- **路径一律用绝对路径**。`DATABASE_URL` / `CHROMA_DIR` / `.env` 都基于项目根目录解析；写相对路径会跟着当前工作目录跑，换目录启动会「凭空」新建一个空库。
- **大模型 client 惰性创建**。模块级创建 client 会在 key 为空时直接抛错，导致 `import` 失败、整个应用（含文档接口）都起不来。
- **统一关闭思维链**。所有调用都带 `extra_body={"thinking": {"type": "disabled"}}`，省 token 也避免思维链混进答案。
- **强制超时**。不设超时时，上游挂住会打满线程池。
- **异常翻译**。未配 key → 503，超时 → 504，连接失败/上游错误 → 502，其余 → 500，避免上游 4xx 被吞成 500。
- **写库与建索引保持一致**。新建文档时先 `flush` 拿 id（不提交），索引建成功才 `commit`；失败则回滚数据库并清掉已写入的向量，不留孤儿记录。删除文档时先删向量再删记录，宁可搜不到也不搜到已删内容。
- **流式接口不能抛 HTTPException**。响应头已发出，错误只能以文本形式写进流里。
- **嵌入默认离线**。`HF_HUB_OFFLINE=1` 在 `config.py` 导入时即设定，避免每次 embed 都去 huggingface.co 校验 etag 而卡住。

---

## 已知限制

- 切块是定长滑窗（按字符数），没有按语义/段落边界切分。
- 检索是纯向量相似度，没有关键词混合检索与重排序。
- 多 Agent 编排是「检索一次 → 作答 → 质检」的固定流程，Agent 之间没有动态工具调用。
- Critic 打回后只重答一次，不做多轮迭代。
- 没有鉴权、限流与前端界面，定位是后端服务。
