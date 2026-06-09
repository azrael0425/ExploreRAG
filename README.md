# ExploreRAG

&emsp;&emsp;ExploreRAG 是一套面向中英文资料的多模态 RAG 工作台——不仅能回答问题，更能展示证据、追溯原文并量化效果。它将 Docling 文档解析与 OCR、BGE-M3 向量检索和重排、LightRAG 知识图谱、流式问答、页码/图片级引用及可审计评测整合进同一套 Web 应用，支持 PDF、DOCX、PPTX 等多种格式，并可灵活切换 Ollama 本地模型与 OpenAI 兼容云端模型，覆盖从知识入库、检索生成到反馈回流与质量验证的完整闭环。

## 项目演示

<div align="center">

https://github.com/user-attachments/assets/0b7f44cb-f9b2-4c38-8631-05ef65075072

</div>

## 目录

- [项目演示](#项目演示)
- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始docker-compose)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [使用流程](#使用流程)
- [RAG 评测](#rag-评测)
- [文档索引](#文档索引)
- [Contributors](#contributors)
- [许可证](#许可证)

## 核心能力

### 文档知识库

- 创建、编辑和删除多个独立知识库。
- 支持 `PDF`、`DOCX`、`PPTX`、`TXT`、`Markdown` 文档。
- 使用 Docling 解析版面、标题层级、表格、公式和图片；TXT / Markdown 使用轻量文本解析路径。
- 自动识别普通 PDF、扫描 PDF 和混合 PDF，并对需要的页面执行 OCR。
- 保存解析后的 Markdown、页码、标题路径、表格、图片及图片描述。
- 支持导入时业务元数据、知识库级元数据字段定义、元数据筛选和并发修改检测。

### 高质量检索与生成

- 使用本地 `BAAI/bge-m3` 生成中英文统一向量，默认维度为 1024。
- 使用 ChromaDB 保存和检索文档分块。
- 可选 `BAAI/bge-reranker-v2-m3` 交叉编码器重排。
- 使用 LightRAG 为每个知识库构建实体—关系图谱。
- 可独立开启查询时的图谱证据增强；关闭增强不影响图谱构建和可视化。
- LangChain / LCEL 统一编排检索链与回答链。
- 支持 `hybrid`、`vector_only`、`local`、`global` 检索模式。
- 通过 SSE 流式返回检索状态、来源、思考过程、回答 token、图片和性能指标。
- 回答可回溯原文页码、图片和关联图谱。

### 多模态与附件

- 文档图片可由视觉模型生成描述，并作为检索与回答上下文。
- 表格以 Markdown 结构保留，可进行表格问答和数值类评测。
- 对话可临时上传 `JPG`、`PNG`、`PDF`、`DOCX`、`PPTX`、`TXT`、`Markdown`。
- 临时附件与正式知识库完全隔离，不写入知识库分块或知识图谱。
- 小附件可直接注入上下文，大附件使用独立临时向量集合检索。
- 清空对话历史时同步清理临时附件和相关产物。

### 可视化与评测

- 查看文档 Markdown、目录结构、页级图片和表格。
- 浏览知识图谱，支持缩放、全屏、全局图谱和引用实体聚焦图谱。
- 内置评测集生成、人工审核、JSONL 导入导出、运行队列、基线对比和结果复核。
- 支持 Vector × Reranker × Knowledge Graph 的 2×2 消融实验。
- 支持对回答和单条来源反馈，并将失败案例提升为评测集草稿，形成知识飞轮。

## 系统架构

### 系统架构总览

```mermaid
flowchart TB
    U["用户"] --> UI["React 19 + TypeScript SPA"]
    UI -->|"REST / SSE"| API["FastAPI /api/v1"]

    subgraph FLOWS["应用工作流"]
        INGEST["文档与知识库管理"]
        CHAT["流式 RAG 问答"]
        EVAL["评测与反馈"]
    end

    API --> INGEST
    API --> CHAT
    API --> EVAL

    subgraph CORE["解析、检索与生成"]
        PARSER["Docling / RapidOCR / HybridChunker"]
        LCEL["LangChain / LCEL 编排"]
        RETRIEVER["Hybrid Retriever"]
        JUDGE["确定性指标 + Ragas"]
    end

    INGEST --> PARSER
    CHAT --> LCEL
    LCEL --> RETRIEVER
    EVAL --> LCEL
    EVAL --> JUDGE

    subgraph INTEL["模型能力"]
        BGE["BGE-M3 Embedding"]
        RERANK["BGE Cross-Encoder Reranker"]
        KG["LightRAG 结构化图谱证据"]
        LLM["OpenAI-compatible LLM / Ollama"]
    end

    PARSER --> BGE
    PARSER --> KG
    RETRIEVER --> BGE
    RETRIEVER --> RERANK
    RETRIEVER --> KG
    LCEL --> LLM
    JUDGE --> LLM

    subgraph DATA["持久化与对象存储"]
        PG[("PostgreSQL<br/>业务、对话、评测")]
        CHROMA[("ChromaDB<br/>正式与临时向量集合")]
        FILES[("本地 Volume<br/>原文、图片、附件、LightRAG")]
    end

    API --> PG
    BGE --> CHROMA
    PARSER --> FILES
    KG --> FILES
```

前端通过 REST 管理知识库、文档和评测，通过 SSE 消费流式问答；后端将解析、检索、生成和评测拆成独立工作流，共用 PostgreSQL、ChromaDB 与本地持久卷。

实现映射：[FastAPI 入口](backend/app/main.py)、[API 路由](backend/app/api/router.py)、[LangChain 对话服务](backend/app/langchain_rag/service.py)、[RAG 服务](backend/app/services/explore_rag_service.py)。

### 文档入库与索引流程

```mermaid
flowchart TB
    UPLOAD["上传 PDF / DOCX / PPTX / TXT / Markdown"]
    UPLOAD --> RECORD["PostgreSQL：登记文档，等待处理"]
    UPLOAD --> SOURCE["uploads Volume：保存原始文件"]

    RECORD --> START["点击分析：启动后台任务"]
    SOURCE --> START
    START --> PROFILE["PDF 扫描特征检测"]
    PROFILE --> PARSE["Docling 结构解析<br/>扫描 PDF 启用 OCR"]
    START -->|"TXT / Markdown"| TEXT["轻量文本解析"]

    PARSE --> STRUCT["Markdown + 标题层级 + 页码<br/>表格 + 图片 + 可选视觉描述"]
    TEXT --> STRUCT
    STRUCT --> PGDATA["PostgreSQL：解析结果、统计与媒体元数据"]
    STRUCT --> NORMALIZE["Embedding token 上限保护<br/>噪声过滤 + 精确去重 + 近似去重"]

    NORMALIZE --> EMBED["BGE-M3 批量向量化"]
    EMBED --> VPENDING["Chroma：写入向量分块<br/>暂不参与检索"]
    STRUCT --> KGINGEST["LightRAG ingest<br/>实体、关系与来源信息"]

    VPENDING --> GATE{"索引阶段全部成功？"}
    KGINGEST --> GATE
    STRUCT -->|"KG 关闭时直接通过"| GATE

    GATE -->|"是"| PUBLISH["Chroma：发布向量索引<br/>PostgreSQL：标记文档可检索"]
    GATE -->|"否"| FAILED["Chroma：清理未完成分块<br/>PostgreSQL：标记处理失败"]
```

向量分块在图谱阶段完成前保持不可见。这样 Docling、Embedding 或 LightRAG 任一阶段失败，都不会让用户检索到不完整文档。后台作业还通过资源调度器协调 Docling、Embedding 和图谱增强任务。图中只呈现数据库职责；实现内部使用 PostgreSQL 文档状态 `PENDING → PROCESSING → PARSING → INDEXING → INDEXED / FAILED`，并使用 Chroma 可见性 `pending → ready` 完成发布控制。

- PDF 会先按真实页文本特征区分普通、扫描或混合类型；正式知识库中的扫描 PDF 使用完整 OCR 路径。
- Docling 负责版面、标题层级、页码、表格、公式与图片提取；TXT / Markdown 使用轻量路径。
- 表格 Markdown 和图片描述会进入分块内容，使它们可以参与文本检索；图片原文件和表格元数据仍单独保存，便于原文回溯。
- 被 Schema 标记为参与语义检索的业务元数据可拼入 Embedding 输入；全部合法业务元数据仍可参与文档范围解析和过滤。

实现映射：[上传接口](backend/app/api/documents.py)、[处理接口](backend/app/api/rag.py)、[入库编排](backend/app/services/explore_rag_service.py)、[Docling 解析器](backend/app/services/document_parser/docling_parser.py)、[分块去重](backend/app/services/chunk_dedup.py)。

### RAG 核心检索流程

```mermaid
flowchart TB
    Q["问题 + 历史消息<br/>文档范围 + 元数据过滤"]
    Q --> SCOPE["解析可检索文档范围"]
    SCOPE --> POLICY["RetrievalPolicy<br/>确定本次有效检索模式"]

    subgraph PARALLEL["知识库检索：按策略并行执行"]
        VSTART["向量检索分支"] --> QEMB["BGE-M3 Query Embedding"]
        QEMB --> PREFETCH["Chroma ready 分块<br/>全局 Top-N + 每个就绪文档候选"]
        PREFETCH --> RR["可选 Cross-Encoder Rerank<br/>独立于图谱开关；超时 / 熔断时按向量顺序回退"]
        RR --> DIVERSE["优先保留不同文档<br/>得到最终 Top-K"]

        GSTART["图谱分支"] --> LGRAPH["LightRAG aquery_data"]
        LGRAPH --> GEVIDENCE["实体 + 关系 + 图谱事实<br/>来源文档 + 可追踪诊断"]
    end

    POLICY -->|"存在可检索文档时执行"| VSTART
    POLICY -->|"仅未限定范围、KG 开启<br/>且未请求 vector_only"| GSTART

    DIVERSE --> MEDIA["按命中文档页关联<br/>图片、表格与视觉输入"]
    DIVERSE --> MERGE["统一上下文与 KB 引用"]
    MEDIA --> MERGE
    GEVIDENCE --> MERGE
    ATTACH["临时附件证据<br/>独立检索链"] -.-> MERGE

    MERGE --> PROMPT["不可信材料边界<br/>抗 Prompt Injection + 强制引用规则"]
    PROMPT --> MODEL["云端 OpenAI-compatible LLM<br/>或本地 Ollama"]
    MODEL --> OUT["回答 + 引用 + 图片<br/>相关实体 + Retrieval Trace + 各阶段耗时"]
```

- 向量分支先过量召回，再用 `bge-reranker-v2-m3` 对 `(问题, 分块)` 联合打分；`vector_only` 只关闭图谱分支，不会关闭 Reranker。重排超时、模型熔断或不可用时会 fail-open，保留向量排序。
- 未限制文档范围时，会给每个已索引文档保留少量候选，并在重排后优先选择不同文档，降低长文档连续分块占满 Top-K 的概率。
- 图谱分支通过 LightRAG 的结构化数据接口获取实体、关系、事实及来源，而不是直接采用 LightRAG 生成的答案。
- 指定 `document_ids` 或元数据过滤时，图谱分支会关闭，因为当前 LightRAG 集成是工作区级的；这是为了避免范围外证据泄漏，而不是能力缺失的静默降级。
- 最终 Prompt 将检索内容明确标记为不可信证据，要求模型忽略其中的指令，并使用系统生成的来源 ID 完成可点击引用。

实现映射：[检索链](backend/app/langchain_rag/chains/retrieval.py)、[DeepRetriever](backend/app/services/deep_retriever.py)、[回答链](backend/app/langchain_rag/chains/answer.py)、[检索策略](backend/app/services/retrieval_policy.py)、[知识图谱服务](backend/app/services/knowledge_graph_service.py)。

### 流式问答调用链

```mermaid
sequenceDiagram
    actor U as 用户
    participant F as ChatPanel / useRAGChatStream
    participant A as FastAPI chat_stream_endpoint
    participant R as LangChainChatService
    participant X as Retrieval + Attachment Services
    participant S as Chroma / LightRAG / Files
    participant L as Model API
    participant D as PostgreSQL

    U->>F: 提交问题、历史、思考开关与附件 ID
    F->>A: POST /api/v1/rag/chat/{workspace}/stream
    A->>D: 保存 user ChatMessage 与附件关联
    A->>R: stream_chat(ChatChainInput)
    R-->>A: status = analyzing
    A-->>F: SSE status

    opt 选择了临时附件
        R->>X: 校验并等待附件准备
        X->>S: 必要时解析、选择性 OCR 或创建临时索引
        X-->>R: attachment_* 状态事件
        R-->>A: attachment_* DomainEvent
        A-->>F: SSE attachment_* 状态
    end

    par 正式知识库检索
        R->>X: LCEL retrieval chain
        X->>S: Chroma 向量召回 + 可选 LightRAG 图谱证据
        S-->>X: 候选、事实、来源与耗时
    and 临时附件检索
        R->>X: direct context 或 tmp_chat_ws_* 检索
        X->>S: 读取临时分块和匹配页图片
        S-->>X: ATT 来源与视觉材料
    end

    X-->>R: 合并后的上下文、来源、图片与 trace
    R-->>A: sources / images
    A-->>F: SSE sources / images
    R->>L: LCEL answer prompt + 历史 + 不可信检索材料

    loop 模型流式输出
        L-->>R: thinking delta 或 text delta
        R-->>A: thinking / token
        A-->>F: SSE thinking / token
        F->>F: requestAnimationFrame 缓冲渲染
    end

    R->>R: 清理特殊 token、补导航引用、匹配图谱实体、汇总性能
    R-->>A: complete DomainEvent
    A->>D: 保存 assistant ChatMessage、来源、图片、步骤与 reply_to
    D-->>A: 返回持久化 message_id
    A-->>F: SSE complete + message_id + performance
    F-->>U: 完整回答、可点击来源、图片和思考时间线
```

SSE 连接每 15 秒发送一次 heartbeat，避免模型或本地资源较慢时被代理断开。后端会先持久化 assistant 消息，再发送 `complete`，因此刚完成的回答可以立即接受点赞、点踩和来源评分。前端对 token 使用 `requestAnimationFrame` 缓冲，减少高频流式更新造成的重复渲染。

主要事件包括：

```text
status
attachment_validating / attachment_queued / attachment_parsing
attachment_ocr_retry / attachment_indexing / attachment_ready
retrieving_knowledge_base / retrieving_attachments
sources / images
thinking / token
complete / error
```

实现映射：[SSE HTTP 层](backend/app/api/chat_agent.py)、[领域事件与对话编排](backend/app/langchain_rag/service.py)、[前端 SSE Hook](frontend/src/hooks/useRAGChatStream.ts)、[聊天 UI](frontend/src/components/rag/ChatPanel.tsx)。

## 技术栈

| 层级 | 主要技术 |
|---|---|
| 前端 | React、TypeScript、Vite|
| 后端 | Python 3.11、FastAPI |
| 数据库 | PostgreSQL |
| 向量库 | ChromaDB |
| 文档解析 | Docling |
| RAG 编排 | LangChain/LCEL |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 知识图谱 | LightRAG |
| 模型服务 | DashScope Qwen、DeepSeek、Ollama |
| 评测 | Ragas |
| 部署 | Docker Compose、Nginx、NVIDIA GPU |

## 快速开始：Docker Compose

### 1. 环境要求

- Windows 11 + Docker Desktop/WSL2，或支持 NVIDIA Container Toolkit 的 Linux。
- NVIDIA 驱动可用，Docker 容器能够访问 GPU。
- Docker Compose v2。
- 至少准备 BGE-M3 和 Docling 模型目录；启用重排时还需 BGE Reranker。
- 使用云端模式时准备相应 API Key；使用本地模式时准备 Ollama 模型。

先确认主机 GPU 正常：

```powershell
nvidia-smi
docker info
```

### 2. 创建环境文件

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

Linux / WSL：

```bash
cp .env.example .env
```

至少检查以下配置：

```dotenv
# 在私有 .env 中设置强密码，并将其 URL 编码形式写入两个连接串
POSTGRES_PASSWORD=replace-with-a-strong-password
DATABASE_URL_DOCKER=postgresql+asyncpg://explorerag:replace-with-a-strong-password@postgres:5432/explorerag
DATABASE_URL=postgresql+asyncpg://explorerag:replace-with-a-strong-password@localhost:5433/explorerag

# 模型在宿主机上的目录，供 Docker Compose 只读挂载
BGE_M3_MODEL_DIR=../bge-m3
BGE_RERANKER_MODEL_DIR=../bge-reranker-v2-m3
DOCLING_MODELS_DIR=../docling-models

# 容器内路径，通常无需修改
KG_EMBEDDING_MODEL=/models/bge-m3
EXPLORERAG_EMBEDDING_MODEL=/models/bge-m3
EXPLORERAG_RERANKER_MODEL=/models/bge-reranker-v2-m3
EXPLORERAG_DOCLING_ARTIFACTS_PATH=/models/docling
```

云端文本模型可选择 DashScope：

```dotenv
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_FAST=qwen-plus
QWEN_VISION_MODEL=qwen-vl-plus
```

或使用 DeepSeek 负责文本生成，同时保留 DashScope Qwen 负责图片理解：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_FAST=your-deepseek-model

# 开启图片描述时仍需配置
DASHSCOPE_API_KEY=your-dashscope-api-key
QWEN_VISION_MODEL=qwen-vl-plus
```

### 3. 构建并启动

```powershell
docker compose up -d --build
docker compose ps
```


### 4. 准备本地 Ollama 模型（可选）

仅在知识库选择“本地模型”时需要：

```powershell
docker exec explorerag-ollama ollama pull qwen3-vl:4b-instruct
docker exec explorerag-ollama ollama list
```

本地模式不会在 Ollama 不可用时静默回退到云端。模型缺失或服务异常会直接显示在知识库的 LLM 状态中。

### 5. 访问服务

| 服务 | 地址 | 说明 |
|---|---|---|
| Web 应用 | <http://localhost:5500> | Nginx 托管前端并代理 API/SSE |
| 后端 API / 文档 | 通过 Web 应用的 `/api` 代理 | 不默认映射宿主机端口 |
| PostgreSQL / ChromaDB / Ollama | Compose 内部网络 | 不默认暴露到宿主机 |

查看日志：

```powershell
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f ollama
```

停止服务但保留数据：

```powershell
docker compose down
```

## 项目结构

```text
ExploreRAG/
├─ backend/
│  ├─ app/
│  │  ├─ api/                    # 知识库、文档、对话、RAG 与评测 API
│  │  ├─ core/                   # 配置、数据库、依赖注入与异常处理
│  │  ├─ evaluation/             # 评测生成、执行、指标和报告
│  │  ├─ langchain_rag/          # LangChain 检索链、回答链及适配器
│  │  ├─ models/                 # SQLAlchemy 数据模型
│  │  ├─ schemas/                # Pydantic 请求与响应模型
│  │  ├─ services/               # 解析、索引、检索、重排、图谱和附件服务
│  │  ├─ main.py                 # FastAPI 应用入口
│  │  └─ schema.sql              # 初始数据库结构
│  ├─ alembic/                   # 数据库迁移脚本
│  ├─ scripts/                   # 数据集、评测、报告和模型辅助脚本
│  ├─ tests/                     # 后端单元测试与契约测试
│  ├─ requirements.txt           # 后端运行依赖
│  └─ requirements-evaluation.txt # 离线评测附加依赖
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  │  ├─ layout/              # 页面布局组件
│  │  │  ├─ rag/                 # 对话、文档、图谱和设置组件
│  │  │  └─ ui/                  # 通用 UI 组件
│  │  ├─ hooks/                  # 工作区、历史记录和流式请求 Hook
│  │  ├─ lib/                    # API 客户端与通用工具
│  │  ├─ pages/                  # 知识库、工作区和评测页面
│  │  ├─ stores/                 # Zustand 状态管理
│  │  └─ types/                  # TypeScript 类型定义
│  ├─ package.json               # 前端依赖和构建命令
│  └─ vite.config.ts             # Vite 开发服务器与代理配置
├─ evaluation/
│  ├─ configs/                   # 消融实验配置
│  ├─ README.md                  # 评测流程、指标口径和实验规范
│  └─ FORMAL_RESULTS.md          # 可公开引用的正式评测摘要
├─ docs/
│  ├─ ARCHITECTURE.md            # 系统边界、模块职责与核心约束
│  ├─ RAG_PIPELINE.md            # 入库、检索、附件与流式问答流程
│  ├─ DEPLOYMENT.md              # Docker 拓扑、启动、验证与升级
│  ├─ CONFIGURATION.md           # 配置层级、变量分组与变更影响
│  ├─ TROUBLESHOOTING.md         # 常见故障诊断与安全恢复路径
│  └─ DATA_AND_BACKUP.md         # 数据归属、生命周期与一致性备份
├─ showcase/
│  ├─ explorerag-demo.mp4        # 压缩后的项目演示视频
│  └─ system-architecture.svg    # 系统架构图
├─ tools/                        # 模型下载与可复现构建工具
├─ .dockerignore                 # Docker 构建忽略规则
├─ .env.example                  # 环境变量示例
├─ .gitignore                    # Git 忽略规则
├─ docker-compose.yml            # 完整单机部署
├─ docker-compose.services.yml   # 本地开发所需数据库与向量库
├─ Dockerfile.backend            # 后端镜像
├─ Dockerfile.frontend           # 前端构建及 Nginx 镜像
├─ Dockerfile.frontend.prebuilt  # 使用预构建前端产物的轻量镜像
├─ nginx.conf                    # 静态资源与后端代理配置
├─ run_bk.sh                     # 本地启动后端
├─ run_fe.sh                     # 本地启动前端
├─ setup.sh                      # 本地环境初始化
├─ LICENSE                       # MIT 许可证
└─ README.md
```


## 配置说明

主要配置如下：

| 配置类别 | 主要变量 | 说明 |
|---|---|---|
| 数据服务 | `DATABASE_URL`、`CHROMA_HOST`、`CHROMA_PORT` | PostgreSQL 和 ChromaDB 的连接地址 |
| 本地模型目录 | `BGE_M3_MODEL_DIR`、`BGE_RERANKER_MODEL_DIR`、`DOCLING_MODELS_DIR` | Docker 挂载的 Embedding、Reranker 和 Docling 模型目录 |
| 云端模型 | `LLM_PROVIDER`、`LLM_MODEL_FAST`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY` | 选择 DashScope 或 DeepSeek，并配置对应密钥 |
| 本地模型 | `LOCAL_LLM_MODEL`、`LOCAL_LLM_BASE_URL` | Ollama 模型名称和服务地址，默认使用 `qwen3-vl:4b-instruct` |
| RAG 能力 | `EXPLORERAG_ENABLE_KG`、`EXPLORERAG_ENABLE_RERANKER` | 控制知识图谱和重排功能 |
| 跨域访问 | `CORS_ORIGINS` | 允许访问后端的前端地址 |



## 使用流程

1. 打开知识库列表，创建一个知识库。
2. 在知识库设置中选择云端或本地 LLM，配置系统提示词、图谱语言、实体类型和图谱增强开关。
3. 如需业务字段，先配置元数据 Schema，再拖拽上传文档并填写元数据。
4. 上传完成后点击“分析”。解析和索引在后台执行，可在文档卡片查看状态。
5. 文档变为“就绪”后开始提问；可选择文档、元数据范围或添加临时资料。
6. 点击回答中的来源查看原文，点击“关联图谱”聚焦引用涉及的实体和关系。
7. 对回答或来源点赞/点踩，必要时在“RAG 评测”页面审核并提升为回归案例。


## RAG 评测

为了验证 Reranker 和知识图谱是否真正改善检索与回答质量，项目在同一套冻结测试集上进行了 2×2 消融实验。

### 评测范围

- 评测集共 120 条，其中开发集 80 条、冻结测试集 40 条。
- 测试语料包含 12 份 PDF / Markdown 文档，共 1564 个分块。
- 测试题覆盖单跳、多跳、跨文档、表格数值、引用、拒答和多轮问题。
- 四组实验使用相同的测试题、模型、Prompt 和检索参数，只改变 Reranker 与 Knowledge Graph 开关。

### 消融方案

| Variant | 向量检索 | Reranker | Knowledge Graph |
|---|:---:|:---:|:---:|
| A Vector | ✓ |  |  |
| B +Reranker | ✓ | ✓ |  |
| C +KG | ✓ |  | ✓ |
| D Full | ✓ | ✓ | ✓ |

### 冻结测试集结果

以下结果均来自同一套冻结测试集，共 40 条问题，其中 37 条可回答问题、3 条不可回答问题。四组实验使用相同语料、模型、Prompt 和检索参数，只改变 Reranker 与 Knowledge Graph 开关。

| 指标 | A Vector | B +Reranker | C +KG | D Full |
|---|---:|---:|---:|---:|
| 正确证据命中率（Hit@4） | 59.46% | 72.97% | 59.46% | **72.97%** |
| 关键证据覆盖率（Recall@4） | 50.00% | 62.16% | 50.00% | **62.16%** |
| 回答事实完整度 | 64.46% | 72.54% | 69.65% | **75.16%** |
| 回答忠实度 | 68.65% | 83.03% | 79.49% | **85.14%** |
| 引用覆盖率 | 50.00% | 59.46% | 50.00% | **63.89%** |
| 拒答准确率 | 92.50% | 95.00% | 90.00% | **97.50%** |
| 图谱证据覆盖率 | 0.00% | 0.00% | 48.72% | **48.72%** |

### 组件带来的提升

与只使用向量检索的 A 组相比：

- Reranker 将关键证据覆盖率从 **50.00% 提升到 62.16%**，回答忠实度从 **68.65% 提升到 83.03%**。
- Knowledge Graph 将回答忠实度从 **68.65% 提升到 79.49%**，并补充了实体、关系和来源信息。
- 完整管线将正确证据命中率从 **59.46% 提升到 72.97%**，回答忠实度从 **68.65% 提升到 85.14%**。

这些结果说明，Reranker 主要改善候选证据排序，Knowledge Graph 主要补充跨文档和结构化关系证据，两者组合后整体效果最好。

### 性能开销

完整管线的端到端延迟 P50 约为 **3.82 秒**；Reranker 阶段 P95 约为 **0.79 秒**，知识图谱查询预热后 P95 约为 **0.21 秒**。总延迟包含云端模型生成和网络波动，因此不同实验之间的总 P95 仅作为运行观测，不直接归因于某个组件。

### 后续优化方向

当前主要优化方向是进一步提高复杂问题的证据召回率和引用准确性，并扩大跨文档、图谱关系、表格和扫描文档测试集。

详细材料：

- [评测方法、数据集与指标说明](evaluation/README.md)
- [完整消融结果、置信区间、延迟与实验限制](evaluation/FORMAL_RESULTS.md)


## 文档索引

| 文档 | 说明 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统边界、模块职责、数据权威来源和核心约束 |
| [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md) | 文档入库、核心检索、临时附件与流式调用链 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker 部署拓扑、本地开发、验证、升级和运行边界 |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | 配置来源、关键变量、工作区策略和重建索引影响 |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 服务、模型、索引、流式请求与 WSL 资源排查 |
| [docs/DATA_AND_BACKUP.md](docs/DATA_AND_BACKUP.md) | 数据归属、生命周期、一致性备份和恢复原则 |
| [.env.example](.env.example) | 环境变量和主要运行配置示例 |
| [evaluation/README.md](evaluation/README.md) | RAG 评测流程、指标口径和实验规范 |
| [evaluation/FORMAL_RESULTS.md](evaluation/FORMAL_RESULTS.md) | 正式评测结果、消融实验和结论 |
| [LICENSE](LICENSE) | MIT 开源许可证 |


## Contributors

- [Azrael](https://github.com/azrael0425)


## 许可证

MIT
