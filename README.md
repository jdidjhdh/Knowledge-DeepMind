# Knowledge-DeepMind

支持多格式文件摄入、自动知识抽取、图谱构建、对话式检索的知识库系统。

## 架构

```
用户浏览器
    │
    ▼
┌──────────────────────────────────────┐
│  Next.js 前端 (:3000)                 │
│  React + TypeScript + Tailwind CSS   │
└──────────────┬───────────────────────┘
               │ /api/*
               ▼
┌──────────────────────────────────────┐
│  FastAPI 后端 (:8000)                 │
│  Python 3.12                         │
├──────────────────────────────────────┤
│  ┌─────────┐ ┌──────┐ ┌───────────┐ │
│  │PostgreSQL│ │Neo4j │ │  Redis    │ │
│  │ 核心数据 │ │知识图谱│ │ 缓存/队列 │ │
│  └─────────┘ └──────┘ └───────────┘ │
│  ┌──────────┐ ┌──────────────────┐  │
│  │DeepSeek │ │  MinIO / R2      │  │
│  │ 大模型   │ │  文件存储         │  │
│  └──────────┘ └──────────────────┘  │
└──────────────────────────────────────┘
```

## 功能特性

### 知识摄入与抽取
- **多格式摄入**：支持 PDF、Word、PPT、图片、网页、代码、视频、音频等文件类型
- **自动知识抽取**：DeepSeek 大模型驱动，自动从文件中提取结构化知识点
- **混合分类系统**：用户手动分类 + AI 自动推荐，形成"领域→主题→知识点"层级

### 混合检索引擎

多路召回 + RRF 融合排序，弥补单一向量检索的不足：

```
用户查询
  ├─→ 向量检索（语义相似，in-memory 向量存储）
  ├─→ 关键词检索（BM25 + 中文分词）
  ├─→ 图谱检索（实体1-2跳邻居，Neo4j）
  └─→ 查询重写（LLM生成2-3种变体，并行检索）
         ↓
    融合排序（RRF, K=60）
         ↓
    返回Top-K结果
```

| 组件 | 技术 | 说明 |
|------|------|------|
| Embedding模型 | `bge-large-zh-v1.5`（1024维） | 提升语义召回率 |
| BM25关键词检索 | 中文分词 + IDF加权 | 改善精确查询命中 |
| 查询重写 | LLM改写（简称→全称，口语→术语） | 改善模糊查询命中 |
| RRF融合排序 | Reciprocal Rank Fusion | 合并多路检索结果 |
| 查询缓存 | 相似查询复用检索结果 | 降低重复查询延迟 |

### 反幻觉四道防线

不依赖模型微调，通过 Prompt 工程与检索质量管控实现：

| 防线 | 机制 | 说明 |
|------|------|------|
| 🛡 检索盲区 | 相似度阈值拦截（<0.5拒绝回答）+ 空检索主动提问 | 减少编造 |
| 🛡 理解偏差 | 多证据交叉验证 + 反事实自检 + 实体锚定 | 减少误读 |
| 🛡 冲突/过时 | 时间衰减标注 + 冲突双向展示 + 版本溯源 | 减少误导 |
| 🛡 用户交互 | 回答元数据标签 + 一键质疑 | 透明可纠错 |

### 推理链优化

- **Chain-of-Thought 模板化**：对比、因果、总结、复杂拆解四类模板
- **动态少样本注入**：检索与当前问题最相似的3-5个高质量QA对
- **多步推理拆分**：复杂问题自动分解为2-4个子问题并行检索
- **知识库健康报告**：自动检测置信度分布、图谱覆盖率、稀疏领域，生成优化建议

### 图谱与可视化
- **知识图谱**：Neo4j 图数据库构建知识网络，支持多跳推理和路径发现
- **2D 力导向图谱**：节点实体可视化 + 多维筛选分类
- **冲突检测**：检测 `CONFLICTS_WITH` 环路，触发主动提问
- **置信度传播**：来源可信度降低时，沿边衰减相关知识点置信度

### 监控与反馈闭环
- **幻觉监控仪表盘**：引用缺失率、用户纠正率、低置信占比、冲突未解决率
- **知识缺口登记**：自动记录无法回答的问题，定期推送给用户
- **知识库健康报告**：置信度分布、图谱覆盖率、稀疏领域检测

### 基础能力
- **对话式检索**：自然语言对话查询知识库，支持多轮上下文记忆
- **用户系统**：注册/登录/JWT认证，用户数据完全隔离
- **置信度管理**：知识点置信度计算、低置信度审核、证据链追溯

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 14, React 18, TypeScript, Tailwind CSS, Framer Motion |
| 后端 | FastAPI, Python 3.12, Pydantic |
| 数据库 | SQLite (本地), PostgreSQL 适配层 (可选) |
| 图数据库 | Neo4j 5 |
| 缓存 | Redis / 内存缓存 |
| 大模型 | DeepSeek API |
| 向量嵌入 | sentence-transformers, BGE (`bge-large-zh-v1.5`) |
| 关键词检索 | BM25（中文分词 + IDF加权） |
| 融合排序 | Reciprocal Rank Fusion（RRF, K=60） |
| 对象存储 | MinIO / Cloudflare R2 |

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Neo4j 5（需 Java 17+）
- PostgreSQL 16（可选，本地开发默认用 SQLite）

### 1. 克隆仓库

```bash
git clone https://github.com/jdidjhdh/Knowledge-DeepMind.git
cd Knowledge-DeepMind
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 DeepSeek API Key 等必要配置。本地开发只需配置 LLM 部分即可运行。

### 3. 启动基础设施（Docker）

```bash
docker-compose up -d
```

这会启动 Neo4j、PostgreSQL、Redis、MinIO。

### 4. 安装后端依赖

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 5. 安装前端依赖

```bash
cd frontend
npm install
```

### 6. 启动服务

```bash
# 终端1：启动后端
cd backend
python main.py

# 终端2：启动前端
cd frontend
npx next dev -p 3000
```

打开 http://localhost:3000 即可访问。

## API 概览

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/chat` | POST | 对话检索（支持流式） |
| `/api/search/rewrite` | POST | LLM查询重写 |
| `/api/bm25/search` | GET | BM25关键词检索 |
| `/api/bm25/sync` | POST | 同步BM25索引 |
| `/api/graph/sync` | POST | 同步知识图谱数据 |
| `/api/graph/explore` | GET | 图谱探索查询 |
| `/api/graph/stats` | GET | 图谱统计 |
| `/api/knowledge/health-report` | GET | 知识库健康报告 |
| `/api/hallucination/metrics` | GET | 幻觉监控指标 |
| `/api/hallucination/gaps` | GET | 知识缺口列表 |
| `/api/hallucination/correct` | POST | 记录用户纠正 |
| `/api/hallucination/challenge` | POST | 记录用户质疑 |

## 项目结构

```
Knowledge-DeepMind/
├── backend/
│   ├── main.py              # FastAPI 入口 + 全部API路由
│   ├── config.py            # 配置管理
│   ├── models/              # Pydantic 数据模型
│   ├── middleware/          # 认证中间件
│   ├── services/
│   │   ├── database.py      # 数据库适配层 (SQLite/PostgreSQL)
│   │   ├── auth_service.py  # 用户认证
│   │   ├── memory_service.py# 用户记忆系统
│   │   ├── conversation_store.py # 对话存储
│   │   ├── dedup_service.py # 去重服务
│   │   ├── agent/           # 对话智能体 + 搜索 + 反幻觉
│   │   │   ├── conversation_agent.py  # 对话引擎(CoT+反幻觉)
│   │   │   └── search_service.py      # 混合检索(RRF融合)
│   │   ├── category/        # 分类聚类引擎
│   │   ├── confidence/      # 置信度计算
│   │   ├── extraction/      # 知识提取 + 时间抽取
│   │   ├── graph/           # Neo4j + 向量 + BM25
│   │   │   ├── neo4j_service.py  # 图数据库服务
│   │   │   ├── vector_service.py # 向量检索服务
│   │   │   └── bm25_service.py   # BM25关键词检索
│   │   ├── ingestion/       # 文件处理 (PDF/Word/媒体/代码)
│   │   └── webgen/          # 网页生成
│   ├── tests/               # 测试脚本
│   ├── render.yaml          # Render 部署配置
│   └── Procfile             # Heroku 启动文件
├── frontend/
│   ├── app/                 # Next.js 页面路由
│   │   ├── page.tsx         # 首页
│   │   ├── wiki/            # 知识库浏览
│   │   ├── chat/            # 对话检索（含反幻觉元数据）
│   │   ├── graph/           # 图谱可视化
│   │   ├── upload/          # 文件上传
│   │   ├── timeline/        # 时间线视图
│   │   ├── login/           # 登录
│   │   ├── register/        # 注册
│   │   ├── profile/         # 个人中心
│   │   └── settings/        # 设置
│   ├── components/          # React 组件
│   ├── contexts/            # Auth Context
│   └── lib/                 # API 客户端 + 缓存 + 工具
├── docker/                  # Docker 初始化脚本
├── docker-compose.yml       # 本地基础设施编排
├── vercel.json              # Vercel 部署配置
├── .env.example             # 环境变量模板
└── start.bat                # Windows 一键启动
```

## 部署

低成本云部署参考：

| 组件 | 服务 | 免费额度 |
|------|------|----------|
| 前端 | Vercel | 100GB 带宽/月 |
| 后端 | Render | 750h/月 |
| 数据库 | Neon/Supabase | 500MB |
| 图数据库 | Neo4j AuraDB | 20万节点 |
| 缓存 | Upstash Redis | 50万次/月 |
| 存储 | Cloudflare R2 | 10GB (免流量) |
| LLM | DeepSeek API | 按量计费 |

详细部署指南见 `.env.example` 和 `render.yaml`、`vercel.json`。

## License

MIT