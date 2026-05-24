# Knowledge-DeepMind

全格式自进化知识库智能体 —— 支持多格式文件摄入、自动知识抽取、图谱构建、对话式检索的智能知识管理系统。

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
│  Python 3.11                         │
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

- **全格式摄入**：支持 PDF、Word、PPT、图片、网页、代码、视频、音频等文件类型
- **自动知识抽取**：DeepSeek 大模型驱动，自动从文件中提取知识点
- **知识图谱**：Neo4j 图数据库构建知识网络，支持多跳推理和路径发现
- **混合分类系统**：用户手动分类 + AI 自动推荐，形成"领域→主题→知识点"层级
- **对话式检索**：自然语言对话查询知识库，支持多轮上下文记忆
- **3D 可视化**：力导向图谱 + 多维筛选 + 时间线视图
- **用户系统**：注册/登录/JWT认证，用户数据完全隔离
- **置信度管理**：知识点置信度计算、低置信度审核、证据链追溯

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 14, React 18, TypeScript, Tailwind CSS, Framer Motion |
| 后端 | FastAPI, Python 3.11, Pydantic, Celery |
| 数据库 | PostgreSQL (pgvector), SQLite (本地开发) |
| 图数据库 | Neo4j 5 |
| 缓存 | Redis |
| 大模型 | DeepSeek API |
| 向量嵌入 | sentence-transformers, BGE |
| 对象存储 | MinIO / Cloudflare R2 |

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Neo4j 5
- PostgreSQL 16 (可选，本地开发默认用 SQLite)

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

## 项目结构

```
Knowledge-DeepMind/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── models/              # Pydantic 数据模型
│   ├── middleware/          # 认证中间件
│   ├── services/
│   │   ├── database.py      # 数据库适配层 (SQLite/PostgreSQL)
│   │   ├── auth_service.py  # 用户认证
│   │   ├── memory_service.py# 用户记忆系统
│   │   ├── conversation_store.py # 对话存储
│   │   ├── dedup_service.py # 去重服务
│   │   ├── agent/           # 对话智能体 + 搜索
│   │   ├── category/        # 分类聚类引擎
│   │   ├── confidence/      # 置信度计算
│   │   ├── extraction/      # 知识提取 + 时间抽取
│   │   ├── graph/           # Neo4j + 向量服务
│   │   ├── ingestion/       # 文件处理 (PDF/Word/媒体/代码)
│   │   └── webgen/          # 网页生成
│   ├── tests/               # 测试脚本
│   ├── render.yaml          # Render 部署配置
│   └── Procfile             # Heroku 启动文件
├── frontend/
│   ├── app/                 # Next.js 页面路由
│   │   ├── page.tsx         # 首页
│   │   ├── wiki/            # 知识库浏览
│   │   ├── chat/            # 对话检索
│   │   ├── graph/           # 图谱可视化
│   │   ├── upload/          # 文件上传
│   │   ├── timeline/        # 时间线视图
│   │   ├── login/           # 登录
│   │   ├── register/        # 注册
│   │   ├── profile/         # 个人中心
│   │   └── settings/        # 设置
│   ├── components/          # React 组件
│   ├── contexts/            # Auth Context
│   └── lib/                 # API 客户端 + 工具
├── docker/                  # Docker 初始化脚本
├── docker-compose.yml       # 本地基础设施编排
├── vercel.json              # Vercel 部署配置
├── .env.example             # 环境变量模板
└── start.bat                # Windows 一键启动
```

## 部署

本项目设计为零成本云部署架构：

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