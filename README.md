# AI Agent 智能客服系统

从零手写的 ReAct Agent + RAG 智能客服系统，不依赖 LangChain，深入理解 LLM Agent 底层原理。

## ✨ 特性

- **手写 ReAct 循环** — while + Function Calling + 记忆管理，不调包
- **完整 RAG 管道** — 文档切片 → Embedding 向量化 → 余弦相似度检索
- **5 个可调用工具** — 计算器、时间查询、知识库检索、订单查询、转人工
- **滑动窗口记忆** — Token 预算控制，自动压缩旧对话
- **对话评估系统** — 记录工具调用轨迹，自动评分（效率分、平均迭代次数）
- **双 API 分离** — DeepSeek（LLM）+ DashScope（Embedding）

## 🏗 架构

```
用户输入
  │
  ▼
┌─────────────────────────────────────────┐
│  main.py — 组装入口                      │
│    ├── OpenAI 客户端（LLM + Embedding）   │
│    ├── RAG 管道（加载知识库）              │
│    ├── 工具注册（5 个 Function Calling）  │
│    ├── 记忆管理（滑动窗口 + Token 预算）   │
│    ├── 评估器（轨迹记录 + 自动评分）       │
│    └── Agent 核心（ReAct 循环）           │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  agent/core.py — ReAct 循环              │
│                                          │
│  while iteration < max_iterations:       │
│    response = LLM(messages, tools)       │
│    if 没有 tool_calls → 返回答案          │
│    if 有 tool_calls   → 执行工具 → 继续   │
└─────────────────────────────────────────┘
  │           │           │
  ▼           ▼           ▼
┌──────┐ ┌──────┐ ┌──────────┐
│memory│ │tools │ │evaluator │
│ 滑动  │ │5个工具│ │轨迹+评分 │
│ 窗口  │ │JSON  │ │          │
└──────┘ └──┬───┘ └──────────┘
            │
       ┌────┴────┐
       ▼         ▼
   ┌──────┐  ┌──────────┐
   │embed │  │retriever │
   │1536维│  │余弦相似度│
   └──────┘  └──────────┘
```

## 📁 项目结构

```
AiAgentProject/
├── main.py                   # 入口：初始化 + CLI 循环
├── agent/
│   ├── core.py               # Agent 核心：ReAct 循环
│   ├── tools.py              # 工具定义（JSON Schema）+ 执行
│   └── memory.py             # 对话记忆：滑动窗口 + Token 预算
├── rag/
│   ├── embedder.py           # 文本 → 1536维向量
│   └── retriever.py          # 文档加载 → 切片 → 索引 → 搜索
├── eval/
│   └── evaluator.py          # 对话轨迹记录 + 自动评分
├── utils/
│   ├── config.example.py     # 配置模板（复制为 config.py）
│   └── config.py             # 实际配置（API Key，已 gitignore）
├── data/
│   └── company_docs.txt      # 模拟电商知识库
└── requirements.txt
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp utils/config.example.py utils/config.py
```

编辑 `utils/config.py`，填入你的 API Key：

- **LLM**：DeepSeek API（[platform.deepseek.com](https://platform.deepseek.com)）
- **Embedding**：DashScope（阿里云，[dashscope.aliyun.com](https://dashscope.aliyun.com)）

### 3. 运行

```bash
python main.py
```

### 4. 交互

```
请输入问题：我是钻石会员，买智能手环X1和无线耳机E3要多少钱？
回答: 原价498元，钻石会员8.5折后423.3元

请输入问题：reset    ← 重置对话
请输入问题：exit      ← 退出并查看评分
```

## 🛠 工具列表

| 工具 | 功能 | 参数 |
|------|------|------|
| `calculator` | 数学计算 | `expression` |
| `get_current_time` | 当前时间 | 无 |
| `search_knowledge` | RAG 知识库检索 | `query` |
| `get_order_status` | 订单状态查询 | `order_id` |
| `transfer_to_human` | 转人工客服 | `reason` |

## 🧠 技术细节

### ReAct 循环

```
Thought → Action → Observation → Thought → ... → Answer
```

不依赖 LangChain，核心就是一个 `while` 循环：调 LLM → 判断是否调工具 → 执行工具 → 结果喂回 LLM → 继续，直到 LLM 认为可以回答。

### RAG 管道

1. 文档加载 → 按段落 + 句子边界切片（避免语义截断）
2. 批量调用 Embedding API 生成 1536 维向量
3. 用户提问时同样生成向量
4. 余弦相似度计算 → 取 Top-K 最相关片段
5. 拼接片段 + 原始问题 → 提交 LLM 生成答案

### 记忆管理

- 滑动窗口策略：Token 超限时从最早的非 system 消息开始删除
- System Prompt 永远保留
- 支持 `reset` 命令手动清空对话

### 评估系统

- 记录每轮对话的工具调用轨迹（工具名、参数、结果）
- 自动计算效率分（迭代越少分越高）、平均迭代次数
- 可扩展为持久化日志 + 可视化面板

## 📊 RAG 优化方向

当前为基线实现，后续优化方向包括：

- NLP 动态句子边界检测（替代简单正则）
- 重叠窗口切片（防止关键信息被切断）
- Query Rewrite（用户问题改写扩展）
- 检索结果语义校验（过滤不相关结果）
- 向量数据库替代暴力搜索（ChromaDB / Milvus）

## 📄 License

MIT
