# AI Agent 应用开发学习项目

从 Prompt 工程、RAG 检索问答到 Agent 工具调用与 LangGraph 状态图的实战学习项目，涵盖 LangChain / LlamaIndex / LangGraph 多框架，共 **30+ 天** 循序渐进。

## 🏆 项目亮点

- **五轮文本抽取 Prompt 迭代**：信息抽取准确率从 78% 提升至 96%，完整记录在 `prompts/extractor_log.md`
- **CoT vs Few-shot 对比实验**：用 10 个测试用例量化三种策略差异，证明 CoT 在模糊意图上准确率领先 30%
- **ReAct 智能代理**：手动实现 Thought-Action-Observation 循环，结合 `stop` 参数消除幻觉
- **完整 RAG 优化链**：切片策略对比 → 混合检索 → 查询改写 → 综合评估，Answer Relevancy 从 0.75 提升至 1.00
- **10 题全面测试集**：覆盖直接匹配、跨段落推理、对比总结、反幻觉、部分信息覆盖五种挑战类型
- **FastAPI 服务化**：`/rag/chat` 接口，返回答案 + 参考来源 + 置信度，附带请求日志
- **双框架对比**：LangChain（组件化精细控制）与 LlamaIndex（封装度高、代码少）的实践感受
- **Agent 与 LangGraph 进阶**：LangChain Agent / 多工具串联 / 对话记忆 / 异常熔断，进阶到 LangGraph 状态图复刻与 FastAPI 多会话服务化

## 📁 项目结构

### 基础阶段（Day 1–4）

| 文件 | Day | 说明 |
|------|-----|------|
| `word_counter.py` | Day 1 | 命令行词频统计工具 — argparse + Counter |
| `document_loader.py` | Day 2 | 多格式文档加载器 — TXT/JSON/CSV + 编码容错 |
| `llm_api.py` | Day 3 | DeepSeek API 封装 — requests 手写 HTTP，stop 参数示例 |
| `main.py` | Day 4 | 第一个 FastAPI 接口 — `/chat` + Pydantic 校验 |

### Prompt 工程（Day 5–8）

| 文件 | Day | 说明 |
|------|-----|------|
| `prompts/extractor.py` | Day 5 | 信息抽取 Prompt — V1→V5 五轮迭代 |
| `prompts/summarizer.py` | Day 5 | 长文摘要 Prompt — Markdown 结构化输出 |
| `prompts/debugger.py` | Day 5 | Bug 排查 Prompt — CoT 三步推理法 |
| `prompts/intent_classifier.py` | Day 6 | 意图分类对比 — Zero-shot / Few-shot / CoT |
| `prompts/tools.py` | Day 7 | ReAct Agent 工具定义 |
| `prompts/react_agent.py` | Day 7 | ReAct Agent — Thought-Action-Observation 循环 |
| `prompts/extractor_langchain.py` | Day 8 | LangChain LCEL 重写信息抽取 |

### RAG 系统（Day 9–19）

| 文件 | Day | 说明 |
|------|-----|------|
| `pdf_splitter.py` | Day 9 | PDF 加载 + 三种 chunk_size 分块策略对比 |
| `embedding_chroma.py` | Day 10 | BGE Embedding + Chroma 向量库存储与检索 |
| `rag_qa.py` | Day 11 | 初版 RAG 问答 — 检索→拼接→生成 |
| `llamaindex_demo.py` | Day 11 | LlamaIndex 框架对比实验 |
| `bm25_retriever.py` | Day 13 | BM25 关键词检索 — jieba 分词 + 稀疏检索 |
| `hybrid_retriever.py` | Day 14 | 混合检索 — 向量 + BM25 加权融合 (α=0.5) |
| `query_rewriter.py` | Day 15 | 查询改写 — LLM 模糊口语→专业检索短语 |
| `build_index.py` | Day 17 | 批量构建三种切片策略的 Chroma 索引 |
| `rag_final.py` | Day 18 | **RAG 最终版** — 查询改写 + 混合检索 + 生成 |
| `rag_api.py` | Day 19 | **RAG API 服务** — FastAPI `/rag/chat` 接口 |
| `manual_eval.py` | Day 19 | **RAG 评估** — Faithfulness + Answer Relevancy |

### Agent 与 LangGraph 阶段（Day 28 之后）

在 RAG 基础上进一步探索 Agent：从 LangChain Agent 框架到自定义工具、多轮记忆与容错，再手写 LangGraph 状态图复现同样的循环，并沉淀为可服务的 FastAPI 接口。

| 文件 | Day | 说明 |
|------|-----|------|
| `agents/agent_basic.py` | Day 28 | Agent 入门 — `tool_calling_agents` + 计算器工具，AgentExecutor 自动跑 Thought→Action→Observation |
| `agents/agent_custom_tools.py` | Day 29 | 自定义三工具（天气 / 库存 / 用户信息），单/多工具串联调用 |
| `agents/agent_resilient.py` | Day 31 | 异常处理与熔断 — 工具连续失败禁用 + 步数上限 + 逐步日志 |
| `agents/agent_memory.py` | Day 32 | 对话记忆 — 全局 `messages` 历史实现多轮上下文 |
| `agents/agent_langgraph.py` | Day 33 | LangGraph 重写 Agent — `StateGraph` + `ToolNode` + `add_messages` |
| `agents/agent_travel.py` | — | LangGraph 旅游规划 — 意图识别 / 条件分支 / LLM 生成行程 / 人工确认 |
| `agents/agent_api.py` | — | Agent 服务化 — FastAPI `/agent/chat`，多会话 + 线程锁并发安全 |

### 数据与配置

| 文件 | 说明 |
|------|------|
| `testset_v2.json` | 10 个问答对的全面测试集 |
| `testset.json` | 4 题初始测试集 |
| `requirements.txt` | Python 依赖清单 |
| `.env.example` | 环境变量模板 |
| `REPORT.md` | RAG 优化报告 — 切片/检索对比指标与踩坑记录 |

## ⚙️ 快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/orange376/ai-agent-challenge.git
   cd ai-agent-challenge
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置 API Key**
   ```bash
   cp .env.example .env
   # 编辑 .env 填入 DEEPSEEK_API_KEY
   ```

4. **构建索引**（首次使用 RAG 前）
   ```bash
   python build_index.py
   ```

5. **启动 RAG API 服务**
   ```bash
   uvicorn rag_api:app --reload --port 8000
   ```

6. **运行评估**
   ```bash
   python manual_eval.py
   ```

7. **（可选）启动多会话 Agent 服务**
   ```bash
   uvicorn agents.agent_api:app --reload --port 8001
   ```
   调用示例：
   ```bash
   curl -X POST http://127.0.0.1:8001/agent/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id": "u1", "question": "北京的天气怎么样？"}'
   ```

---

> 📺 项目视频介绍：https://www.bilibili.com/video/av116975447580144/?vd_source=b1df6ecd6f9d6fd9571ffe14f617080a
