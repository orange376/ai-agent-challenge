# AI Agent 应用开发学习项目

涵盖 Prompt 工程、RAG 系统、ReAct Agent 和 LangChain/LlamaIndex 双框架实践，共 **19 天** 循序渐进。

## 🏆 项目亮点

- **五轮文本抽取 Prompt 迭代**：信息抽取准确率从 78% 提升至 96%，完整记录在 `prompts/extractor_log.md`
- **CoT vs Few-shot 对比实验**：用 10 个测试用例量化三种策略差异，证明 CoT 在模糊意图上准确率领先 30%
- **ReAct 智能代理**：手动实现 Thought-Action-Observation 循环，结合 `stop` 参数消除幻觉
- **完整 RAG 优化链**：切片策略对比 → 混合检索 → 查询改写 → 综合评估，Answer Relevancy 从 0.75 提升至 1.00
- **10 题全面测试集**：覆盖直接匹配、跨段落推理、对比总结、反幻觉、部分信息覆盖五种挑战类型
- **FastAPI 服务化**：`/rag/chat` 接口，返回答案 + 参考来源 + 置信度，附带请求日志
- **双框架对比**：LangChain（组件化精细控制）与 LlamaIndex（封装度高、代码少）的实践感受

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

### 数据与配置

| 文件 | 说明 |
|------|------|
| `testset_v2.json` | 10 个问答对的全面测试集 |
| `testset.json` | 4 题初始测试集 |
| `requirements.txt` | Python 依赖清单 |
| `.env.example` | 环境变量模板 |

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
