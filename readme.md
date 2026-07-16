# AI Agent 应用开发学习初期项目

涵盖 Prompt 工程、RAG 系统、ReAct Agent 和 LangChain/LlamaIndex 双框架实践。

##  项目亮点

- **五轮 文本抽取Prompt 迭代**：信息抽取准确率从 78% 提升至 96%，完整记录在 `extractor_log.md`
- **CoT vs Few-shot 对比实验**：用 10 个测试用例量化三种策略差异，证明 CoT 在模糊意图上准确率领先 30%
- **ReAct 智能代理**：手动实现 Thought-Action-Observation 循环，结合 `stop` 参数消除幻觉
- **完整 RAG 问答系统**：从 PDF 加载 → 文本分割 → 向量嵌入 → Chroma 检索 → DeepSeek 生成答案
- **双框架对比**：LangChain（组件化精细控制）与 LlamaIndex（封装度高、代码少）的实践感受

##  项目结构

| 文件 | 说明 |
|------|------|
| `llm_api.py` | 封装 DeepSeek API 调用，使用 requests 手写 HTTP 请求 |
| `main.py` | FastAPI `/chat` 接口，Pydantic 模型校验 |
| `prompts/extractor.py` | 信息抽取 Prompt 模板（V1 → V5 迭代） |
| `prompts/summarizer.py` | 长文摘要 Prompt 模板 |
| `prompts/debugger.py` | Bug 排查 Prompt 模板（CoT 三步推理） |
| `prompts/intent_classifier.py` | Zero-shot / Few-shot / CoT 对比实验 |
| `prompts/react_agent.py` | ReAct 模式 Agent，支持真实工具调用 |
| `prompts/extractor_langchain.py` | 用 LangChain LCEL 重写信息抽取 |
| `pdf_splitter.py` | PDF 加载 + RecursiveCharacterTextSplitter 对比实验 |
| `embedding_chroma.py` | 文本向量化 + Chroma 向量库存储与检索 |
| `rag_qa.py` | 完整 RAG 问答：检索 → 拼接 Prompt → 生成 |
| `llamaindex_demo.py` | LlamaIndex 快速体验 Demo |
| `prompts_log.md` | Prompt 版本迭代记录（V1→V5） |

##  快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/orange376/ai-agent-challenge.git
   cd ai-agent-challenge