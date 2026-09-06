# RAG 系统优化报告

## 1. 优化历程总览

| 阶段 | 优化项 | 技术方案 | 效果 |
|------|--------|----------|------|
| 基线（Day 11） | 初版 RAG | 纯向量检索 + 默认切片 | Faithfulness 1.00, Relevancy 0.75 |
| 优化1（Day 14） | 混合检索 | BM25 + 向量加权融合 (α=0.5) | 召回更全面，关键词匹配增强 |
| 优化2（Day 15） | 查询改写 | LLM 改写模糊查询为专业短语 | 修复“国内外研究”问题 Relevancy 0→1 |
| 优化3（Day 17） | 切片策略 | 512/50 最优参数 | 平衡上下文与信息密度 |
| 最终版（Day 19） | 全量整合 | 最优切片 + 混合检索 + 查询改写 | Faithfulness 0.85, Relevancy 1.00 |

## 2. 关键指标对比

### 切片策略对比（Day 17）

| 切片策略 | 文本块数 | Faithfulness | Answer Relevancy |
|----------|----------|--------------|------------------|
| 256 / 30 | 17 | 1.000 | 0.625 |
| **512 / 50** | **9** | **0.925** | **1.000** |
| 1024 / 100 | 5 | 0.425 | 1.000 |

> **结论**：512/50 在信息覆盖和上下文连贯之间取得最佳平衡。

### 检索方式对比（Day 18）

| 检索方式 | Faithfulness | Answer Relevancy |
|----------|--------------|------------------|
| 纯向量检索 | 1.000 | 1.000 |
| 纯 BM25 | 1.000 | 1.000 |
| 混合检索 | 0.875 | 0.800 |

> **结论**：小数据集下差异不大，但 Day 13 的“糖尿病 vs 糖尿病视网膜病变”案例已证明混合检索在关键词区分场景下的优势。

### 最终版 vs 初版（公共 4 题）

| 版本 | Faithfulness | Answer Relevancy |
|------|-------------|------------------|
| Day 12 初版 | 1.000 | 0.750 |
| Day 19 最终版 | 1.000 | 1.000 |

> **Answer Relevancy 提升 33%**，源于查询改写成功将模糊问题转为精准检索短语。

## 3. 技术栈

- **大模型**：DeepSeek Chat API
- **框架**：FastAPI, LangChain
- **向量数据库**：Chroma
- **Embedding**：BAAI/bge-small-zh-v1.5（本地免费）
- **关键词检索**：BM25 + jieba 分词
- **评估**：手动实现 Faithfulness / Answer Relevancy 评判




# 踩坑记录

## 坑1：PyPDF 加载器被弃用

**现象**：`langchain_community` 被标记为 sunset，`PyPDFLoader` 导入时出现 `DeprecationWarning`。

**解决**：改用 `pypdf.PdfReader` 直接解析 PDF，再手动构建 LangChain 的 `Document` 对象。代码更简洁，且无依赖风险。

**教训**：优先使用底层稳定的库（如 `pypdf`），避免过度依赖社区包的某个特定版本。

---

## 坑2：RAGAS 安装连环失败

**现象**：`pip install ragas` 失败，先是 `scikit-network` 编译报错（缺少 C++ 工具链），降级后又报 `langchain_community.chat_models.vertexai` 模块不存在。

**解决**：放弃 pip 安装，手动用 LLM 实现 Faithfulness 和 Answer Relevancy 评估逻辑。理解了指标的底层原理，反而比调库更有收获。

**教训**：环境问题不要死磕，评估的核心是“指标定义 + LLM 评判”，不一定需要特定库。

---

## 坑3：评估时 ZeroDivisionError

**现象**：运行 `manual_eval.py` 时报 `ZeroDivisionError: division by zero`，原因是 `faith_scores` 列表为空。

**原因**：修改生成答案代码块时，误删了 `for item in test_data:` 循环头，导致评估循环没有执行。

**解决**：补回循环头，确保 `question`、`ground_truth` 等变量在循环内被正确赋值。

**教训**：修改循环结构时务必检查缩进和变量作用域。

---

## 坑4：反幻觉问题被评估 LLM 误判

**现象**：问题“系统用什么编程语言？”答案“根据文档无法找到”，但评估 LLM 给出 Faithfulness 0.00。

**原因**：评估 Prompt 没有充分处理“正确答案就是无答案”的边界情况。

**解决**：在评估 Prompt 中显式增加规则：“如果回答如实说明信息不足，且上下文确实没有相关信息，应给满分。”

**教训**：评估系统本身也需要迭代优化，不能盲目信任自动评分。