## Day 33 LangGraph 基础学习

**学习内容**：
- LangGraph 的核心概念：`StateGraph`, `Node`, `Edge`, `Conditional Edge`
- 用 LangGraph 重写计算器 Agent，完全复现手动循环的功能
- 理解 `ToolNode` 如何自动处理工具调用和 ToolMessage 生成
**出现问题**：
- 初次接触 `StateGraph` 时，对 `add_messages` reducer 的作用感到困惑，后理解其自动化状态更新机制
- 条件边的映射字典格式需严格匹配函数返回值，否则会报错