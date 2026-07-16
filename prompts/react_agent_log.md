## ReAct Prompt 迭代日志

### V1：基础格式引导
- **设计思路**：通过格式约束（Thought/Action/Observation/Final Answer）让模型自行生成推理和行动。
- **问题**：模型会生成虚假的 Observation，因为它无法真正执行函数。
- **效果**：格式基本遵循，但 Observation 是编造的，不可信。
- **改进方向**：需要在外部程序里解析 Action，实际调用工具，将真实 Observation 注入 Prompt。


### V2：引入 API stop 参数彻底消除幻觉 Observation
- **问题**：V1 中模型会自行编造 Observation
- **解决**：利用 DeepSeek API 的 `stop=["Observation:"]` 参数，强制模型在生成 Action 后立即停止，杜绝自行生成 Observation 的可能。
- **效果**：每一步的 Observation 完全由程序注入真实工具数据，模型不再产生幻觉。
- **代码改动**：call_llm 增加 stop 参数支持；react_agent 调用时传入 stop=["Observation:"]。

### V3：引入 API stop 参数彻底消除幻觉 Observation
- **完全消除了幻觉 Observation**，每一步 Observation 都是真实工具返回值。
- **思考-行动-观察循环稳定**，模型能基于真实数据持续推理。