## Day 21 RAG 系统 API 化

- **接口**：POST `/rag/chat`，请求体 `{"question": "..."}` 
- **响应**：`answer`（回答）、`sources`（参考文本块）、`confidence`（置信度）、`rewritten_query`（改写后查询）
- **日志**：请求时间、问题、耗时记录到 `rag_api.log`
- **测试**：通过 Swagger 验证，接口运行稳定'