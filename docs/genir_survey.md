---
title: From Matching to Generation
---

# From Matching to Generation: A Survey on Generative Information Retrieval

1. 引言
   - **传统信息检索 (Information Retrieval, IR) 系统**：依赖相似性匹配。
   - **生成式信息检索 (Generative Information Retrieval, GenIR)**：作为一种新兴范式，随着预训练语言模型的进步而受到越来越多的关注。
   - GenIR 的主要研究方向：
     1. **生成式文档检索 (Generative Document Retrieval, GR)**：利用生成模型的参数记忆文档，通过直接生成相关文档标识符进行检索，无需显式索引。
     2. **可靠响应生成 (Reliable Response Generation)**：利用语言模型直接生成用户所需信息，打破了传统IR在文档粒度和相关性匹配方面的限制，同时提供了灵活性、高效性和创造性。
2. 本文目标与内容
   - 系统回顾 GenIR 领域的最新研究进展。
   - **总结 GR 的进展**：涉及模型训练、模型结构、文档标识符、增量学习等方面。
   - **总结可靠响应生成的进展**：涉及内部知识记忆、外部知识增强等方面。
   - 回顾 GenIR 系统的评估、挑战和未来发展。
   - 为研究人员提供全面的参考，鼓励 GenIR 领域的进一步发展。