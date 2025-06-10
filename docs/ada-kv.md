---
title: Ada-KV
---

# [2407.11550] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference

> [!NOTE]
> 将所有attention head的注意力权重铺平，然后按照每个头入选全局Top-B的权重数量来分配预算。
> 用滑动窗口内部的token评估每个head的其他token重要性，并据此分配每个head的预算。

1.  **引言与背景 (Introduction & Background)**
    *   **问题**: 大语言模型 (LLM) 在处理长序列推理时，因 Key-Value (KV) cache 持续增长而面临效率挑战。
    *   **现有方法**: 近期工作通过在运行时驱逐 (evicting) 大量非关键的 cache 元素来减小 KV cache 大小，同时保持生成质量。
    *   **现有方法的局限性**: 这些方法通常在所有 attention heads 之间统一分配压缩预算，忽略了每个 head 独特的注意力模式。

2.  **理论分析与方法动机 (Theoretical Analysis & Motivation)**
    *   **理论贡献**: 建立了一个驱逐前后 attention 输出之间的理论损失上界 (theoretical loss upper bound)。
    *   **理论作用**:
        1.  解释了先前 cache 驱逐方法的优化目标。
        2.  为自适应预算分配的优化提供了指导。

3.  **提出的方法：Ada-KV (Proposed Method: Ada-KV)**
    *   **核心思想**: 基于上述理论分析，提出首个 head-wise 的自适应预算分配策略 Ada-KV。
    *   **主要特点**:
        *   具有即插即用 (plug-and-play) 的优点，能够与现有的 cache 驱逐方法无缝集成。

4.  **实验评估 (Evaluation)**
    *   **数据集 (Datasets)**: 在 Ruler (13个数据集) 和 LongBench (16个数据集) 上进行了广泛评估。
    *   **评估场景 (Scenarios)**: 实验覆盖了 question-aware 和 question-agnostic 两种场景。
    *   **实验结果**: 结果表明，与现有方法相比，Ada-KV 带来了显著的质量提升。

## Introduction

好的，这是根据您提供的论文文本提取的中文大纲：

1. 背景：长序列推理中的KV Cache挑战
- **LLM的长序列处理能力**：自回归LLM在多种NLP应用中取得成功，其处理序列长度的能力显著增长，例如GPT支持128K token，Gemini-Pro-1.5支持高达2M token。
- **KV Cache带来的效率问题**：
    1.  **GPU内存效率**：长序列导致KV cache急剧膨胀（例如，8B模型处理2M token需256GB cache），严重消耗GPU显存。
    2.  **运行时效率**：在decoding阶段，巨大的cache增加了I/O延迟，降低了推理速度。
- **推理的两个阶段**：
    1.  **Prefilling**：计算并存储输入prompt中所有token的KV cache。
    2.  **Decoding**：自回归地使用最新生成的token从cache中检索信息，迭代生成输出。

2. 现有Cache Eviction方法的局限性
- **现有方法**：通过保留cache元素的子集来限制cache大小，大多依赖基于注意力权重的Top-k选择策略来决定保留或丢弃哪些元素。
- **核心局限性：统一预算分配 (Uniform Budget Allocation)**：
    - **问题**：现有方法为每个attention head分配相同的cache预算。
    - **忽略特性**：不同的attention head表现出多样的注意力模式，有些是稀疏集中的（sparse concentration），有些是广泛分散的（dispersed distribution）。
    - **导致低效**：
        - 对于注意力稀疏的head，cache预算被浪费。
        - 对于注意力分散的head，预算不足导致严重的eviction损失，影响生成质量。

3. Ada-KV：自适应预算分配策略
- **核心思想**：提出首个自适应预算分配策略Ada-KV，旨在动态地将预算从注意力稀疏的head重新分配给注意力分散的head。
- **理论指导**：
    1.  **Eviction Loss分析**：本文首先分析并证明了Top-k eviction方法在特定预算分配下，等价于最小化eviction前后注意力输出损失的一个上界。
    2.  **设计原则**：Ada-KV的设计目标是最小化这个理论上的eviction loss上界，从而根据各head的注意力模式进行原则性的自适应预算分配。
- **特性**：
    - **即插即用 (Plug-and-Play)**：Ada-KV可作为一个模块，轻松集成并增强现有的Top-k eviction方法。

4. 实验与评估
- **集成案例**：将Ada-KV与两种SOTA方法SnapKV和Pyramid集成，形成了Ada-SnapKV和Ada-Pyramid。
- **Benchmarks**：在两个综合性benchmark（Ruler和LongBench，共包含29个数据集）上进行评估。
- **评估场景 (Scenarios)**：
    1.  **Question-aware**：在压缩时利用问题信息的常见场景下，Ada-KV能有效提升性能。
    2.  **Question-agnostic**：在不依赖问题信息进行压缩的、更具挑战性的场景下，Ada-KV展现出更显著的优势。
- **实现**：通过高效的CUDA kernel实现，确保了其兼容性和性能。

5. 主要贡献总结
- **自适应预算分配**：识别并解决了当前KV cache eviction方法中统一预算分配的局限性，提出了首个自适应策略Ada-KV，提高了cache预算的利用效率。
- **理论洞见**：建立了cache eviction的理论框架，定义了eviction loss及其上界，不仅解释了现有方法的优化目标，也为Ada-KV的自适应设计提供了理论指导。
- **实证进展**：通过将Ada-KV集成到SOTA方法中，在两大benchmark的29个数据集上，以及在question-aware和question-agnostic两种场景下，均取得了显著的性能提升。

## Related works

1.  **背景：长序列推理中的KV Cache问题**
    *   **问题描述**：在长序列推理中，巨大的KV cache会导致内存瓶颈和高I/O延迟。
    *   **主流解决方案**：通过驱逐（eviction）非关键的cache元素来减小其规模。

2.  **KV Cache驱逐方法**
    *   **2.1 滑动窗口驱逐 (Sliding Window Eviction)**
        *   **代表方法**：StreamingLLM。
        *   **工作原理**：简单地保留最初的几个cache元素和滑动窗口内的元素，驱逐其余部分。
        *   **缺点**：无差别的驱逐方式会导致生成质量显著下降。

    *   **2.2 Top-k驱逐 (Top-k Eviction)**
        *   **核心思想**：基于注意力权重识别并保留$k$个最关键的cache元素，以保证驱逐后的生成质量。
        *   **方法演进**：
            1.  **早期工作 (FastGen)**：根据注意力头的特点，组合多种策略（如保留特殊符号、标点、近期元素和Top-$k$元素）。
            2.  **代表性工作 (H2O)**：利用**所有**token的query states来识别关键cache元素。
                *   **缺陷**：在LLM的单向注意力掩码下，这种全局聚合方式常导致近期的KV cache被错误驱逐，损害后续生成质量。
            3.  **近期SOTA方法 (SnapKV, Pyramid)**：通过使用一个“观察窗口”（observation window）内的query states来识别关键元素，解决了错误驱逐近期元素的问题，达到了SOTA性能。
        *   **现有Top-k方法的共同局限性**：
            *   通常将总预算（overall budget）**均匀地**分配给不同的注意力头（heads）。
            *   这种方式会导致预算的**错误分配**（misallocation）。

## Methodology

### Algorithm 1 Ada-KV: Adaptive Budget Allocation

**Input**: total budget $B$, attention weights for each head $i$ $\{A_i\}$;  
**Output**: allocated budgets $\{B_i^*\}$  

1. Concatenate all attention weights across heads $A = \text{Cat}(\{A_i\})$  
2. Select top $B$ weights from $A$: $\text{Top-k}(A, k = B)$  
3. Count the number of selected weights for each head $i$: $\{f_i\}$  
4. Set the allocated budgets as $\{B_i^* = f_i\}$  
   Return allocated budgets $\{B_i^*\}$  

### Ada-SnapKV/Ada-Pyramid in One Layer

**Input**: total budget $B$, tokens in observation window $X^{win} \in \mathbb{R}^{win \times d}$, cache in observation window $\{K_i^{win}, V_i^{win}\}$, cache outside observation window $\{K_i, V_i\}$

**Output**: retained cache $\{\hat{K}_i, \hat{V}_i\}$

1. **for** $i \leftarrow 1$ to $h$ **do**
   1. $Q_i^{win} = X^{win} W_i^Q$
   2. $\bar{A}_i = \text{softmax}(Q_i^{win} K_i^T)$
   3. $\bar{A}_i = \bar{A}_i.\text{maxpooling}(dim=1).\text{mean}(dim=0)$
2. **end for**

3. $B = B - \text{winsize} \times h$

4. Derive budget allocation $\{B_i^*\}$ using Algorithm 1$(B, \{\bar{A}_i\})$

5. Safeguard $\{B_i^*\} = \alpha \times \{B_i^*\} + (1 - \alpha) \times (B / h)$

6. Determine the Top-$k$ eviction decision $\{\mathcal{I}_i^*\}$ based on $\{B_i^*\}$

7. Select $\{\hat{K}_i, \hat{V}_i\}$ from $\{K_i, V_i\}$ according to $\{\mathcal{I}_i^*\}$

8. $\{\hat{K}_i, \hat{V}_i\} = \text{Cat}(\{\hat{K}_i, \hat{V}_i\}, \{K_i^{win}, V_i^{win}\})$

**Return** retained cache $\{\hat{K}_i, \hat{V}_i\}$

### Implementation of Computation under Adaptive Budget Allocation

1.  **可变长度注意力与可变大小缓存元素 (Variable-length Attention with Variable-sized Cache Elements)**
    *   **挑战**：自适应缓存分配（Adaptive allocation）导致不同注意力头（attention heads）的缓存元素大小可变，给高效计算带来困难。
    *   **解决方案**：采用可变长度的 FlashAttention 技术来支持自适应分配下的高效计算。
    *   **技术实现**：
        1.  **扁平化缓存存储布局 (flattened cache storage layout)**：将一个层内所有注意力头的缓存连接（concatenate）成单一的 tensor 结构。
        2.  **自定义 CUDA kernel**：与扁平化布局结合，实现高效的缓存更新操作。
    *   **效果**：这些组件协同工作，使得自适应分配下的计算效率能与常规 FlashAttention 相媲美。

2.  **与分组查询注意力 (GQA) 的兼容性 (Compatibility with Group Query Attention)**
    *   **背景**：GQA 技术已被 Llama 和 Mistral 等 SOTA LLM 广泛用于减小 KV cache 的大小。
    *   **问题**：现有的 KV cache 驱逐方法（如 SnapKV、Pyramid）不兼容 GQA，它们会在不同的头之间冗余地复制分组的 KV cache，未能利用 GQA 的效率优势。
    *   **解决方案**：实现了一种简单的 GQA 兼容驱逐方法。
    *   **实现方式**：使用每个组内的**平均注意力权重 (mean attention weight)** 作为选择标准，从而消除冗余。
    *   **效果**：使 SnapKV、Pyramid 及其自适应变体（Ada-SnapKV 和 Ada-Pyramid）能够在 GQA 模型上实现显著的缓存大小缩减，例如在 Llama-3.1-8B 模型中可减少4倍。