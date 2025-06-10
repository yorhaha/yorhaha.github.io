---
title: OmniKV
---

# OmniKV: Dynamic Context Selection for Efficient Long-Context LLMs

> [!NOTE]
> 文章的大致意思很清晰，但是图和写作都非常令我迷惑。

1.  **引言与背景**
    *   **问题陈述**：在长上下文的大语言模型（LLM）推理阶段，KV cache 占用了大量 GPU 内存，并且其内存占用随序列长度的增长而增加。
    *   **现有方法的局限性**：
        *   先前研究通过识别注意力分数（attention scores）的稀疏性来丢弃不重要的 token，以减少 KV cache 的内存占用。
        *   **核心缺陷**：论文指出，注意力分数是基于当前隐藏状态计算的，因此无法预示一个 token 在未来生成迭代中的重要性。

2.  **OmniKV 方法**
    *   **定义**：提出了一种名为 OmniKV 的推理方法，它无需丢弃 token 且无需训练。
    *   **核心洞察**：在单次生成迭代（single generation iteration）中，连续的层（consecutive layers）所识别出的重要 token 具有高度的相似性。

3.  **性能与优势**
    *   **性能提升**：
        *   在不损失任何性能的前提下，实现了 1.68 倍的推理加速。
        *   非常适合与 offloading 技术结合，可将 KV cache 的内存使用量减少高达 75%。
    *   **实验验证**：
        *   在多个 benchmark 上实现了 SOTA（state-of-the-art）性能。
        *   在 CoT（Chain-of-Thoughts）场景中具有特别的优势。
    *   **实践价值**：
        *   显著扩展了模型的最大上下文处理能力。
        *   **具体案例**：将单张 A100 GPU 上 Llama-3-8B 模型支持的最大上下文长度从 128K 扩展到了 450K。

## Introduction

1. 问题与动机
- **KV Cache 的内存挑战**:
  - 在长上下文场景下，LLM 推理会产生巨大的 GPU 内存开销，其中一大部分源于为加速生成而设计的 KV cache。
  - KV cache 占用的内存随序列长度线性增长。例如，对于 Llama-3-8B 模型，在 128K 上下文长度和批处理大小为 8 的情况下，仅 KV cache 就占用超过 134GB 的 GPU 内存。
- **现有方法的局限性**:
  - **主流方案**: 以往研究试图基于注意力稀疏性，识别并丢弃累计注意力分数较低的 "不重要" token，从而减少 GPU 内存占用。
  - **核心缺陷**: 作者认为，在多步推理场景中，token 的重要性是动态变化的。注意力分数仅反映 token 对**当前**推理步骤的相关性。因此，丢弃低分 token 可能导致在**后续**推理步骤中丢失关键信息。

2. OmniKV 方法
- **核心思想**: 提出一种名为 OmniKV 的新型推理方法，它**保留所有 token 的 KV cache**，以避免信息丢失，同时在上下文长度超过 32K 时加速解码。
- **关键洞察 (Inter-Layer Attention Similarity)**: 对于一个特定的上下文，不同层中具有高注意力分数的 token 集合是高度相似的。
- **工作流程**:
  - **Prefill 阶段 (处理输入)**:
    1. 将**大部分层**的 KV cache 卸载 (offload) 到 CPU 内存。
    2. 在 GPU 中完整保留少数**过滤器 (filter) 层**的 KV cache。
  - **Decode 阶段 (生成新 token)**:
    1. 首先，利用“过滤器”层中的注意力稀疏性，通过 top-k 算法选出得分最高的少数 token。
    2. 接着，其他层直接使用这个由“过滤器”层选出的 token 子集作为上下文进行注意力计算。
    3. 这样，每个解码步骤只需从 CPU 加载一小部分 token 的 KV cache 到 GPU。
- **效率提升机制**:
  - **减少数据传输**: 由于多层共享相同的 token 索引，每次迭代只需进行极少数（≤ 3 次）的 GPU-CPU 数据传输。
  - **计算与传输重叠**: 使用异步传输 (asynchronous transfer) 来重叠计算与数据传输时间。
  - **缩短计算上下文**: 大部分层在更短的上下文上进行计算，从而加速解码。

## Related work

1.  **Token 丢弃与卸载 (Token Dropping and Offloading)**
    *   **基于注意力分数的 Token 丢弃**：
        *   **主流方法**：在 prefill 阶段后，根据累积的注意力分数丢弃不重要的 token。
        *   **缺陷**：可能会丢弃在未来推理步骤中变得重要的 token。
        *   **与 Quest 对比**：Quest 认识到动态选择的重要性，但它无法减少内存使用，并且由于用单个向量表示一个 block，可能会损害召回率。
    *   **KV cache 卸载**：
        *   **主流方法**：在 VRAM 不足时，将 KV cache 卸载到 CPU 内存。
        *   **缺陷**：未利用注意力中的稀疏性，导致通过 PCIe 传输的数据量过大。
        *   **与 InfLLM 对比**：InfLLM 将序列分块并选择代表性向量作为检索键，但选出的少数向量可能无法完全捕捉块内信息，导致召回率较低。
    *   **本文方法思路**：为保证信息无损，在每次生成迭代中动态选择 KV cache 的一个稀疏子集。

2.  **其他效率优化方法 (Other Efficient Methods)**
    *   **KV cache 压缩**：
        *   **方法**：利用 LLM 作为 auto-encoder 将上下文压缩为更短的序列。
        *   **例子**：ICAE、Gist。
    *   **提示词压缩 (Prompt Compression)**：
        *   **方法**：在语言层面直接压缩提示词，从而间接压缩 KV cache。
        *   **例子**：LLMLingua。
    *   **KV cache 量化 (Quantization)**：
        *   **方法**：类似于模型权重 量化，对 KV cache 进行量化。
        *   **例子**：KIVI、SmoothQuant。
    *   **与本文方法的关系**：这些压缩或量化方法与本文提出的方法是**正交 (orthogonal)**的，可以结合使用。

3.  **LLM 中的稀疏性 (Sparsity in LLMs)**
    *   **核心观察**：在长上下文场景中，注意力是稀疏的。例如，Minference 研究表明，在 128k 上下文中，仅需 4k token 即可累积 96.4% 的总注意力分数。
    *   **关键挑战**：稀疏模式是dynamic的，随生成迭代而变化。这似乎意味着需要在每一层、每一次迭代中都计算完整的注意力。
    *   **现有应对策略**：
        *   **近似注意力**：Quest 和 SparQ 使用近似注意力方法来规避高昂的完整注意力计算。
        *   **跨层相似性**：Infini-Gen 利用**连续两层**之间的相似性来预选关键 KV cache，但其加载时间仍可能超过计算时间，导致 GPU 空闲。
    *   **本文的新发现与贡献**：
        *   **新发现**：不仅是连续层，**不同层之间**的稀疏模式也表现出高度相似性。
        *   **本文方法**：仅计算少数几层的完整注意力，然后利用其稀疏模式来指导后续层，从而节省计算。
        *   **独创性**：本文是第一个强调并利用这一发现的研究。

## Insights

1.  **层内注意力稀疏性 (Intra-Layer Attention Sparsity)**
    *   **核心特性**: LLM层内的注意力矩阵是稀疏的，这意味着模型只需关注一小部分token子集，就能生成几乎等效的输出。
    *   **应用价值**: 此特性已被用于提升推理速度或减少GPU内存占用。
    *   **OmniKV的应用**: OmniKV利用此特性，在大多数层中仅使用一小部分token，从而同时减少了计算量和CPU与GPU之间的通信量。

2.  **层间注意力相似性 (Inter-Layer Attention Similarity)**
    *   **概念定义**: 在特定层中获得高注意力的一个固定token子集，在后续连续的多层中会持续保持其重要性。
    *   **“过滤器”能力 (Filter Ability)**: 层的相似性值可视为该层的“过滤器”能力。其计算方式为：一个固定的token子集在后续层中注意力分数的总和的平均值。
    *   **实验观察**: 经过一定数量的浅层网络后，某一层与后续n层的相似性会变得非常高。
    *   **“过滤器”层 (Filter Layers)**: 部分层展现出比其他层更强的“过滤器”能力。
    *   **OmniKV的应用**: 这些“过滤器”层在OmniKV中充当上下文选择器，为每个生成迭代识别关键token，从而为后续层实现稀疏注意力提供基础。

3.  **Token间注意力可变性 (Inter-Token Attention Variability)**
    *   **核心直觉**: 在LLM的生成过程中，重要的token集合是动态变化的，尤其是在多任务或多步推理场景（如CoT）中。
    *   **实例观察**: 在一个多跳（multi-hop）问题的CoT场景中，两个不同解码步骤中注意力得分最高的token集合（除BOS token外）是完全不同的。
    *   **实验验证**: 在Multi-Hop QA任务上的研究表明，在每个生成步骤中，一些在预填充阶段未被识别为关键的token（即不在最重要的25% token集合中），后续会获得非常高的注意力分数。
    *   **结论**: 关键token的子集在不同生成步骤之间存在显著波动。
    *   **对OmniKV的启发**: 基于此洞察，OmniKV选择保留完整的KV cache，以确保模型性能不受影响。

## Method

1.  **核心思想与目标**
    *   **定义**: OmniKV是一种无需训练（training-free）且不丢弃token（token-dropping-free）的推理方法。
    *   **目标**: 在多重推理（multi-reasoning）场景下，保持大型语言模型（LLM）的性能。
    *   **核心组件**:
        *   Context Bank
        *   Context Selector

2.  **在自回归LLM推理流程中的工作机制**
    *   **在Prefill阶段**:
        *   OmniKV初始化**Context Bank**。
        *   基于层间注意力相似性（inter-layer attention similarity），将大部分“非筛选层”（non-filter layers）的KV cache存储在CPU内存中。
    *   **在Decode阶段**:
        *   **Context Selector**作为一个即插即用（plug-and-play）模块，在少数“筛选层”（filter layers）上动态识别出重要的KV cache子集。
        *   **Context Bank**将这些选择传播到所有“非筛选层”（因为这些层共享相同的token索引）。
        *   随后，将选定的KV cache子集打包（in a pack）加载到GPU内存中。

通过上述机制，OmniKV有效降低了计算成本和数据传输开销。

### Context Bank

- **核心思想**：利用层间注意力（inter-layer attention）的相似性来预取（prefetch）重要的 token。
- **内存优化**：在GPU内存不足的情况下，可以从CPU内存异步预加载相应的KV cache，从而缓解内存限制。

**Prefill 阶段与 KV Cache**
- **KV Cache 生成**：在一个 $L$ 层的LLM中，通过对隐藏状态 $\mathbf{h}_i^p$ 应用注意力投影矩阵 $\mathbf{W}_i^k$ 和 $\mathbf{W}_i^v$，为长度为 $N$ 的 prompt 生成键值对缓存 $\{\mathbf{K}_i, \mathbf{V}_i\}_{i=1}^L$。
- **张量维度**：生成的键（Key）和值（Value）张量维度为 $\mathbb{R}^{H \times N \times d}$，其中 $H$ 是注意力头数，$N$ 是 prompt token 长度，$d$ 是每个注意力头的隐藏维度。

**OmniKV: 关键 Token 选择机制**
- **"Filter" 层的选择**：
    - **依据**：通过分析层间注意力相似度来确定哪些层最适合识别重要 token。
    - **策略**：为提升性能，使用一个超参数集合 $\mathbb{L}$（大小为 $m, m \leq 3$）来定义一组 "filter" 层。使用多个 "filter" 层理论上能让非 "filter" 层的子上下文（sub-context）具有更高相似度。
- **Token 选择流程**：
    - **"Filter" 层 ($i \in \mathbb{L}$)**：执行全注意力（full attention）来识别一个小的关键 token 子集 $\mathbf{T}_i$。
    - **浅层 ($l < \mathbb{L}_0$)**：由于浅层稀疏性较低，同样执行全注意力而不进行选择。
    - **稀疏注意力层 ($\mathbb{L}_i < l < \mathbb{L}_{i+1}$)**：仅使用由前一个 "filter" 层识别出的重要 token $\mathbf{T}_i$ 作为子上下文。
    - **重要 Token 传递机制**：
    $$
    \mathbf{T}_i =
    \begin{cases}
    \text{ContextSelector}(\mathbf{h}_i^w, \mathbf{K}_i) & \text{if } i \in \mathbb{L} \\
    \mathbf{T}_{i-1} & \text{otherwise}
    \end{cases}
    \quad \text{for } i \geq \mathbb{L}_0
    $$
    其中 $\mathbf{h}_i^w$ 是观测窗口（observation window）的隐藏状态。

**整体注意力机制与优化**
- **计算与数据传输交错**：为避免GPU等待，OmniKV对 "filter" 层的相邻层（即 $\{l+1\}_{l \in \mathbb{L}}$）也执行全注意力，从而将数据传输与计算重叠。
- **最终注意力公式**：
    $$
    \text{out}_i =
    \begin{cases}
    \text{Attention}_i(\mathbf{h}_i^l, \mathbf{K}_i, \mathbf{V}_i) & \text{if } i \in \mathbb{L} \text{ or } i-1 \in \mathbb{L} \text{ or } i < \mathbb{L}_0 \\
    \text{Attention}_i(\mathbf{h}_i^l, \mathbf{K}_i[\mathbf{T}_i], \mathbf{V}_i[\mathbf{T}_i]) & \text{otherwise}
    \end{cases}
    $$
    其中 $\mathbf{h}_i^l$ 是最后一个 token 的隐藏状态。
- **性能提升**：在稀疏注意力层，序列长度被显著减少到不足10%，从而降低了时间复杂度。

**内存管理与加载优化 (Packed Load)**
- **CPU-GPU 数据流**：在 "filter" 层识别出关键 token $\mathbf{T}_i$ 后，从CPU内存中检索后续稀疏层所需的KV cache子集（$\mathbf{K}_j[\mathbf{T}_i], \mathbf{V}_j[\mathbf{T}_i]$）。
- **Packed Load 机制**：
    - **原理**：由于 "filter" 层之间的多个连续稀疏层共享相同的子上下文 token 索引 $\mathbf{T}$。
    - **操作**：可以将这些稀疏层的KV cache打包，在最近的前一个 "filter" 层进行一次性加载（从CPU到GPU）。
    - **效果**：将加载次数减少到仅 $m$ 次（$m \leq 3$），显著降低了缓慢的PCIe传输开销。

### Context Selector

1.  **OmniKV 的 Token 选择框架**
    *   **核心思想**：OmniKV 在指定的 "filter" 层 $\mathbb{L}$ 中，基于一个分数向量 $\mathbf{S}_i \in \mathbb{R}^N$ 来选择重要的 tokens $\mathbf{T}_i$。
    *   **分数计算**：分数 $\mathbf{S}_i$ 是使用一个观察窗口 $\mathbf{h}_i^w$ 计算得出的。

2.  **Token 选择的具体步骤**
    *   **第一步：计算注意力分数（Attention Scores）**
        *   使用局部窗口 $\mathbf{h}_i^w$ 作为 query 状态，完整上下文 $\mathbf{h}_i^c$ 作为 key 状态。
        *   计算 query $\mathbf{Q}_i = \mathbf{W}_i^q \mathbf{h}_i^w$ 和 key $\mathbf{K}_i = \mathbf{W}_i^k \mathbf{h}_i^c$。
        *   通过以下公式计算注意力分数 $\mathbf{A}_i$：
            $$
            \mathbf{A}_i = \text{Softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d}}\right)
            $$
    *   **第二步：计算聚合分数（Score Vector）**
        *   首先，对所有 attention heads 应用 `reduce-max` 操作，获得每个 token 的最大注意力分数。
        *   然后，使用一个权重向量 $\alpha$ 对上述分数进行加权求和，得到最终的分数向量 $\mathbf{S}_i$。
        *   计算公式为：
            $$
            \mathbf{S}_i = \sum_{j=0}^{|\mathbf{h}_i^w|-1} \alpha_j \max_{0 \leq h < H} \mathbf{A}_i[h, j]
            $$
    *   **第三步：识别重要 Tokens**
        *   利用 `topk` 函数在分数向量 $\mathbf{S}_i$ 上选出得分最高的 $k$ 个 tokens 作为重要 tokens $\mathbf{T}_i$。
        *   计算公式为：
            $$
            \mathbf{T}_i = \arg \text{top } k(\mathbf{S}_i)
            $$

3.  **权重向量 $\alpha$ 的探索**
    *   **研究目的**：探究观察窗口中的哪些 tokens 具有更强的“过滤”能力来识别重要 tokens $\mathbf{T}$。
    *   **三种方法**：
        1.  **Uniform (均匀权重)**：
            *   公式：$\alpha = \{1\}_{i=0}^{|\mathbf{h}_i^w|}$。
            *   含义：窗口中的每个 token 在加权求和时贡献相等。
        2.  **Exponential (指数权重)**：
            *   公式：$\alpha = \{2^{i - |\mathbf{h}_i^w|}\}_{i=0}^{|\mathbf{h}_i^w|}$。
            *   含义：越靠近窗口末尾的 token 贡献越高。
        3.  **Last Token (末位 Token)**：
            *   公式：$\text{concat}(\alpha = \{0\}_{i=0}^{|\mathbf{h}_i^w|-1}, \{1\})$。
            *   含义：仅考虑窗口中最后一个 token 的注意力分数。

## Experiments

实验分析:

1.  **"Filter" 层的任务无关性 (Task-Independent Filter Layers)**
    *   **研究问题**：探究 "filter" 层的能力是依赖于特定任务，还是模型自身的固有特性。
    *   **实验与发现**：
        *   在 Llama-3-8B 和 Yi-9B 模型上跨多种任务进行实验。
        *   结果显示，不同任务下的相似度曲线趋势基本一致。
    *   **结论**："filter" 能力并非任务相关，而更可能是层（layer）自身的内在特性。因此，一旦选定合适的超参数 $\mathbb{L}$，该方法可以适应任何任务。

2.  **上下文选择的准确性 (Accuracy of Context Selection)**
    *   **研究问题**：哪些层能够更准确地识别出真正重要的 token？
    *   **实验与评估**：
        *   使用 CLongEval 数据集，测试模型层为包含答案的上下文分配更高注意力分数的能力。
        *   通过计算重要 token 在标准答案所在参考块（reference chunk）内的命中率（hit ratio）进行评估。
    *   **核心发现**：
        *   "filter" 能力更强的层表现出更高的命中率。
        *   **具体例子**：Llama3-8B 的第 8 层和 Yi-9B 的第 14 层，在 "filter" 能力和命中率两个指标上都呈现出峰值。
    *   **结论**：模型中的特定层在训练后形成了更强的关键 token 检索能力。

3.  **性能随 "Filter" 能力的变化 (Performance Vary with Filter Ability)**
    *   **研究问题**：模型性能是否随着 "filter" 能力的变化而变化？
    *   **实验与发现**：
        *   在 LongBench benchmark 上使用 Llama3-8B-262K 模型进行测试。
        *   结果表明，模型层的性能表现与 "filter" 能力高度对应。
        *   **具体例子**：表现优越的层（如第 8, 10, 11, 13 层）正是 "filter" 能力较强的层。同时，性能的突变（如第 12 层的下降和第 4 到第 5 层的提升）也与 "filter" 能力的变化趋势一致。