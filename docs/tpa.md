---
title: Tensor Product Attention
---

# Tensor Product Attention Is All You Need

## Abstract

1. **研究背景与挑战**
   - **长序列建模的内存瓶颈**：扩展语言模型处理长输入序列时需依赖大规模键值缓存（KV cache），导致推理阶段内存开销显著增加。
   
2. **Tensor Product Attention (TPA)**
   - **核心思想**：通过张量分解技术对查询（query）、键（key）、值（value）进行低秩紧凑表示，显著压缩推理时的KV缓存规模。
   - **上下文因子分解**：将表示分解为上下文相关的低秩成分（contextual low-rank components），动态适应输入内容。
   - **与RoPE的融合**：无缝集成旋转位置编码（RoPE），在提升内存效率的同时保持模型质量。
   
3. **模型架构：Tensor ProducT ATTenTion Transformer (T6)**
   - **基于TPA的架构设计**：提出专为序列建模优化的新型Transformer变体，命名为T6。
   
4. **实验评估**
   - **基线对比方法**：对比标准Transformer变体，包括MHA（多头注意力）、MQA（多查询注意力）、GQA（分组查询注意力）及MLA（多级注意力）。
   - **评估指标**：
     1. 困惑度（Perplexity）
     2. 主流基准测试（如GLUE、SuperGLUE等）性能
   - **内存效率验证**：在固定资源约束下，验证TPA支持更长序列处理的能力。
   - **关键结果**：T6在多项指标上超越基线方法，同时降低内存占用。
   
5. **核心贡献与意义**
   - **技术突破**：通过张量分解实现内存效率与模型质量的协同优化，缓解现代语言模型的可扩展性挑战。
   - **开源代码**：https://github.com/tensorgi/T6。

## Introduction

1. **背景与挑战**  
   - **KV缓存瓶颈**：LLM推理时键值缓存（KV cache）内存消耗随序列长度线性增长，受限于硬件资源。  
   - **现有方案缺陷**：  
     1. **稀疏注意力**：通过剪枝/压缩缓存，但可能丢失关键信息（如LeetCode验证依赖完整上下文）。  
     2. **多查询注意力（MQA）与分组查询注意力（GQA）**：共享键值降低内存需求，但牺牲灵活性或需修改架构。  
     3. **低秩微调（LoRA）**：优化训练内存，但未解决推理阶段KV缓存主导的开销。  
     4. **多头潜在注意力（MLA）**：压缩键值表示，但需额外位置编码参数且与RoPE不兼容。  

2. **Tensor Product Attention (TPA)**  
   - **核心机制**：  
     1. 使用高阶张量分解动态因子化查询（$𝐐$）、键（$𝐊$）、值（$𝐕$）的激活状态，构建低秩上下文表征。  
     2. 相较标准多头注意力（MHA），推理时KV缓存内存减少$10 \times$以上。  
   - **优势**：  
     1. **性能提升**：预训练验证困惑度（perplexity）更低，下游任务表现优于MHA/MQA/GQA/MLA。  
     2. **统一视角**：MHA/MQA/GQA可视为TPA的非上下文特例。  

3. **T6模型架构**  
   - **设计目标**：基于TPA构建序列建模框架，平衡内存效率与建模能力。  
   - **实验验证**：  
     1. 语言建模任务中验证困惑度持续优化。  
     2. 下游任务性能随KV缓存缩减同步提升。  

4. **与RoPE的兼容性**  
   - **关键特性**：原生支持旋转位置嵌入（RoPE），无需额外参数调整。  
   - **应用价值**：可无缝替换LLaMA/Gemma等主流架构中的MHA层，降低部署门槛。

## Tensor Product Attention

### Tensor Factorization of Queries, Keys, and Values

1. **背景与标准注意力机制**  
   - **变量定义**：设序列长度为$T$，每个token的隐藏状态$\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$，多头注意力包含$h$个头，每个头维度$d_h$，满足$d_{\text{model}} = h \times d_h$。  
   - **标准注意力流程**：通过线性映射生成查询（$\mathbf{Q}$）、键（$\mathbf{K}$）、值（$\mathbf{V}$）张量，维度为$\mathbb{R}^{T \times h \times d_h}$。  

2. **上下文分解（Contextual Factorization, CF）**  
   - **核心思想**：将$\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t$分解为（上下文相关）张量积的加权和，各分解秩分别为$R_q, R_k, R_v$。  
     - **公式表示**：  
       $$
       \mathbf{Q}_t = \frac{1}{R_Q} \sum_{r=1}^{R_Q} \mathbf{a}_r^Q(\mathbf{x}_t) \otimes \mathbf{b}_r^Q(\mathbf{x}_t), \quad \text{其中} \ \mathbf{a}_r^Q \in \mathbb{R}^h, \ \mathbf{b}_r^Q \in \mathbb{R}^{d_h}
       $$
       （类似适用于$\mathbf{K}_t$和$\mathbf{V}_t$，公式3.2-3.3）。  
   - **张量积叠加**：通过加权求和生成最终的头维度切片$\mathbf{Q}_t \in \mathbb{R}^{h \times d_h}$。  

3. **潜在因子映射（Latent Factor Maps）**  
   - **因子生成**：通过线性变换$\mathbf{W}_r^{a_Q}, \mathbf{W}_r^{b_Q}$从$\mathbf{x}_t$生成$\mathbf{a}_r^Q$和$\mathbf{b}_r^Q$。  
   - **秩合并优化**：将分解秩$R_q$合并至输出维度，例如：  
     $$
     \mathbf{a}^Q(\mathbf{x}_t) = \mathbf{W}^{a_Q} \mathbf{x}_t \in \mathbb{R}^{R_q \cdot h} \rightarrow \text{重塑为} \ \mathbf{A}_Q(\mathbf{x}_t) \in \mathbb{R}^{R_q \times h}
     $$  
   - **最终分解**：通过$\mathbf{Q}_t = \frac{1}{R_Q} \mathbf{A}_Q^\top \mathbf{B}_Q$重建$\mathbf{Q}$张量（公式3.5）。  

4. **缩放点积注意力**  
   - **计算流程**：与传统Transformer一致，但基于分解后的$\mathbf{Q}, \mathbf{K}, \mathbf{V}$：  
     $$
     \text{head}_i = \text{Softmax}\left( \frac{1}{\sqrt{d_h}} \mathbf{Q}_i \mathbf{K}_i^\top \right) \mathbf{V}_i \in \mathbb{R}^{T \times d_h}
     $$  
   - **输出投影**：拼接所有头后通过权重矩阵$\mathbf{W}^O$映射回$\mathbb{R}^{T \times d_{\text{model}}}$（公式3.4-3.5）。  

5. **参数初始化**  
   - **Xavier初始化**：权重矩阵$\mathbf{W}_r^{a_Q}, \mathbf{W}_r^{b_Q}$等从均匀分布$\left[-\sqrt{6/(n_{\text{in}} + n_{\text{out}})}, \sqrt{6/(n_{\text{in}} + n_{\text{out}})}\right]$采样，以保持激活值与梯度方差稳定。

### RoPE Compatibility and Acceleration

1. **RoPE在多头注意力中的应用流程**
   - **标准计算步骤**：  
     1. 计算第$t$个token的查询矩阵$\mathbf{Q}_t$和第$s$个token的键矩阵$\mathbf{K}_s$（维度为$\mathbb{R}^{h \times d_h}$）。  
     2. 分别应用旋转位置编码（RoPE）得到$\widetilde{\mathbf{Q}}_t = \text{RoPE}_t(\mathbf{Q}_t)$和$\widetilde{\mathbf{K}}_s = \text{RoPE}_s(\mathbf{K}_s)$。

2. **RoPE与TPA的整合优化**  
   - **预旋转策略**：  
     1. 在张量并行注意力（TPA）分解中直接集成RoPE，例如对键表示进行预旋转：  
        $$
        \widetilde{\mathbf{B}}_K(\mathbf{x}_t) \longleftarrow \text{RoPE}_t\big(\mathbf{B}_K(\mathbf{x}_t)\big).
        $$  
     2. 生成预旋转后的键表示$\widetilde{\mathbf{K}}_t$，通过缓存消除解码时的显式旋转需求，加速自回归推理。  
   - **硬件适配性**：根据硬件和性能需求，训练与推理阶段可采用不同的RoPE整合方案。

3. **定理1：RoPE与TPA的兼容性**  
   - **数学形式化证明**：  
     1. 若查询矩阵$\mathbf{Q}_t$通过TPA分解为$\mathbf{Q}_t = \frac{1}{R_Q} \mathbf{A}_Q(\mathbf{x}_t)^\top \mathbf{B}_Q(\mathbf{x}_t)$，则RoPE作用后仍保持TPA结构：  
        $$
        \text{RoPE}(\mathbf{Q}_t) = \frac{1}{R_Q} \mathbf{A}_Q(\mathbf{x}_t)^\top \widetilde{\mathbf{B}}_Q(\mathbf{x}_t),
        $$  
        其中$\widetilde{\mathbf{B}}_Q(\mathbf{x}_t) = \text{RoPE}_t\big(\mathbf{B}_Q(\mathbf{x}_t)\big)$。  
     2. 对查询-键内积的保持性：  
        $$
        \text{RoPE}_{t-s}(\mathbf{Q}_t) \mathbf{K}_s^\top = \widetilde{\mathbf{Q}}_t \, \widetilde{\mathbf{K}}_s^\top.
        $$  
   - **关键结论**：  
     1. RoPE作为分块对角正交变换（矩阵$\mathbf{T}_t$）作用于$\mathbf{B}_Q(\mathbf{x}_t)$，保持$\mathbf{A}_Q(\mathbf{x}_t)$不变，且每列$\mathbf{B}_Q(\mathbf{x}_t)$被适当旋转，保留TPA结构。  
     2. 定理验证了TPA与RoPE的相对平移性质兼容性，证明详见附录C.1。

### KV Caching and Memory Reduction

### Unifying MHA, MQA, and GQA as Non-contextual TPA

### Computational Cost

### Model Architectures
