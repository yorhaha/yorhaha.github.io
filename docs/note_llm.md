---
NoteLLM
---

# NoteLLM: A Retrievable Large Language Model for Note Recommendation

![](https://arxiv.org/html/2403.01744v2/x1.png)

1. **背景与挑战**  
   - **笔记推荐的重要性**：用户在社区分享笔记，推荐符合兴趣的笔记是核心任务。  
   - **现有方法局限性**：  
     1. **BERT-based模型问题**：仅依赖笔记内容生成嵌入，忽略关键概念（如hashtag/category）。  
     2. **LLM潜力**：LLMs在语言理解上显著优于BERT，但未被充分应用于笔记推荐。  

2. **NoteLLM框架**  
   - **核心思想**：统一框架结合LLM解决item-to-item (I2I)笔记推荐。  
   - **关键组件**：  
     1. **Note Compression Prompt**：将笔记压缩为单个特殊token（如`<note>`），实现高效表征。  
     2. **对比学习（Contrastive Learning）**：通过正负样本对齐学习相关笔记的嵌入。  
     3. **指令调优（Instruction Tuning）**：  
       - **自动摘要生成**：利用LLM生成笔记摘要。  
       - **Hashtag/Category生成**：通过任务指令引导模型预测关键标签。  

3. **实验验证**  
   - **数据集与场景**：基于小红书（Xiaohongshu）真实场景验证。  
   - **性能提升**：  
     1. 相比在线baseline，推荐系统效果显著提升。  
     2. 验证了LLM在压缩、对比学习与多任务生成中的协同作用。
    
## Introduction

1. **研究背景与问题定义**  
   - **社交平台笔记推荐的重要性**：以小红书、Lemon8为例，UGC（用户生成内容）推荐通过个性化笔记提升用户参与度。  
   - **I2I推荐的核心挑战**：从海量笔记库中基于内容或协同信号检索相关笔记，但现有方法（如BERT-based模型）对标签/分类（hashtags/categories）的利用不足。  
   - **关键观察**：标签/分类浓缩笔记核心信息，与笔记嵌入生成过程本质相似，可互为监督信号增强表征学习。  

2. **NoteLLM框架设计**  
   - **统一多任务目标**：基于LLM（如LLaMA 2）联合优化I2I推荐与标签/分类生成任务，通过压缩笔记关键概念提升推荐性能。  
   - **Note Compression Prompt构造**：  
     1. 使用特殊token压缩笔记内容，同时生成标签/分类。  
     2. 通过共现分数（co-occurrence scores）构建相关笔记对（从用户行为数据统计）。  
   - **训练策略**：  
     1. **Generative-Contrastive Learning (GCL)**：以压缩token作为笔记嵌入，训练模型从负样本中识别相关笔记。  
     2. **Collaborative Supervised Fine-tuning (CSFT)**：监督生成标签/分类，强化关键概念提取能力。  

3. **技术贡献**  
   - **首个基于LLM的I2I推荐框架**：验证LLM在I2I任务中的有效性，揭示其增强推荐系统的潜力。  
   - **多任务学习机制**：通过标签/分类生成任务优化笔记嵌入，实验证明压缩概念学习对推荐质量的提升作用。  
   - **工业级验证**：在小红书离线实验与在线场景中验证框架有效性，证明其实际应用价值。  

4. **方法创新点**  
   - **标签/分类与嵌入的协同优化**：将标签生成任务融入推荐模型训练，突破传统I2I方法对内容理解的局限性。  
   - **压缩token驱动的对比学习**：利用LLM解码能力，通过对比学习直接优化嵌入空间，提升相关笔记检索精度。

## Related work

1. **I2I Recommendation**  
   - **核心定义**：基于目标物品从大规模物品池中生成排序列表，依赖预构建I2I索引或在线近似k近邻检索（Johnson et al., 2019）。  
   - **传统方法局限**：  
     1. **协同过滤依赖**：仅利用用户行为信号（Zhu et al., 2018），无法处理冷启动物品（cold-start items）。  
     2. **文本匹配演进**：从基于稀疏向量的关键词匹配（Robertson et al., 2009）转向深度学习嵌入表示（Mikolov et al., 2013；Devlin et al., 2018）。  
   - **LLM应用现状**：  
     1. 现有研究仅将LLM作为编码器生成嵌入（Jiang et al., 2023），未充分利用其生成能力。  
     2. **NoteLLM创新点**：通过LLM生成标签/类别（hashtags/categories）增强物品嵌入表示。  

2. **LLMs for Recommendation**  
   - **三大范式**：  
     1. **数据增强**：利用LLM知识库生成多样化数据（Xi et al., 2023），但依赖生成质量且需测试数据对齐。  
     2. **直接推荐**：通过提示工程（Wang et al., 2023b）或微调（Bao et al., 2023b）实现重排序（reranking），受限于上下文长度（仅数十候选）。  
     3. **编码器应用**：提取物品嵌入（Li et al., 2023），但忽略生成能力。  
   - **NoteLLM差异化**：在召回阶段（recall phase）集成LLM，并通过学习标签生成优化嵌入能力。  

3. **Hashtag/Category Generation from Text**  
   - **主流方法对比**：  
     1. **抽取式**：提取文本关键词（Zhang et al., 2016），无法获取未见标签。  
     2. **分类式**：视为文本分类任务（Zeng et al., 2018），受制于人工标签多样性。  
     3. **生成式**：端到端生成标签（Wang et al., 2019b），但局限于单一任务。  
   - **NoteLLM多任务框架**：  
     1. 联合I2I推荐与标签生成，利用任务相似性（task similarity）提升协同效果。  
     2. 通过LLM生成能力强化物品表征学习（representation learning）。
    
## Problem Definition

- **笔记池（note pool）**：定义为$\mathcal{N} = \{n_1, n_2, ..., n_m\}$，其中$m$为笔记总数。每个笔记$n_i = (t_i, tp_i, c_i, ct_i)$包含标题$t_i$、标签$tp_i$、类别$c_i$和内容$ct_i$。
- **I2I笔记推荐任务**：
 - **目标**：给定目标笔记$n_v$，基于LLM的检索器需从$\mathcal{N} \setminus \{n_v\}$中排序并推荐与$n_v$相似的top-$k$笔记。
- **任务关联性**：
 1. **标签生成（Hashtag generation）**：根据标题$t_i$和内容$ct_i$，利用LLM生成标签$tp_i$。
 2. **分类生成（Category generation）**：根据标题$t_i$、标签$tp_i$和内容$ct_i$，利用LLM生成类别$c_i$。

## Methodology

![](https://arxiv.org/html/2403.01744v2/x2.png)

1. **NoteLLM框架**
   - **组成**：包含Note Compression Prompt Construction、GCL、CSFT三个核心组件。
   - **整合机制**：通过LLMs隐藏状态融合**协同信号**（collaborative signals）与**语义信息**。
2. **Note Compression Prompt Construction**
   - **功能**：灵活管理I2I recommendation和hashtag/category generation任务。
   - **处理流程**：生成的prompt经tokenized后输入LLMs。
3. **GCL**（Graph Contrastive Learning）
   - **作用**：基于生成的压缩词（compressed word）隐藏状态执行**对比学习**（contrastive learning），提取协同信号。
4. **CSFT**（Collaborative Semantic Fine-Tuning）
   - **作用**：结合笔记的语义与协同信息，输出hashtag和category。
  
### Generative-Contrastive Learning

1. **背景与动机**
   - **传统LLM训练方法局限**：预训练LLM通过指令微调或RLHF增强语义能力，但推荐任务需协同信号（collaborative signals）辅助识别用户兴趣。
   - **协同信号缺失问题**：现有LLM未显式建模用户行为中的协同关系（如笔记共现模式）。
   - **GCL提出**：通过对比学习（contrastive learning）从全局视角建模笔记间关联性。

2. **协同信号建模**
   - **共现机制构建相关笔记对**：
     1. **假设基础**：频繁共同浏览的笔记具有潜在关联性。
     2. **数据统计**：基于一周用户行为数据，统计用户点击序列（如$n_A \to n_B$）。
     3. **加权共现分数**：
        $s_{n_A \to n_B} = \sum_{i=1}^{U} \frac{1}{N_i}$，其中$U$为用户数，$N_i$为第$i$个用户的点击总量。
     4. **异常值过滤**：移除共现分数高于$u$或低于$l$的笔记。
     5. **相关笔记选择**：保留过滤后得分最高的$t$个笔记作为正样本。

3. **NoteLLM训练**
   - **笔记表示生成**：
     1. **虚拟词压缩**：通过prompt压缩笔记信息生成虚拟词（virtual word）。
     2. **嵌入空间映射**：取[EMB]前一token的隐藏状态，经线性层映射到维度为$d$的嵌入空间。
   - **对比学习损失函数**：
     $$
     L_{cl} = -\frac{1}{2B} \sum_{i=1}^{2B} \log \frac{e^{sim(\boldsymbol{n}_i, \boldsymbol{n}_i^+) \cdot e^\tau}}{\sum_{j \in [2B] \setminus \{i\}} e^{sim(\boldsymbol{n}_i, \boldsymbol{n}_j) \cdot e^\tau}}
     $$
     - $\boldsymbol{n}_i$：第$i$个笔记的嵌入向量。
     - $\boldsymbol{n}_i^+$：其对应的正样本嵌入。
     - $sim(a, b) = \frac{a^\top b}{\|a\| \|b\|}$：余弦相似度。
     - $\tau$：可学习温度参数。
