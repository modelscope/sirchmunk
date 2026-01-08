# open-agentic-search
Open-Agentic-Search



## TODO-LIST
- [ ] [P0] Add core pipeline code
- [ ] [P0] Add rerank for Ripgrep
- [ ] [P1] Add cache mechanism
- [ ] [P1] Add Qwen3-VL-2B-Instruct for local-mode
- [ ] [P1] Support self-improvement based on histories
- [ ] [P1] MDP-based planning & execution
- [ ] [P0] KnowledgeCluster中的scripts，支持PIL生成画图（自行判断）
- [ ] [P0] 考虑去掉scan模块？ 直接搜索时动态创建FileInfo，以及高效利用sampling机制。
- [ ] [P0] 增加langextract依赖 ！ ！ ！
- [ ] [P0] 自动识别检索路径中的“死路”，回溯并切换搜索词
- [ ] [P1] 服务化，增加快捷入口/快捷键，类似Alfred



```text

1. scan模块：输入原始的文件夹or文件，输出每个文件的metadata
   - 支持多种文件格式的扫描（如PDF、Word、Excel等）
   - metadata schema定义：
        单文件的metadata，包括文件名、路径、大小、创建时间、修改时间、文件类型、内容摘要等；
        对文件内容的进一增强分析，类似PageIndex，必要时用LLM辅助生成摘要
   - tree-walking相关算法设计与实现 ？？？
   - 整体文件目录的metadata可视化（树结构，思维导图），过程中可实时显示

2. 分层检索模块：基于scan模块输出的metadata，设计并实现分层检索算法
   - 第一层：基于metadata的快速过滤与定位，例如：[file_meta1, file_meta2, ...]，使用hybrid检索快速定位具体的文件
   - 第二层：找到具体候选文件列表，结合内容的深入检索与匹配（实现层面可考虑第一、二层合并，直接hybrid检索）；在定位到关键词所在的句子情况下，做context外扩，回溯前后文，形成完整的context段落
   - 实现：
     (1) 结合ripgrep和embeddings-based检索，设计hybrid检索算法：
        可参考： Claude-Context：结合了Claude-Code和Embeddings检索的框架：https://github.com/zilliztech/claude-context
     (2) Zoom-In和Zoom-Out机制；

   - query信息稀疏，针对长文档，context限制的条件下，设计高效的自主探索（语义探针, probe）系统
   - llm增加另一个协处理器


3. 进阶： 多模态快速检索
    - 支持图片、视频等多模态文件的扫描与metadata生成
    - 多模态检索算法


4. search主链路：
    - QuickSearch：基于分层检索模块，快速返回top-k的context段落
    - DeepSearch：基于QuickSearch的结果，结合LLM进行深度理解与回答生成；支持自我反馈与改进
    - 以上两种search模式均支持Web Search数据源
    - 基于MDP的实现  （Search Engine Reinforcement Learning, SERL）
        （参考：TongyiDeepResearch中的MDP建模思路做扩展，并考虑使用Test time scaling： https://zhuanlan.zhihu.com/p/1976391891126862536）
        1. 状态（State）：描述当前检索阶段的上下文，例如：用户查询、已检索文档集合、当前会话历史、点击/跳转行为、当前页面/段落位置等。
        2. 动作（Action）：在给定状态下可执行的操作，如：执行关键词扩展、重排序、翻页、发起新 query、跳转到某文档/段落、终止检索等。
        3. 奖励（Reward）：衡量动作效用的数值信号，主要指标如下：
            - nDCG@k（Normalized Discounted Cumulative Gain）：衡量检索结果的相关性和排序质量，优化nDCG@k有助于提升用户满意度和检索效果。
            - 信噪比（SNR, Signal-to-Noise Ratio）：衡量有用信息与无用信息的比例，提升信噪比意味着减少冗余和噪声，提高检索结果的相关性和质量。
            - 信息熵（Information Entropy）：衡量信息的不确定性和多样性，优化信息熵有助于提供更丰富和多样化的检索结果，满足用户的不同需求。
            - 记忆系统使用率：类似Memory-R1的reward设计，如果答案成功使用到了记忆系统中的内容，则给予正向奖励，鼓励模型更好地利用已有知识。
        4. 转移概率（Transition）：状态转移的动态（可建模为确定性或概率性），取决于用户行为与系统响应的交互。（比如从request -> learnings forest root -> domain learnings node -> target files -> retrieve answers -> final answers）

        5. 实现：
            - 维护两个矩阵：发射矩阵（Emission Matrix）和转移矩阵（Transition Matrix），分别表示状态到观察的概率分布和状态之间的转移概率。以此来记录用户个性化行为，可以作为上下文输入到LLM中。
            - 其中，发射矩阵与query情况（难度/复杂度/与本地文档的相关性等）和用户历史行为有关
            - 转移矩阵与用户的检索路径和选择有关

5. 智能体自我进化
    1) 基于原始metadata，后台异步触发任务，用LLM学习得出 Learnings（子类MetaLearnings）， 使用树结构表征(可能是很多个树组成的森林，对知识进行分门别类)，对存储目标进行高度总结和结构化
        基于上述learnings forest，可以抽象出一个根节点，该根节点就是对整个corpus的高度总结和概括，可动态加载到模型上下文中（注意考虑动态回溯机制，及时刷新整个森林）
    2) 基于search的中间过程和最终结果，沉淀高质量的trajectories（包含检索路径、refine过程等），作为智能体未来决策的参考依据，即SearchLearnings，沉淀以后则称之为"经验回放库"(Experience Replay Library, ERL)
    3) 索引回溯机制
    - 基于检索答案，经过refine后的learnings，回写到metadata中，不断增强metadata的质量与覆盖率（需要注意避免重新扫描后，覆盖掉这些高质量的回写learnings）
    - 在某个topic的检索轮次完成后，模型可视情况自动发起reflection任务，对本次检索进行总结，并且进一步深入分析相关的文件,补充到learnings中。

    4) 将检索路径RL化，训练"Navigator"模型
    - 基于高质量的metadata和trajectories，训练一个Navigator模型，指导智能体更高效地进行分层检索
    - 优化的目标：参考上述主链路中的论述

```

Reference:
    1. Agentic Search RL: https://mp.weixin.qq.com/s/-9_jM-u2D4RHGX5xbrOl7A


核心关注： 信噪比 ！

Notes:
1. 基于蒙特卡洛重要性采样：
    输入：给定llm, 文档D，查询query  输出： D中跟query最相关的k个语义片段
    step1: 根据

2. 策略：如果模型自己最终reflection时，觉得回答不好，可以自动重新发起语义密度重采样，对region of interests进行重点采样，补充缺失语义。


v0.0.1 Key features:
* 以文件为中心
* 无需提前建索引，无视目标路径的大小，时间复杂度O(1)，原生适合large-scale文件系统
* 占内存极少（基于mmap读写），无需加载索引或向量数据库或图数据库等
* knowledge cluster动态生成，为后续自我进化打基础
* 个性化：基于用户的搜索历史和行为，动态调整knowledge cluster和搜索策略
* 增加对title的match和加权
* 蒙特卡洛采样过程，对于超大文件，需要单独处理，提高效率
* 蒙特卡洛采样过程，对应在应用端增加文件片段预览功能；在chat输出区域嵌入预览窗口

v0.0.2 Key Features:
* 多模态检索
* evidence和knowledge cluster中增加原文的位置引用
* 考虑跨语种的召回能力
