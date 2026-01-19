# Academic Literature on Agentic RAG Systems and Recursive Methods

This curated bibliography compiles **50+ high-quality academic papers** addressing the benchmarking and comparison of Agentic RAG systems versus long-context recursive methods. The collection spans foundational architectures, evaluation frameworks, benchmark datasets, and emerging research from top venues including NeurIPS, ICLR, ACL, EMNLP, and ICML, with emphasis on 2023-2025 publications.

---

## Foundational RAG and retrieval-augmented architectures

These seminal papers established the retrieve-then-generate paradigm that underlies all subsequent RAG research.

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
*Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela*  
NeurIPS 2020 (arXiv:2005.11401)  
The foundational RAG paper combining parametric memory (seq2seq models) with non-parametric memory (dense vector retrieval). Established the standard architecture for knowledge-intensive generation tasks.

**REALM: Retrieval-Augmented Language Model Pre-Training**  
*Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang*  
ICML 2020 (arXiv:2002.08909)  
Pioneered joint pre-training of retrievers with language models in an unsupervised manner. Achieves **4-16% improvement** on Open-QA benchmarks through end-to-end retrieval learning.

**RETRO: Improving Language Models by Retrieving from Trillions of Tokens**  
*Sebastian Borgeaud et al. (DeepMind)*  
ICML 2022 (arXiv:2112.04426)  
Demonstrates retrieval-enhanced transformers matching GPT-3 performance with **25× fewer parameters** using a 2 trillion token database. Introduces chunked cross-attention for efficient retrieval integration.

**Dense Passage Retrieval for Open-Domain Question Answering (DPR)**  
*Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih*  
EMNLP 2020 (arXiv:2004.04906)  
Foundational dense retrieval using dual BERT encoders. Outperforms BM25 by **9-19%** on passage retrieval, enabling the retrieval component of modern RAG systems.

**Atlas: Few-shot Learning with Retrieval Augmented Language Models**  
*Gautier Izacard et al. (Meta AI)*  
JMLR 2023 (arXiv:2208.03299)  
Demonstrates an 11B parameter retrieval-augmented model outperforming 540B parameter models. State-of-the-art on NaturalQuestions and MMLU benchmarks through sophisticated retrieval-generation fusion.

---

## Agentic RAG systems and architectures

These papers define the core patterns for LLM agents that dynamically retrieve, reason, and act.

### Self-reflective and adaptive retrieval

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**  
*Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi*  
ICLR 2024 Oral (arXiv:2310.11511)  
Introduces reflection tokens enabling LLMs to adaptively retrieve on-demand and self-critique outputs for relevance, support, and factuality. Addresses indiscriminate retrieval limitations in standard RAG.

**FLARE: Active Retrieval Augmented Generation**  
*Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig*  
EMNLP 2023 (arXiv:2305.06983)  
Forward-Looking Active REtrieval triggers retrieval only when low-confidence tokens are detected, using predicted future content as queries—a key agentic pattern for adaptive RAG.

### Reason-and-act frameworks

**ReAct: Synergizing Reasoning and Acting in Language Models**  
*Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao*  
ICLR 2023 (arXiv:2210.03629)  
Foundational paper establishing interleaved reasoning traces with task-specific actions. Enables LLMs to dynamically reason while interfacing with external environments (Wikipedia API, search engines). Basis for most modern agentic frameworks.

**Reflexion: Language Agents with Verbal Reinforcement Learning**  
*Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao*  
NeurIPS 2023 (arXiv:2303.11366)  
Agents verbally reflect on task feedback and store reflections in memory to improve subsequent attempts. Key contribution to self-reflective agentic patterns without gradient updates.

**Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models (LATS)**  
*Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang*  
NeurIPS 2023 (arXiv:2310.04406)  
Combines Monte Carlo Tree Search with LLM agents to unify reasoning, acting, and planning. Enables systematic exploration and evaluation of action sequences.

### Planning and decomposition patterns

**Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning**  
*Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim*  
ACL 2023 (arXiv:2305.04091)  
Addresses missing-step errors in CoT by first devising a plan to divide tasks into subtasks before execution. Directly influenced LangChain's Plan-and-Execute agent architecture.

**Understanding the Planning of LLM Agents: A Survey**  
*Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, Enhong Chen*  
arXiv 2024 (arXiv:2402.02716)  
Comprehensive taxonomy of LLM-Agent planning including Task Decomposition, Plan Selection, External Module planning, Reflection and Memory. Essential reference for planning approaches.

### Tool-augmented language models

**Toolformer: Language Models Can Teach Themselves to Use Tools**  
*Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom*  
NeurIPS 2023 (arXiv:2302.04761)  
Seminal paper demonstrating LMs learning tool use (calculator, search, translation, Q&A) in a self-supervised manner. Decides which APIs to call, when, and how to incorporate results.

**TALM: Tool Augmented Language Models**  
*Aaron Parisi, Yao Zhao, Noah Fiedel*  
arXiv 2022 (arXiv:2205.12255)  
Earlier foundational work combining text-to-text tool interface with "self-play" bootstrapping. Demonstrates tool augmentation significantly outperforming non-augmented LMs at equivalent scales.

**Augmented Language Models: A Survey**  
*Grégoire Mialon, Roberto Dessì, Maria Lomeli et al.*  
TMLR 2023 (arXiv:2302.07842)  
Comprehensive survey covering retrieval-augmented, tool-augmented, and reasoning-augmented approaches. Essential context for the ALM landscape.

---

## Multi-hop reasoning and interleaved retrieval

Papers addressing complex reasoning requiring multiple retrieval and reasoning steps.

**IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions**  
*Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal*  
ACL 2023 (arXiv:2212.10509)  
Each CoT step guides retrieval of additional evidence, creating mutual reinforcement. Improves retrieval by **up to 21 points** and QA by **up to 15 points** on HotpotQA, 2WikiMultihopQA, and MuSiQue.

**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**  
*Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning*  
ICLR 2024  
Hierarchical tree-structured retrieval through recursive embedding, clustering, and summarization. Enables retrieval at different abstraction levels, achieving **20% improvement** on QuALITY benchmark.

**AIR-RAG: Adaptive Iterative Retrieval for Retrieval-Augmented Generation**  
*Wenhan Han, Meng Fang, Jun Wang, Mykola Pechenizkiy et al.*  
Neurocomputing 2025  
Adaptive iterative framework optimizing document relevance and LLM alignment without complex retraining. Superior performance on TriviaQA, PopQA, and HotpotQA.

---

## Recursive methods and iterative refinement

These papers explore recursive, iterative, and self-improving approaches to generation.

### Chain-of-thought foundations and variants

**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**  
*Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou*  
NeurIPS 2022  
Foundational CoT paper demonstrating intermediate reasoning steps significantly improve arithmetic, commonsense, and symbolic reasoning. State-of-the-art on GSM8K with just 8 exemplars.

**Self-Consistency Improves Chain of Thought Reasoning in Language Models**  
*Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou*  
ICLR 2023  
Samples diverse reasoning paths and selects most consistent answer via majority voting. Dramatic improvements: GSM8K (**+17.9%**), SVAMP (**+11.0%**), AQuA (**+12.2%**).

**Tree of Thoughts: Deliberate Problem Solving with Large Language Models**  
*Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan*  
NeurIPS 2023 (arXiv:2305.10601)  
Generalizes CoT through tree-structured exploration with self-evaluation, lookahead, and backtracking. Game of 24 success rate: **4% (CoT) → 74% (ToT)**.

**Graph of Thoughts: Solving Elaborate Problems with Large Language Models**  
*Maciej Besta, Nils Blach, Ales Kubicek et al.*  
AAAI 2024  
Models reasoning as arbitrary graphs where thoughts are vertices and dependencies are edges. Enables synergistic thought combination with feedback loops. Improves sorting by **62% over ToT** while reducing costs by **>31%**.

### Self-refinement and bootstrapping

**Self-Refine: Iterative Refinement with Self-Feedback**  
*Aman Madaan, Niket Tandon, Prakhar Gupta et al.*  
NeurIPS 2023  
Iterative self-refinement where LLMs generate, provide feedback, and refine—all using a single LLM without additional training. Improves performance by **~20% absolute** across 7 diverse tasks.

**STaR: Self-Taught Reasoner - Bootstrapping Reasoning With Reasoning**  
*Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman*  
NeurIPS 2022  
Iterative bootstrapping where models generate rationales, fine-tune on correct ones, and repeat. Uses "rationalization" for failed problems. **+35.9%** over few-shot baseline on CommonsenseQA.

**Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs**  
*Bowen Jin, Chulin Xie, Jiawei Zhang et al.*  
ACL Findings 2024  
Iterative LLM reasoning on text-attributed graphs via reasoning, interaction, and execution cycles. Introduces GRBench with 1,740 questions across 10 domain graphs.

---

## Long context versus retrieval approaches

Research directly comparing long-context windows with retrieval augmentation strategies.

### Empirical comparisons

**Long Context vs. RAG for LLMs: An Evaluation and Revisits**  
*Xinze Li, Yixin Cao, Yubo Ma, Aixin Sun*  
arXiv 2024 (arXiv:2501.01880)  
Comprehensive comparison showing long-context generally outperforms RAG in QA benchmarks, especially Wikipedia-based questions. Summarization-based retrieval performs comparably while chunk-based retrieval lags.

**Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach**  
*Zhuowan Li et al.*  
arXiv 2024 (arXiv:2407.16833)  
Benchmarks RAG vs LC using latest LLMs. LC consistently outperforms when well-resourced, but RAG's lower cost is key advantage. Proposes **Self-Route** method routing queries based on self-reflection.

### Lost in the middle phenomenon

**Lost in the Middle: How Language Models Use Long Contexts**  
*Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang*  
TACL 2024 (arXiv:2307.03172)  
Seminal paper demonstrating **U-shaped performance curve**—performance degrades when relevant information is in the middle of long contexts. Best results when information is at beginning or end.

**Never Lost in the Middle: Mastering Long-Context Question Answering with Position-Agnostic Decompositional Training**  
*Junqing He, Kunhao Pan, Xiaoqun Dong et al.*  
arXiv 2023 (arXiv:2311.09198)  
Proposes Position-Agnostic Multi-step QA (PAM QA) training. Achieves **13.7% absolute gain** in shuffled settings, **21.5%** in passage retrieval.

**Found in the Middle: How Language Models Use Long Contexts Better via Plug-and-Play Positional Encoding (Ms-PoE)**  
*arXiv 2024 (arXiv:2403.04797)*  
Multi-scale Positional Encoding addresses lost-in-the-middle without fine-tuning through position index rescaling with multi-scale context fusion.

---

## Context compression and efficient attention

Methods for efficiently handling long contexts through compression and architectural innovations.

### Prompt and context compression

**LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**  
*Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu*  
EMNLP 2023 (arXiv:2310.05736)  
Coarse-to-fine prompt compression using budget controller and token-level iterative compression. Achieves **up to 20x compression** with minimal performance loss.

**LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**  
*Huiqiang Jiang et al.*  
ACL 2024 (arXiv:2310.06839)  
Extends LLMLingua for long contexts with document reordering and subsequence recovery. Boosts NaturalQuestions performance by **21.4%** with **4x fewer tokens**.

**In-context Autoencoder for Context Compression (ICAE)**  
*Tao Ge et al.*  
arXiv 2023 (arXiv:2307.06945)  
Compresses long contexts into compact memory slots using autoencoding and language modeling. Achieves **4x context compression** with ~1% additional parameters.

### Efficient attention mechanisms

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
*Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré*  
NeurIPS 2022 (arXiv:2205.14135)  
IO-aware exact attention using tiling to reduce memory transfers. **Up to 7.6x faster**, **20x more memory efficient**, enables **4x longer context**.

**Longformer: The Long-Document Transformer**  
*Iz Beltagy, Matthew E. Peters, Arman Cohan*  
arXiv 2020 (arXiv:2004.05150)  
Foundational efficient transformer combining local windowed attention with global attention. Scales linearly with sequence length, SOTA on WikiHop and TriviaQA.

**Big Bird: Transformers for Longer Sequences**  
*Manzil Zaheer, Guru Guruganesh et al.*  
NeurIPS 2020 (arXiv:2007.14062)  
Sparse attention combining random, sliding window, and global patterns. Reduces quadratic to linear complexity, handles **8x longer sequences**, proven Turing complete.

### Position embedding extensions

**LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**  
*Yiran Ding, Li Lyna Zhang et al.*  
ICML 2024 (arXiv:2402.13753)  
Extends context to **2048k tokens** via non-uniform RoPE rescaling with evolutionary search. Integrated into Microsoft Phi-3.

**Scaling Laws of RoPE-based Extrapolation**  
*ICLR 2024*  
Unified framework describing extrapolation performance, base value, and tuning context length relationships. Explains RoPE extrapolation issues via "critical dimension."

---

## RAG evaluation frameworks and metrics

Automated evaluation methodologies for assessing RAG system quality.

**RAGAS: Automated Evaluation of Retrieval Augmented Generation**  
*Shahul Es, Jithin James, Luis Espinosa Anke, Steven Schockaert*  
EACL 2024 (arXiv:2309.15217)  
Reference-free evaluation with metrics for **faithfulness, answer relevance, and context relevance**. Foundational work for automated RAG evaluation without human annotations.

**ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**  
*Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia*  
NAACL 2024 (arXiv:2311.09476)  
Uses synthetic training data to fine-tune lightweight LM judges. Evaluates context relevance, answer faithfulness, and answer relevance with prediction-powered inference.

**RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation**  
*Ruiyang Ren et al. (Amazon Science)*  
NeurIPS 2024 Datasets Track (arXiv:2408.08067)  
Fine-grained diagnostic metrics for retrievers (claim recall, context precision) and generators (context utilization, noise sensitivity, hallucination, faithfulness).

**Evaluation of Retrieval-Augmented Generation: A Survey**  
*Hao Yu, Aoran Gan, Kai Zhang et al.*  
CCF Big Data 2024 (arXiv:2405.07437)  
Introduces Auepora (A Unified Evaluation Process of RAG), comparing 12 evaluation frameworks and analyzing metrics for retrieval and generation components.

**RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems**  
*Robert Friel, Masha Belyi et al.*  
arXiv 2024 (arXiv:2407.11005)  
First large-scale RAG benchmark with **100k examples** across five industry-specific domains. Introduces TRACe evaluation framework with explainable, actionable metrics.

**Survey of Hallucination in Natural Language Generation**  
*Ziwei Ji, Nayeon Lee, Rita Frieske et al.*  
ACM Computing Surveys 2023 (arXiv:2202.03629)  
Comprehensive taxonomy of hallucination types (intrinsic/extrinsic) and evaluation methods across NLG tasks. Essential reference for faithfulness metrics.

**BeIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**  
*Nandan Thakur et al.*  
NeurIPS 2021 Datasets Track  
Heterogeneous retrieval benchmark across 18 datasets and 9 domains for zero-shot retrieval evaluation. Widely used for evaluating retriever components in RAG.

---

## Benchmark datasets for multi-hop QA and knowledge-intensive tasks

Original papers introducing key evaluation datasets.

**HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**  
*Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, Christopher D. Manning*  
EMNLP 2018 (arXiv:1809.09600)  
**113k Wikipedia-based Q&A pairs** requiring reasoning over multiple documents with sentence-level supporting facts for explainability. Primary benchmark for multi-hop RAG.

**MuSiQue: Multihop Questions via Single-hop Question Composition**  
*Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal*  
TACL 2022 (arXiv:2108.00573)  
Addresses shortcut exploitation through bottom-up composition. **25K challenging 2-4 hop questions** with 3x human-machine gap versus prior datasets.

**2WikiMultiHopQA: Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps**  
*Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa*  
COLING 2020 (arXiv:2011.01060)  
Evidence information with reasoning paths combining structured (Wikidata) and unstructured data. Four question types: comparison, inference, compositional, bridge-comparison.

**KILT: A Benchmark for Knowledge Intensive Language Tasks**  
*Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis et al.*  
NAACL 2021 (arXiv:2009.02252)  
Unified benchmark grounding **11 diverse tasks** (QA, fact-checking, slot filling, entity linking, dialogue) on a single Wikipedia snapshot.

**Natural Questions: A Benchmark for Question Answering Research**  
*Tom Kwiatkowski, Jennimaria Palomaki et al.*  
TACL 2019  
First large-scale dataset using real Google search queries with Wikipedia answers. **307k training examples** with long and short answer annotations.

**TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension**  
*Mandar Joshi, Eunsol Choi, Daniel Weld, Luke Zettlemoyer*  
ACL 2017 (arXiv:1705.03551)  
**650K question-answer-evidence triples** authored by trivia enthusiasts with complex compositional questions requiring cross-sentence reasoning.

---

## Framework papers and multi-agent systems

Research on agent frameworks, knowledge graph integration, and multi-agent architectures.

**From Local to Global: A Graph RAG Approach to Query-Focused Summarization (GraphRAG)**  
*Darren Edge, Ha Trinh, Steven Truitt, Jonathan Larson et al. (Microsoft Research)*  
arXiv 2024 (arXiv:2404.16130)  
Uses LLM-generated knowledge graphs with community detection (Leiden algorithm) to answer global sensemaking questions. Substantial improvements in comprehensiveness and diversity over baseline RAG.

**AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**  
*Qingyun Wu, Gagan Bansal, Jieyu Zhang et al. (Microsoft Research & Penn State)*  
arXiv 2023 (arXiv:2308.08155)  
Foundational multi-agent framework with customizable, conversable agents combining LLMs, human inputs, and tools. Supports natural language and code for programming conversation patterns.

**Generative Agents: Interactive Simulacra of Human Behavior**  
*Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein*  
UIST 2023 **Best Paper** (arXiv:2304.03442)  
Memory stream architecture with reflection and planning. Demonstrates emergent social behaviors in 25-agent sandbox environment.

**A Survey on Large Language Model based Autonomous Agents**  
*Lei Wang, Chen Ma, Xueyang Feng et al.*  
Frontiers of Computer Science 2024 (arXiv:2308.11432)  
First comprehensive survey proposing unified framework covering agent construction (profile, memory, planning, action modules), applications, and evaluation strategies.

**MemGPT: Towards LLMs as Operating Systems**  
*Charles Packer, Vivian Fang, Shishir G. Patil et al. (UC Berkeley)*  
arXiv 2023 (arXiv:2310.08560)  
OS-inspired memory management with virtual context and hierarchical memory tiers. Enables extended context beyond LLM limits for document analysis and multi-session chat.

---

## Conclusion

This bibliography reveals several key research trajectories relevant to benchmarking Agentic RAG versus recursive methods:

- **Convergence of approaches**: Self-RAG, IRCoT, and RAPTOR demonstrate that the boundary between "agentic" and "recursive" methods is blurring, with hybrid approaches showing strongest performance
- **Evaluation gaps**: While RAGAS, ARES, and RAGChecker provide automated metrics, standardized benchmarks specifically comparing agentic versus long-context approaches remain limited
- **Cost-performance tradeoffs**: Empirical studies consistently show long-context outperforms RAG on quality metrics when resources allow, but RAG maintains advantages in efficiency and cost
- **Position sensitivity**: The "lost in the middle" phenomenon affects both long-context and RAG systems, motivating recursive and hierarchical retrieval architectures like RAPTOR
- **Multi-hop as key benchmark**: HotpotQA, MuSiQue, and 2WikiMultiHopQA remain essential for evaluating reasoning-intensive RAG architectures

The field is rapidly evolving toward unified architectures that combine adaptive retrieval, iterative refinement, and explicit planning—suggesting future benchmarks must evaluate these hybrid capabilities rather than treating agentic and recursive approaches as distinct paradigms.
