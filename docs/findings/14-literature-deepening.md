# Finding 14: Literature Deepening — Concurrent Work & Missing Citations

**Filed:** June 26, 2026
**Status:** Complete
**Tags:** `literature` `concurrent-work` `positioning` `citations`

---

## Summary

Research into concurrent and related work reveals:
1. **"Is Agentic RAG Worth It?" (Ferrazzi et al., ACL 2026 Industry Track)** — direct concurrent work with the same thesis question. Must be discussed thoroughly.
2. **BCAS (McCleary & Ghawaly, LREC 2026)** — budget-constrained agentic search on HotpotQA. Concurrent work.
3. **7 missing citations** that reviewers will likely flag.
4. **PSC metric is confirmed novel** — no prior RAG prompt-sensitivity metric exists.
5. **Equal-compute normalization is confirmed novel** — no prior work does token-budget Pareto.

---

## 1. Ferrazzi et al. (ACL 2026) — "Is Agentic RAG Worth It?"

**arXiv:** 2601.07711 (v1: Jan 12 2026, v2: Apr 20 2026)
**Venue:** ACL 2026 Industry Track (accepted)

### What they do
Compare "Enhanced RAG" (fixed modular pipeline: router + HyDE + retriever + ELECTRA reranker) vs "Agentic RAG" (single-tool LLM orchestrator). Evaluate on FIQA, NQ, FEVER, CQADupStack (single-hop/domain QA). Uses Qwen3 models (0.6B-32B).

### Key findings
- Neither paradigm is universally superior
- Agentic wins at intent handling and query rewriting (+2.8 NDCG@10)
- Enhanced wins at document refinement (reranking)
- **Agentic is 3.3x more input tokens, 1.9x more output, 1.5x more time, up to 3.6x cost**
- Changing LLM produces identical patterns

### Overlap with COMPASS
- **Conceptual:** Both ask "is agentic complexity worth it?" and conclude no universal gain
- **Substantive: LOW.** Different datasets (multi-hop vs single-hop), different architectures (7 vs 2), different metrics (EM/F1 vs NDCG), COMPASS adds theory/PSC/error-taxonomy/equal-compute/significance-testing

### How to position
Cite as concurrent work. Emphasize:
- COMPASS studies *multi-hop QA* (where iteration becomes necessary on MuSiQue)
- COMPASS spans *three paradigms including RLM* (absent in Ferrazzi)
- COMPASS introduces *equal-compute normalization* (Ferrazzi only reports raw token counts)
- Ferrazzi's finding (Agentic 3.3x input tokens) is *consistent with COMPASS's quadratic cost theorem* and 11x token finding — this strengthens external validity

---

## 2. BCAS (McCleary & Ghawaly, LREC 2026) — Budget-Constrained Agentic Search

**arXiv:** 2603.08877 (Mar 2026)

### What they do
Budget-Constrained Agentic Search harness on HotpotQA (among others). Compares 6 LLMs. Finds accuracy improves with searches up to a small cap; hybrid retrieval + reranking gives largest gains.

### Overlap with COMPASS
- **Moderate.** Both study budget-constrained agentic RAG cost-accuracy on HotpotQA.
- COMPASS's equal-compute normalization (token-budget Pareto) is more principled than BCAS's tool-call budget gating.

### How to position
Cite as concurrent work on budgeted agentic search. Note COMPASS uses token-budget normalization (model-independent) vs BCAS's tool-call budget.

---

## 3. Missing Citations (reviewers will flag these)

### High priority (must add)
| Paper | arXiv | Venue | Why |
|-------|-------|-------|-----|
| Singh et al., Agentic RAG survey | 2501.09136 | 2025 | THE agentic RAG survey — reviewers expect it |
| Yang et al., CRAG | 2406.04744 | NeurIPS 2024 | Major RAG benchmark |
| Es et al., RAGAS | 2309.15217 | EACL 2024 | Foundational RAG eval framework |
| Saad-Falcon et al., ARES | 2311.09476 | NAACL 2024 | Automated RAG evaluation |
| Ren et al., RAGChecker | 2408.08067 | NeurIPS 2024 | Diagnostic RAG metrics (closest to our error taxonomy) |
| Friel et al., RAGBench | 2407.11005 | 2024 | Large-scale RAG benchmark |

### Medium priority (should add)
| Paper | arXiv | Why |
|-------|-------|-----|
| Wang et al., Best Practices in RAG | 2407.01219 | Widely cited RAG practices study |
| McCleary & Ghawaly, BCAS | 2603.08877 | Concurrent budgeted agentic search |
| Li et al., RAG vs Long-Context | 2407.16833 | Cost-performance precedent for equal-compute |

### PSC novelty validation
- **GeoRepEval** (Jawandhia et al., 2604.16421) — closest analog: McNemar + bootstrap + Invariance@3 metric for geometry representations. Cite as methodological parallel.
- **Alkaeed et al.** (2606.07237) — prompt sensitivity in healthcare LLMs. Cite as related prompt-sensitivity work.
- **No prior RAG prompt-sensitivity metric exists.** PSC is novel.

### Equal-compute novelty validation
- **No dedicated paper found** on equal-compute normalization for RAG architecture comparison.
- Closest precedents: Li et al. (2407.16833, RAG vs long-context cost) and BCAS (tool-call budgets).
- **COMPASS's token-budget Pareto normalization appears novel.**

---

## 4. TraceRAG — NOT relevant

arXiv:2509.08865 is about Android malware detection, not RAG benchmarking. Do not cite.

---

## Actions

- [x] Research concurrent work
- [x] Identify missing citations
- [ ] Add citations to paper bibliography
- [ ] Update "Comparison with Concurrent Work" section
- [ ] Add RAGAS/ARES/RAGChecker to Related Work
