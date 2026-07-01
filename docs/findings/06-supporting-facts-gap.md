# Finding 6: Supporting Facts — All Architectures Score Zero

**Filed:** June 17, 2026  
**Severity:** 🟡 Notable (documentation issue)  
**Tags:** `evaluation` `supporting-facts` `hotpotqa`

---

## The Finding

**No architecture in the benchmark populates `RAGResponse.supporting_facts`.** The evaluation pipeline computes supporting fact EM/F1 and joint EM/F1, but all scores are zero because the input is `None`.

## The Data

For every architecture, every question:
```
supporting_fact_em = 0.0
supporting_fact_f1 = 0.0
joint_em = 0.0
joint_f1 = 0.0
supporting_fact_status = "not_provided"
```

## Why This Happens

- `RAGResponse` has `supporting_facts: list[tuple[str, int]] | None = None`
- `BaseRAG` and all 7 architectures return `RAGResponse(answer=..., ...)` without setting `supporting_facts`
- The evaluator checks `getattr(response, "supporting_facts", None)` — gets `None` — records zero

## Two Options

### Option A: Accept as a finding (Recommended)
The paper can state: *"Multi-hop QA architectures identify answers but cannot pinpoint supporting evidence — even when gold evidence is available in the dataset."* This is a legitimate negative result.

### Option B: Implement in 1-2 architectures
Requires modifying each architecture's `answer()` to extract `(title, sent_idx)` from the reasoning chain. Estimated 2-3 days per architecture. The supporting fact extraction logic needs:
- Match predicted answer span to source documents
- Identify which retrieved document(s) contain the evidence
- Map to HotpotQA's gold `(title, sentence_index)` format

## Recommendation
Go with **Option A** for now. The zero scores are a honest finding. If reviewers flag it, implement supporting fact extraction in Vanilla RAG (simplest) and RLM (best performer) as a rebuttal.
