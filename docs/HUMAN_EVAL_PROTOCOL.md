# Human Evaluation Protocol for COMPASS

## Purpose

To validate automated metrics (EM, F1) against human judgment and establish
a human upper bound on multi-hop QA performance.

## Annotation Schema

For each question, the annotator evaluates:

### 1. Gold Answer Quality
- **Is the gold answer correct?** (Yes/No/Uncertain)
- **Is the gold answer complete?** (Yes/No - e.g., missing middle name)
- **Gold answer category:** (Person, Place, Date, Number, Yes/No, Other)

### 2. Prediction Quality (per architecture)
For each of the 7 architectures' predictions on this question:

| Rating | Code | Description |
|--------|------|-------------|
| Correct | 3 | Answer is correct and complete |
| Partially Correct | 2 | Answer is mostly correct but incomplete or has minor errors |
| Near Miss | 1 | Answer is in the right semantic neighborhood but factually wrong |
| Wrong | 0 | Answer is completely incorrect |

### 3. Error Type (if rating < 3)
- **RETRIEVAL_FAIL**: Missing key evidence (prediction shows lack of relevant info)
- **REASONING_FAIL**: Had the right info but reached wrong conclusion
- **ENTITY_CONFUSION**: Mixed up similar entities (e.g., two people with same name)
- **PREMATURE_TERMINATION**: Gave up ("I don't know", "unknown")
- **HALLUCINATION**: Stated facts not supported by evidence
- **PARTIAL_ANSWER**: Got part of a multi-part answer correct
- **FORMAT_ERROR**: Correct answer but wrong format (e.g., "Yes" vs "yes")
- **OTHER**: Specify in notes

## Question Selection

Select 150 questions using stratified sampling:

| Stratum | Count | Source |
|---------|-------|--------|
| All-arch-fail (hardest) | 30 | Questions where all 7 architectures fail |
| All-arch-correct (easiest) | 20 | Questions where all 7 succeed |
| Mixed (some succeed, some fail) | 80 | Random sample from remaining |
| Near-miss cases | 20 | F1 > 0.5 but EM = 0 |

This ensures coverage of:
- Easy questions (ceiling analysis)
- Hard questions (failure analysis)
- Borderline cases (metric validation)

## Procedure

1. **Run `scripts/failure_case_studies.py`** to generate the candidate pool
2. **Select 150 questions** using the stratified sampling above
3. **Create annotation spreadsheet** with columns:
   - question_id, question_text, gold_answer
   - For each of 7 architectures: predicted_answer, EM, F1
   - Empty columns: gold_correct, gold_complete, pred_rating_{arch}, error_type_{arch}, notes
4. **Annotate** each question (estimated 2-3 minutes per question = ~5-7 hours)
5. **Compute agreement** between human ratings and EM/F1:
   - Spearman correlation between human rating and F1
   - Fraction of cases where EM disagrees with human judgment

## Expected Outcomes

- **Human upper bound**: Fraction of questions where gold answer is deemed correct
- **Metric validation**: Correlation between F1 and human judgment
- **Error type distribution**: How human-assigned error types compare to the
  automated 7-category taxonomy
- **Format error rate**: Fraction of EM=0 cases that are actually correct but
  formatted differently

## Paper Integration

Results feed into:
- §5.7 (Error Analysis): "Human evaluation on 150 questions confirms..."
- §6 (Limitations): "Automated metrics agree with human judgment in X% of cases"
- Potential new finding: "Y% of EM failures are format errors, not reasoning errors"
