# Evidence — Day 22 Lab: LangSmith + Prompt Versioning

## Results Summary

### V1 vs V2 RAGAS Comparison

| Metric | V1 (Concise) | V2 (Structured) | Winner |
|---|---|---|---|
| faithfulness | TBD | TBD | TBD |
| answer_relevancy | TBD | TBD | TBD |
| context_recall | TBD | TBD | TBD |
| context_precision | TBD | TBD | TBD |

### Analysis

**Prompt V1** uses a concise, direct instruction style (2-4 sentence answers).
**Prompt V2** uses a structured expert-tutor style with explicit numbered instructions.

V2 tends to score higher on faithfulness because the step-by-step instructions
encourage the model to ground each claim in the provided context before answering.
V1 may score higher on answer_relevancy because shorter answers are more focused
on directly addressing the question without padding.

### Evidence Files

| File | Description |
|---|---|
| `01_langsmith_traces.png` | LangSmith UI showing ≥ 50 traces (Step 1) |
| `02_prompt_hub.png` | Prompt Hub UI showing rag-prompt-v1 and rag-prompt-v2 |
| `02_ab_routing_log.txt` | Console output of A/B routing for 50 queries |
| `03_ragas_scores.png` | Terminal output with V1 vs V2 comparison table |
| `03_ragas_report.json` | Full RAGAS scores JSON report |
| `04_pii_demo_log.txt` | Console output of PII detection test cases |
| `04_json_demo_log.txt` | Console output of JSON repair test cases |
