# DRAGON Submission

Hybrid submission to the [DRAGON challenge](https://dragon.grand-challenge.org/): **encoder fine-tuning as the backbone, zero-shot LLM extraction for weak tasks, with validation-set-weighted fusion**.

## Method description

This submission combines two complementary prediction strategies and fuses them per task:

1. **Encoder fine-tuning (all 28 tasks).** The DRAGON baseline pipeline with the strongest publicly available foundation model (`joeranbosma/dragon-roberta-large-domain-specific`, 0.819 mean test score in the DRAGON paper). The baseline automatically handles all eight task types (single/multi-label classification, regression, NER).

2. **Zero-shot LLM extraction (weak tasks only).** For the tasks where all encoder models struggle — T06, T08, T12, T13, T14 (classification) and T24 (multi-label regression) — we additionally query a small large language model (Gemma2 via Ollama, temperature 0, structured JSON output) with per-task prompts that ask for a reasoning step followed by an answer from a fixed label vocabulary. This mirrors the organizers' `llm_extractinator` approach, restricted to weak tasks.

3. **Validation-set-weighted fusion.** For each weak task, the blending weight λ (encoder weight) is selected on the validation set by grid search over {0, 0.1, ..., 1.0}, maximizing the task's official metric (AUC for T06/T08, Cohen's kappa — linearly weighted for ordinal T14 — for T12/T13/T14, RSMAPE with ε=4 mm for T24). Predictions are fused per case: probability blending for binary tasks, distribution blending for multi-class tasks, and element-wise value blending for T24. If the LLM output is missing or invalid for a case, the encoder prediction is kept; if the LLM stage fails entirely, the submission falls back to the pure encoder result.

The LLM stage is disabled for all other tasks and for NER tasks (fusion of sequence labels is out of scope for this submission).

## Implementation notes

- All code changes relative to the baseline are in `process.py` (fusion logic) and `llm_fusion.py` (LLM extraction wrapper), plus `llm_tasks/` (per-task prompts, mirrored from the organizers' LLM submission template) and `tiktoken_cache/` (offline tokenizer cache).
- The encoder is switched from the template default to `dragon-roberta-large-domain-specific` (both in `process.py` and the Dockerfile pre-download step).
- The Docker image bundles Ollama and the LLM weights at build time; evaluation runs fully offline (`--network=none`).
- Environment variables for local experimentation: `DISABLE_LLM=1` (encoder-only), `LLM_MODEL` (default `gemma2:9b`), `LLM_EXTRA_TASKS` (e.g. `102,106` to additionally apply LLM fusion to synthetic tasks).

If you are using DRAGON resources, please cite the following article:

> J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)
