# DRAGON Submission

Hybrid submission to the [DRAGON challenge](https://dragon.grand-challenge.org/): **encoder fine-tuning as the backbone, zero-shot LLM extraction for weak tasks, with validation-set-weighted fusion**.

## Method description

This submission combines two complementary prediction strategies and fuses them per task:

1. **Encoder fine-tuning (all 28 tasks).** The DRAGON baseline pipeline with the strongest publicly available foundation model (`joeranbosma/dragon-roberta-large-domain-specific`, 0.819 mean test score in the DRAGON paper). The baseline automatically handles all eight task types (single/multi-label classification, regression, NER).

2. **Few-shot LLM extraction (weak tasks only).** For the tasks where the encoder has most headroom — T03, T04, T06, T07, T08, T09, T12, T13, T14, T16, T17, T24 — we additionally query a small large language model (Gemma2 9B via Ollama, temperature 0, structured JSON output) with per-task prompts that ask for a reasoning step followed by an answer from a fixed label vocabulary. For scalar-label tasks (binary/multi-class) the prompt is augmented with 3 training examples retrieved by maximal marginal relevance (few-shot); structured-output tasks (multi-label) run zero-shot. This mirrors and extends the organizers' `llm_extractinator` approach.

3. **Validation-set-weighted fusion.** For each weak task, the blending weight λ (encoder weight) is selected on the validation set by grid search over {0, 0.1, ..., 1.0}, maximizing the task's official metric (AUC, Cohen's kappa — linearly weighted for ordinal tasks —, or RSMAPE with ε=4 mm for T24). Predictions are fused per case: probability/distribution blending where the encoder exposes scores, source switching for multi-label classification. If the LLM output is missing or invalid for a case, the encoder prediction is kept; if the LLM stage fails entirely, the submission falls back to the pure encoder result. A shrinkage rule (a λ below 1 is accepted only if it beats the encoder by a small margin) protects against validation-set noise.

The LLM stage is disabled for all other tasks and for NER tasks (fusion of sequence labels is out of scope for this submission).

## Implementation notes

- All code changes relative to the baseline are in `process.py` (fusion logic) and `llm_fusion.py` (LLM extraction wrapper), plus `llm_tasks/` (per-task prompts, mirrored from the organizers' LLM submission template) and `tiktoken_cache/` (offline tokenizer cache).
- The encoder is switched from the template default to `dragon-roberta-large-domain-specific` (both in `process.py` and the Dockerfile pre-download step).
- The Docker image bundles Ollama and the LLM weights at build time; evaluation runs fully offline (`--network=none`).
- Environment variables for local experimentation: `DISABLE_LLM=1` (encoder-only), `LLM_MODEL` (default `gemma2:9b`), `LLM_EXTRA_TASKS` (e.g. `102,106` to additionally apply LLM fusion to synthetic tasks), `LLM_SUBSET` (limit LLM extraction to the first N cases for quick CPU tests), `ENCODER_MODEL` (override the encoder, e.g. a smaller model for local iteration).

## Local validation notes

The fusion machinery was validated end-to-end on the synthetic tasks (task 102 with `LLM_EXTRA_TASKS=102, LLM_SUBSET=12`): zero-shot extraction, per-row validity filtering, validation-set λ selection, blending, and output verification all behave as designed; the LLM stage degrades gracefully to the encoder when its output is missing or invalid. Note that the Dutch-clinical-pretrained encoder scores lower on the English-templated synthetic tasks than multilingual models do (e.g. ~0.35 vs ~0.68 kappa on task 102 under identical training); this is a property of the pretraining domain, not of the pipeline, and does not apply to the real Dutch benchmark tasks.

If you are using DRAGON resources, please cite the following article:

> J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)
