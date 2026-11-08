# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# The pytorch runtime image ships WITHOUT curl/wget (easy to miss and fatal:
# `curl | sh` then no-ops with exit 0). Ubuntu archive is on GC's egress allowlist.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ollama server binary, copied from the official image on Docker Hub.
# GC build egress blocks ollama.com and github.com (504 via their proxy) but
# allows docker.io, PyPI, ubuntu archives and registry.ollama.ai.
# The ollama image ships the binary plus its runner libraries (ggml runners),
# which the binary resolves relative to /usr/local.
RUN --mount=type=bind,from=ollama/ollama:0.40.14,source=/,target=/ollama_img \
    mkdir -p /usr/local/lib/ollama \
    && echo "=== ollama image: relevant paths ===" \
    && (ls -laR /ollama_img/usr/local/lib/ollama 2>/dev/null | head -60 || echo "no /usr/local/lib/ollama") \
    && (ls -laR /ollama_img/usr/lib/ollama 2>/dev/null | head -60 || echo "no /usr/lib/ollama") \
    && (cp -a /ollama_img/usr/local/lib/ollama/. /usr/local/lib/ollama/ 2>/dev/null || true) \
    && (cp -a /ollama_img/usr/lib/ollama/. /usr/local/lib/ollama/ 2>/dev/null || true) \
    && (cp /ollama_img/usr/local/bin/ollama /usr/local/bin/ollama 2>/dev/null \
        || cp /ollama_img/bin/ollama /usr/local/bin/ollama) \
    && chmod 755 /usr/local/bin/ollama \
    && echo "=== installed ===" \
    && find /usr/local/lib/ollama -maxdepth 3 | head -60 \
    && echo "=== llama-server check ===" \
    && find /usr/local/lib/ollama -name "llama-server" -type f | grep . \
    && ollama --version

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


# Install the requirements
COPY --chown=user:user requirements.txt .
RUN python -m pip install --user -r requirements.txt
# Enforce langchain 0.3 after dependency resolution: llm_extractinator 0.4.2 imports
# langchain.globals, which only exists in langchain 0.3.x (1.x removed it and breaks at runtime).
RUN python -m pip install --user "langchain==0.3.25" "langchain-ollama==0.3.3" "langchain-chroma==0.2.3"

# Download the encoder model, tokenizer and metrics
COPY --chown=user:user download_model.py .
RUN python download_model.py --model_name joeranbosma/dragon-roberta-large-domain-specific
COPY --chown=user:user download_metrics.py .
RUN python download_metrics.py

# Bake the LLM with retries and a hard verification (ollama list must show it).
ENV OLLAMA_MODELS=/opt/app/ollama_models
RUN ollama serve & sleep 12; \
    for i in 1 2 3; do ollama pull gemma2:9b && break; echo "[pull retry $i]"; sleep 20; done; \
    ollama list; \
    ollama list | grep -q gemma2 || (echo "MODEL PULL FAILED - image must not be used" && exit 1); \
    ollama stop gemma2:9b || true; pkill ollama || true; sleep 2; \
    ls -lh /opt/app/ollama_models/blobs/

# Set the environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy the algorithm code and LLM assets
COPY --chown=user:user process.py .
COPY --chown=user:user llm_fusion.py .
COPY --chown=user:user llm_tasks /opt/app/llm_tasks
COPY --chown=user:user tiktoken_cache /opt/app/tiktoken_cache

ENTRYPOINT [ "python", "-m", "process" ]
