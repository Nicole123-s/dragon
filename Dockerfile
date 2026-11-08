FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install Ollama (for zero-shot LLM extraction on weak tasks)
RUN curl -fsSL https://ollama.com/install.sh | sh

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
# Without this line the resolver may pick langchain 1.x and the LLM stage fails at runtime.
RUN python -m pip install --user "langchain==0.3.25" "langchain-ollama==0.3.3" "langchain-chroma==0.2.3"

# Download the encoder model, tokenizer and metrics
COPY --chown=user:user download_model.py .
RUN python download_model.py --model_name joeranbosma/dragon-roberta-large-domain-specific
COPY --chown=user:user download_metrics.py .
RUN python download_metrics.py

# Download the LLM used for weak-task extraction (build-time only; runtime is offline).
# extractinate() runs `ollama pull` at startup, which no-ops when the model is already present.
# &&-chained on purpose: if the pull fails (e.g. registry unreachable), the build FAILS
# instead of silently producing an image without model weights that hangs at runtime.
ENV OLLAMA_MODELS=/opt/app/ollama_models
RUN ollama serve & sleep 15 \
    && ollama pull gemma2:9b \
    && ollama stop gemma2:9b \
    && pkill ollama \
    && sleep 2 \
    && ls -lh /opt/app/ollama_models/blobs/

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
