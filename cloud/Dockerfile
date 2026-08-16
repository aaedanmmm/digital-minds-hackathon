FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /workspace

# transformers must be new enough to know model_type qwen3_5. Verified at
# implementation time: qwen3_5 landed as a first-class model in the 5.15.0
# PyPI release (src/transformers/models/qwen3_5/), so the >=5.0 floor with
# --upgrade resolves to a release that already knows this architecture. If
# a future resolve ever regresses, fall back to
# `pip install git+https://github.com/huggingface/transformers.git`.
RUN python -m pip install --no-cache-dir --upgrade \
    "transformers>=5.0" \
    "accelerate>=1.2" \
    "google-cloud-storage>=2.19,<4" \
    "safetensors" \
    "sentencepiece"

COPY personas ./personas

ENTRYPOINT ["python", "-m"]
CMD ["personas.probe"]
