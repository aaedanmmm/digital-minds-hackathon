FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /workspace

# transformers must be new enough to know model_type qwen3_5. Verified at
# implementation time: qwen3_5 landed as a first-class model in the 5.15.0
# PyPI release (src/transformers/models/qwen3_5/), and the probe job
# confirmed it loads Qwen/Qwen3.6-27B correctly at that version. Pinned
# exactly (not >=5.0) deliberately: Stage A and Stage B of this study run as
# separate Vertex jobs built from separately-built images days apart, and
# their results must be directly comparable. An unpinned floor could resolve
# to a different transformers release between those builds and silently
# change model behaviour mid-study. If this pin ever needs to move, re-run
# the probe against the new version first. If a version this old ever stops
# recognising qwen3_5, fall back to
# `pip install git+https://github.com/huggingface/transformers.git`.
RUN python -m pip install --no-cache-dir --upgrade \
    "transformers==5.15.0" \
    "accelerate>=1.2" \
    "google-cloud-storage>=2.19,<4" \
    "safetensors" \
    "sentencepiece"

COPY personas ./personas

ENTRYPOINT ["python", "-m"]
CMD ["personas.probe"]
