FROM --platform=linux/amd64 \
    pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime \
    AS segment-algorithm-amd64

#FROM --platform=linux/amd64 \
#    pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
#    AS segment-algorithm-amd64


ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/user/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PATH=/home/user/.local/bin:${PATH}

# Runtime libraries required by OpenCV and related dependencies.
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         libgl1 \
#         libglib2.0-0 \
#         libxcb1 \
#     && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    find /etc/apt -type f \
        \( -name "*.list" -o -name "*.sources" \) \
        -exec sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
            {} +; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::https::No-Cache=true \
        -o Acquire::https::No-Store=true \
        update; \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1; \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r user \
    && useradd \
        -m \
        --no-log-init \
        -r \
        -g user \
        user

USER user
WORKDIR /opt/app

# Install Python dependencies before copying the large model so this layer
# remains cached when only the application code changes.
COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# The complete local InternVL snapshot is stored under resources/.
# Its approximately 8.9 GB size is below the 50 GB per-layer limit.
COPY --chown=user:user resources/ /opt/app/resources/

# Copy the Docker entrypoint and reusable inference utilities.
COPY --chown=user:user \
    inference.py \
    answer_utils.py \
    model_utils.py \
    prompt_utils.py \
    video_utils.py \
    sampling_utils.py \
    /opt/app/

ENTRYPOINT ["python", "inference.py"]
