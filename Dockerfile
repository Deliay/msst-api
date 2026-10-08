# syntax=docker/dockerfile:1

# CUDA 12.8 runtime with cuDNN, Ubuntu 24.04 (system Python 3.12).
FROM core.harbor.internal.fffdan.com/docker-hub-proxy/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Switch Ubuntu apt sources to the Tsinghua mirror, then install the system
# toolchain + audio codecs. `ffmpeg` is needed for mp3 decode/encode.
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do \
        [ -f "$f" ] || continue; \
        sed -i \
            -e 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' \
            -e 's|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' \
            -e 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' \
            -e 's|http://mirrors.tuna.tsinghua.edu.cn|https://mirrors.tuna.tsinghua.edu.cn|g' \
            "$f"; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        build-essential; \
    rm -rf /var/lib/apt/lists/*

# Install uv from the Tsinghua PyPI mirror.
RUN pip3 install --no-cache-dir --break-system-packages uv

WORKDIR /app

# 1) Resolve and install third-party dependencies (cached across source edits).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Copy the application source and install the project itself.
COPY src ./src
RUN uv sync --frozen --no-dev

# Model storage + runtime configuration.
ENV MSST_MODEL_DIR=/models \
    MSST_TEMP_DIR=/tmp/msst-api \
    MSST_HOST=0.0.0.0 \
    MSST_PORT=8000 \
    MSST_DEVICE=cuda:0 \
    MSST_ALLOW_CPU=false \
    MSST_DOWNLOAD_BACKEND=modelscope

RUN mkdir -p /models /tmp/msst-api
VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["/opt/venv/bin/msst-api"]
