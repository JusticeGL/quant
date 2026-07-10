FROM python:3.11-slim-bookworm AS base

ARG UV_VERSION=0.10.4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgomp1 \
        libhdf5-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /workspace

FROM base AS locker

CMD ["uv", "lock"]

FROM base AS research

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups --no-install-project

COPY README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups

CMD ["bash"]
