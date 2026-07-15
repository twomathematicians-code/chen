# CHEN — Multi-stage Dockerfile
#
# Builds a production-ready image with the CHEN CLI + HTTP API server.
# Default backend is MockBackend (no model downloads); for real inference
# mount your model cache or set CHEN_HF_* env vars.
#
# Build:
#   docker build -t chen:latest .
#
# Run the API server:
#   docker run -p 8000:8000 chen:latest serve --host 0.0.0.0 --port 8000
#
# Run the CLI:
#   docker run --rm chen:latest run --prompt "Explain recursion."
#
# Run benchmarks:
#   docker run --rm chen:latest bench --phase 1

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

# Install build deps for numpy/torch (if user adds hf extra)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency manifest first for better layer caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install with server + dev extras (no hf — too large for default image)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[server,dev]"

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# Runtime deps: git for version detection, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --uid 1000 chen
USER chen
WORKDIR /home/chen

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source (already installed in editable mode, but include for ref)
COPY --chown=chen:chen --from=builder /build /home/chen/app
WORKDIR /home/chen/app

# Persist runs to a volume
RUN mkdir -p /home/chen/chen_data
ENV CHEN_RUN_STORE_PATH=/home/chen/chen_data/runs.sqlite3
ENV CHEN_LOG_LEVEL=INFO
ENV CHEN_LOG_JSON=1
VOLUME ["/home/chen/chen_data"]

# Healthcheck: hit the /v1/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

EXPOSE 8000

# Default entrypoint: chen CLI
ENTRYPOINT ["chen"]
CMD ["--help"]
