# IWNET inference server.
#
# The image ships the code, not the weights or the dataset: a trained checkpoint
# (~39 MB) and the generated dataset (~600 MB) are mounted at run time. Building
# them is a separate, GPU-oriented step - see README.md.
#
# The mount point must be <project>/Balanced_From_Sources, because that is where
# iwnet/config.py resolves the dataset, the checkpoints and results_dir.
#
#   docker build -t iwnet .
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/Balanced_From_Sources:/app/Balanced_From_Sources:ro" \
#     iwnet
#
# The service starts and reports status "degraded" if no checkpoint is present,
# rather than refusing to boot. Checkpoint loading never needs network access.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    IWNET_HOST=0.0.0.0 \
    IWNET_PORT=8000

WORKDIR /app

# libGL/libglib are needed by matplotlib; libgomp by scikit-learn/torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small. For a GPU deployment, install torch from
# the CUDA index in the build instead (see README "Deploying").
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY iwnet/ ./iwnet/
COPY frontend/ ./frontend/
COPY grape.py config.yaml ./

# Run as a non-root user: an inference endpoint should not be root.
RUN useradd --create-home --uid 10001 iwnet \
    && chown -R iwnet:iwnet /app
USER iwnet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${IWNET_PORT}/api/health" || exit 1

CMD ["sh", "-c", "python grape.py --serve --host ${IWNET_HOST} --port ${IWNET_PORT}"]
