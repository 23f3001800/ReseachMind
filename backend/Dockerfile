FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build from the lockfile so an image is reproducible and a dependency can't
# change contract underneath a deploy. requirements.txt stays the human-edited
# source of intent; requirements.lock is what actually ships.
COPY requirements.lock requirements-rag.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock -r requirements-rag.txt


FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

# Bake the embedding model into the image. Downloading it on first use would put
# a multi-second network fetch on the first user request after every cold start,
# and would fail outright on an egress-restricted network.
ENV FASTEMBED_CACHE=/opt/fastembed
RUN mkdir -p /opt/fastembed \
 && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')"

# Storage root. Mount a volume here in production; anything written to the
# container's own filesystem is gone when the container is replaced.
ENV DATA_DIR=/app/data

# Run as non-root. The data dir is created and owned up front so a mounted
# volume doesn't land read-only for appuser.
RUN useradd -m -u 1000 appuser \
 && mkdir -p /app/data \
 && chown -R appuser /app /opt/fastembed
USER appuser

# The platform injects PORT (Azure Container Apps, Cloud Run, HF Spaces all do).
ENV PORT=8000
EXPOSE 8000

# Secrets come from the platform's env/secret store — never baked into the image.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
