# Pinned to 3.11 for a reproducible build. The stack also runs on 3.12/3.13;
# 3.14 still hits source builds in the transitive dependency tree.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep model downloads on a mountable volume instead of the image layer.
    HF_HOME=/models

# ffmpeg decodes whatever audio format the episodes happen to be in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default PyPI wheel drags in ~2.5GB of CUDA libraries
# that are useless in a CPU container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-ingest.txt ./
RUN pip install --no-cache-dir -r requirements-ingest.txt

COPY app ./app
COPY ingest.py rag.py main.py eval.py eval_set.json ./

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/data /app/episodes \
    && chown -R app:app /models /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
