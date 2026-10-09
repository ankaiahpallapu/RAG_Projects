FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vsearch ./vsearch

# Mount the built index read-only at /data/index (e.g. -v $PWD/index:/data/index:ro)
ENV VS_INDEX_DIR=/data/index \
    VS_OMP_THREADS=1

EXPOSE 8000
# One process, many threads: every thread shares the same memory-mapped index.
CMD ["uvicorn", "vsearch.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
