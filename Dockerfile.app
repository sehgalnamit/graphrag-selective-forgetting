FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt .
# Install everything except the optional heavy ML extras up front for a
# fast, reliable build; sentence-transformers/spacy are attempted too but
# the application code gracefully falls back if either is unavailable at
# runtime (see app/embeddings.py, app/entities.py).
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] pydantic pydantic-settings \
        neo4j qdrant-client apscheduler httpx python-dotenv numpy \
        pytest pytest-asyncio ruff

COPY app ./app
COPY decay_engine ./decay_engine
COPY evaluator ./evaluator

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
