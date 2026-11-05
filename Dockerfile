# First-iteration API (app.main) with TF-IDF models baked in at build time.
# The runtime image keeps requirements-api.txt only: no MLflow, no torch,
# no Hugging Face download when the container starts.
#
#   docker compose up --build
#   docker build -t article-sentiment-api .
#
# Default bake (SENTIMENT_DATA=prepared) downloads the two public sentence
# files, fits tfidf_linearsvc and tfidf_logreg, and copies only the joblib
# files into the runtime stage. Offline toy bake, no dataset download:
#
#   docker build --build-arg SENTIMENT_DATA=dummy -t article-sentiment-api .
#
# The second-iteration CUDA MiniLM path is not part of this image.

FROM python:3.11-slim AS train

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker-train.txt .
RUN pip install --no-cache-dir -r requirements-docker-train.txt

COPY src/__init__.py src/config.py src/data.py src/features.py src/prepare_data.py \
    src/sklearn_models.py src/interface.py src/labels.py src/
COPY data/sample/sentiment_sample.csv data/sample/sentiment_sample.csv
COPY data/sample/opinion_supplement.csv data/sample/opinion_supplement.csv
COPY scripts/bake_sklearn_models.py scripts/bake_sklearn_models.py

# Declared after the pip layer so SENTIMENT_DATA=dummy does not reinstall packages.
ARG SENTIMENT_DATA=prepared
ENV SENTIMENT_DATA=${SENTIMENT_DATA}

RUN python scripts/bake_sklearn_models.py \
    && test -s models/tfidf_linearsvc.joblib \
    && test -s models/tfidf_logreg.joblib \
    && rm -rf /root/.cache data/processed

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SENTIMENT_MODEL=tfidf_linearsvc

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN useradd --create-home --uid 1000 api

COPY --chown=api:api app/ app/
COPY --chown=api:api src/ src/
COPY --from=train --chown=api:api \
    /srv/models/tfidf_linearsvc.joblib \
    /srv/models/tfidf_logreg.joblib \
    /srv/models/

USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json,urllib.request; body=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); raise SystemExit(0 if body.get('model_loaded') is True else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
