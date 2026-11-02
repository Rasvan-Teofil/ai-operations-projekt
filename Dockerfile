# Minimal container for the first-iteration API (app.main, default tfidf_linearsvc).
# Build after `python -m src.train`, because the joblib artefacts in models/ are not in git:
#   docker build -t article-sentiment-api .
#   docker run --rm -p 8000:8000 article-sentiment-api
# Without artefacts the container still starts; /health reports model_loaded=false.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTIMENT_MODEL=tfidf_linearsvc

WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ app/
COPY src/ src/
COPY models/ models/

RUN useradd --create-home api && chown -R api /srv
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
