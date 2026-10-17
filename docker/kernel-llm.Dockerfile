FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ARG GIT_COMMIT=unknown
ARG SOURCE_URL=https://github.com/RangelTech/kernel-llm
# ENVIRONMENT=prod: recusa subir sem INTERNAL_TOKEN e desliga /stub/script por padrao.
ENV GIT_COMMIT=${GIT_COMMIT} PORT=8080 ENVIRONMENT=prod
LABEL org.opencontainers.image.source=${SOURCE_URL} \
      org.opencontainers.image.revision=${GIT_COMMIT} \
      org.opencontainers.image.description="kernel-llm: kernel LangGraph do agent-llm (Rangel Tech)"

RUN useradd -r -u 10001 -m kernel && mkdir -p /srv/artifacts && chown kernel /srv/artifacts
USER kernel
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2).status==200 else 1)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
