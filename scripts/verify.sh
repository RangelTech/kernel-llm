#!/usr/bin/env bash
# Gate local (e do CI): ruff + pytest contra um pgvector descartavel (localhost:5433,
# agent/agent/agent_llm; provider LLM = stub) e smoke da imagem de producao (/health).
set -euo pipefail
cd "$(dirname "$0")/.."

PG=kernel-llm-verify-pg
CID=""
cleanup() {
  docker rm -f "$PG" >/dev/null 2>&1 || true
  if [ -n "$CID" ]; then docker stop "$CID" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
docker rm -f "$PG" >/dev/null 2>&1 || true

docker run -d --name "$PG" -e POSTGRES_USER=agent -e POSTGRES_PASSWORD=agent -e POSTGRES_DB=agent_llm \
  -p 5433:5432 pgvector/pgvector:pg16 >/dev/null
for i in $(seq 1 60); do
  docker exec "$PG" pg_isready -U agent -d agent_llm >/dev/null 2>&1 && break
  sleep 1
  [ "$i" -eq 60 ] && { echo "postgres nao subiu"; exit 1; }
done
sleep 3

PY="${PYTHON:-python3}"
"$PY" --version >/dev/null 2>&1 || PY=python
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CI:-}" ]; then
  [ -d .venv ] || "$PY" -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate
fi
python -m pip install -q -r requirements-dev.txt

ruff check .
# Drivers externos (mssql/firebird/mongo/oracle): testes *_ci pulam sozinhos sem *_HOST.
DATABASE_URL=postgresql://agent:agent@localhost:5433/agent_llm \
  python -m pytest -q -m "integration or not integration"

# Smoke da imagem de producao (ENVIRONMENT=prod exige INTERNAL_TOKEN).
IMG="kernel-llm-verify:local"
docker build -q -f docker/kernel-llm.Dockerfile \
  --build-arg GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo local)" -t "$IMG" .
CID="$(docker run -d --rm -p 18080:8080 -e INTERNAL_TOKEN=verify "$IMG")"
for i in $(seq 1 40); do
  curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1 && break
  sleep 1
  [ "$i" -eq 40 ] && { echo "health nao respondeu"; docker logs "$CID" | tail -20; exit 1; }
done
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ok"'
# prod sem INTERNAL_TOKEN tem que recusar subir
if docker run --rm "$IMG" python -c "import app.main" >/dev/null 2>&1; then
  echo "imagem prod subiu sem INTERNAL_TOKEN"; exit 1
fi
echo "verify OK"
