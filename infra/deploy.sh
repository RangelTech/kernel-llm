#!/usr/bin/env bash
# Deploy the kernel (LangGraph) to Cloud Run (project rangel-tech).
#
# Split de repo (agent-llm mega spec, infra-05): este repo era `kernel/`
# dentro do monorepo `agent-platform`. Backend+frontend continuam lá,
# deployando na VPS (rangeltech.net) — não aqui.
#
# Correção de higiene (infra-01 seção 2, achado #4 — 21/08/2026): isso
# deployava no projeto GCP errado (eduk-prd-lake, é o projeto GCP da
# MindLab/Eduk, outro cliente do dono, nada a ver com Rangel Tech). Migrado
# pro projeto certo (rangel-tech). DATABASE_URL passa a apontar pro Postgres
# real da VPS via porta 5433 (postgres-direct, TLS — mesmo achado do
# `litellm-router`: a porta 5432/PgBouncer não faz TLS server-side). S3 usa
# as credenciais reais do MinIO da VPS (já eram essas, só mudou de onde o
# Secret Manager guarda). SERPER_API_KEY não foi migrada (chave de terceiro
# fora do GCP, não dá pra ler do projeto antigo sem misturar contexto — o
# kernel já degrada sozinho pra busca via DuckDuckGo sem ela, `app/tools.py`
# linha 528, não é bloqueio; se quiser a Serper de volta, cadastrar uma
# secret nova e reativar aqui).
#
# Kernel público + shared secret (infra-04): sem metadata server fora do GCP,
# a proteção é KERNEL_INTERNAL_TOKEN via env var direta (--set-env-vars, NÃO
# --set-secrets — decisão infra-05: minimizar Secret Manager, token vem do
# GitHub Actions secret KERNEL_INTERNAL_TOKEN neste repo).
#
# Usage: ./infra/deploy.sh kernel  (KERNEL_INTERNAL_TOKEN precisa estar no ambiente)
set -euo pipefail

PROJECT=rangel-tech
REGION=us-central1
REPO=us-central1-docker.pkg.dev/$PROJECT/containers

target=${1:-kernel}
cd "$(dirname "$0")/.."
SHORT_SHA=$(git rev-parse --short HEAD)
GCLOUD_BIN=${GCLOUD_BIN:-gcloud}

SOURCE_DIRS=(app infra .github)
if [ -n "$(git status --porcelain --untracked-files=normal -- "${SOURCE_DIRS[@]}")" ]; then
  echo "source tree is dirty; commit source changes before deploy" >&2
  git status --short --untracked-files=normal -- "${SOURCE_DIRS[@]}" >&2
  exit 1
fi

build() {
  local service=$1
  "$GCLOUD_BIN" builds submit --project=$PROJECT \
    --config=infra/cloudbuild-$service.yaml \
    --substitutions=SHORT_SHA=$SHORT_SHA .
}

deploy_kernel() {
  : "${KERNEL_INTERNAL_TOKEN:?KERNEL_INTERNAL_TOKEN precisa estar setado no ambiente antes de rodar}"
  build kernel
  # INTERNAL_TOKEN pode já existir na revisão anterior como referência de
  # Secret Manager (deploy intermediário durante o split, infra-05) — Cloud
  # Run recusa trocar uma env var de "secret" pra "literal" no mesmo comando
  # sem removê-la explicitamente primeiro.
  "$GCLOUD_BIN" run services update kernel-llm \
    --project=$PROJECT --region=$REGION \
    --remove-secrets=INTERNAL_TOKEN 2>/dev/null || true
  "$GCLOUD_BIN" run deploy kernel-llm \
    --project=$PROJECT --region=$REGION \
    --image=$REPO/teste_ia-kernel:$SHORT_SHA \
    --set-secrets=DATABASE_URL=kernel-database-url:latest,S3_ACCESS_KEY_ID=gcs-hmac-access-key:latest,S3_SECRET_ACCESS_KEY=gcs-hmac-secret-key:latest \
    --set-env-vars="ENABLE_STUB_CONTROL=false,STORAGE_BACKEND=s3,S3_BUCKET=rangel-tech-storage,S3_ENDPOINT_URL=https://storage.googleapis.com,S3_PUBLIC_BASE_URL=https://storage.googleapis.com/rangel-tech-storage/teste-ia,S3_REGION=us-east-1,S3_PREFIX=teste-ia/agent-llm,AWS_REQUEST_CHECKSUM_CALCULATION=when_required,AWS_RESPONSE_CHECKSUM_VALIDATION=when_required,INTERNAL_TOKEN=${KERNEL_INTERNAL_TOKEN},PLATFORM_BACKEND_URL=https://ia.rangeltech.net" \
    --allow-unauthenticated \
    --memory=1Gi --cpu=1 --min-instances=1 --max-instances=3 \
    --timeout=600
  # min-instances=1: achado real 23/08/2026 (mega-spec-reestrutura, item B) --
  # 10,7s de cold start real medido em `GET /health` (1ª de 6 amostras). O
  # kernel processa toda mensagem real de todo tenant, é o serviço mais
  # sensível a essa latência do stack. `infra/terraform/main.tf` deste repo
  # também tem `min_instance_count=1` (mesmo achado, mesma correção) mas
  # NÃO é o que roda de verdade -- o CI (`ci.yml`) chama este script direto,
  # não `terraform apply` (confirmado lendo o log real de um deploy, não só
  # supondo pelo nome do workflow `deploy-cloudrun.yml`, que existe mas não
  # dispara). Mesma classe de achado já repetida 2x hoje em outros repos.
  # NOTA: se INTERNAL_TOKEN ficar vazio por engano, require_internal_auth()
  # no kernel não bloqueia NADA — ver app/runs.py, é fail-open por padrão
  # (modo dev), não fail-closed. O `:?` acima recusa rodar sem o token.
}

case $target in
  kernel) deploy_kernel ;;
  *) echo "usage: $0 kernel" >&2; exit 1 ;;
esac
