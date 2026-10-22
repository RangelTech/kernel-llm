# kernel-llm

> **DESCONTINUADO.** O kernel LangGraph foi substituído pelo kernel-harness.
>
> Substituto: o `kernel-harness` (futuro `luar_harness`), que executa os agentes por skills. Os agentes `graph` estão em extinção (Fase L); o container ainda roda e os dados ficam preservados. Arquivamento só quando não restar agente `graph` em produção.
>
> Este repositório fica público e somente para consulta. Não receberá novas funcionalidades e será arquivado no GitHub quando a condição acima for cumprida.

Kernel LangGraph do agent-llm (Rangel Tech) — runtime de agentes de IA (supervisor + especialistas, tools MCP, memória, RAG, artifacts).

Split de `agent-platform/kernel/` (17/08/2026). Entrega na VPS por release: cada tag `v*` publica `image.tar.gz` (docker-archive) e `release-manifest.json`; quem implanta e so o `vps_rt_infra`.

## Rodando localmente

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements-dev.txt
DATABASE_URL=postgresql://agent:agent@localhost:5433/agent_llm uvicorn app.main:app --reload --port 8080
```

## Testes

```bash
pytest -q
```

Schema de teste é autocontido (`tests/schema.sql`) — só as tabelas que o kernel realmente usa (`artifacts`, `tool_calls`, `usage_records`, `memories`, `memory_extraction_state`, `files`, `file_chunks`, `payment_charges`), não depende de nenhum outro repo.

## Release e deploy

Tag `v*` (ou `workflow_dispatch`) dispara `.github/workflows/release.yml`: roda `scripts/verify.sh` (ruff + pytest com pgvector descartavel + smoke da imagem), builda `docker/kernel-llm.Dockerfile` e publica o GitHub Release com `image.tar.gz` e `release-manifest.json`. Este repo nao faz deploy nem guarda credencial da VPS. Gate local: `bash scripts/verify.sh` (precisa de Docker).

## Producao e autenticacao

`ENVIRONMENT=prod` (ou `RT_ENV=prod`; ja e o default da imagem): o processo recusa subir com `INTERNAL_TOKEN` vazio e `ENABLE_STUB_CONTROL` passa a `false` por padrao. Fora de prod (dev/testes) a autenticacao e opcional e o `/stub/script` fica ligado. Backend chama o kernel com `Authorization: Bearer <INTERNAL_TOKEN>` (`app/runs.py::require_internal_auth`).

Env obrigatoria em runtime: `DATABASE_URL`, `INTERNAL_TOKEN`, `PLATFORM_BACKEND_URL`, `STORAGE_BACKEND=s3`, `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` (opcionais: `S3_REGION`, `S3_PREFIX`, `S3_PUBLIC_BASE_URL`, `SERPER_API_KEY`). Com `STORAGE_BACKEND=s3` nada de GCP e carregado no boot.
