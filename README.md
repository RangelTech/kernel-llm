# kernel-llm

Kernel LangGraph do agent-llm (Rangel Tech) — runtime de agentes de IA (supervisor + especialistas, tools MCP, memória, RAG, artifacts).

Split de `agent-platform/kernel/` em 17/08/2026 (mega spec `agent-llm`, `infra-05`) — backend e frontend continuam no repo `agent-platform`, deployando na VPS (rangeltech.net). Este repo continua deployando no **Cloud Run** (projeto `eduk-prd-lake`), sozinho.

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

## Deploy

Automático via GitHub Actions (`ci.yml`) em push para `main`, usando Workload Identity Federation (sem chave de service account guardada como secret). Manual: `./infra/deploy.sh kernel` (precisa de `KERNEL_INTERNAL_TOKEN` no ambiente e `gcloud` autenticado no projeto `eduk-prd-lake`).

## Autenticação

O kernel é público no Cloud Run (`--allow-unauthenticated`), protegido por um shared secret (`INTERNAL_TOKEN` no processo, `KERNEL_INTERNAL_TOKEN` do lado de quem chama — o backend do `agent-platform`, rodando na VPS). Ver `app/runs.py::require_internal_auth`.
