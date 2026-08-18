-- Schema mínimo pros testes do kernel — autocontido, sem depender das
-- migrations do backend (agent-llm mega spec, infra-05: kernel vira repo
-- próprio, `kernel-llm`, não pode mais alcançar `backend/migrations`).
--
-- Contém só as tabelas que o kernel de fato lê/escreve em produção
-- (confirmado por grep em kernel/app/*.py por `FROM|INTO|UPDATE|JOIN`) mais
-- `tenants`, que existe aqui só como alvo de FK — não é a tabela completa do
-- backend, é um stub mínimo pra satisfazer as duas FKs reais que o kernel
-- respeita (`files.tenant_id`, `payment_charges.tenant_id`).
--
-- O checkpointer do LangGraph (`checkpoints`, `checkpoint_writes`) e as
-- tabelas do backend que o kernel NUNCA consulta diretamente (`templates`,
-- `chats`, `chat_messages`, etc. — o backend resolve tudo isso e manda pronto
-- no payload do POST /v1/runs) não entram aqui de propósito.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Stub mínimo — só o suficiente pra ser alvo de FK e pros testes de kernel
-- inserirem uma linha completa (ver kernel/tests/test_rag.py, que insere
-- tenant_key+name) — não é a tabela real do backend (sem is_active/timestamps).
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    chat_id TEXT,
    agent_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    schema_json JSONB,
    preview_json JSONB,
    row_count INTEGER,
    storage_path TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/json',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    chat_id TEXT,
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input JSONB,
    output TEXT,
    status TEXT NOT NULL,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_calls_chat_idx ON tool_calls (chat_id, created_at);

CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    user_id UUID,
    chat_id TEXT,
    agent_name TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_records_tenant_time_idx ON usage_records (tenant_id, created_at);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    source_chat TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memories_scope_idx ON memories (tenant_id, user_id);

CREATE TABLE IF NOT EXISTS memory_extraction_state (
    thread_id TEXT PRIMARY KEY,
    messages_read INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'ready', 'error')),
    error_detail TEXT,
    chunk_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS files_tenant_idx ON files (tenant_id);

CREATE TABLE IF NOT EXISTS file_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    UNIQUE (file_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS file_chunks_tenant_idx ON file_chunks (tenant_id);

CREATE TABLE IF NOT EXISTS payment_charges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    provider TEXT NOT NULL DEFAULT 'mercado_pago',
    external_id TEXT NOT NULL,
    chat_id UUID,
    amount NUMERIC(12, 2) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    reference_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    qr_code TEXT,
    qr_code_base64 TEXT,
    ticket_url TEXT,
    sandbox BOOLEAN NOT NULL DEFAULT TRUE,
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, external_id)
);
CREATE INDEX IF NOT EXISTS payment_charges_tenant_idx ON payment_charges (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS payment_charges_reference_idx ON payment_charges (tenant_id, reference_id);
