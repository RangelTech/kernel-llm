"""RAG pipeline: ingestion (chunk+embed) and retrieval through the agent
tool, all with the deterministic stub embedding."""

import uuid

import psycopg
import pytest

from app.config import settings
from app.ingestion import ingest_file, search_chunks
from app.storage import save_payload
from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STUB = {"provider": "stub", "model": "stub-1"}
STUB_EMBED = {"provider": "stub"}

POLICY = (
    "Política de reembolso da ACME. Despesas de viagem devem ser lançadas em "
    "até 30 dias. O limite diário de alimentação é de R$ 120. Reembolsos de "
    "transporte exigem nota fiscal. "
) * 3 + (
    "Política de férias da ACME. Férias devem ser agendadas com 60 dias de "
    "antecedência e aprovadas pelo gestor direto. "
) * 3


def _seed_file(tenant_id: str) -> str:
    """Insert a files row + payload in local storage, return file_id."""
    file_id = str(uuid.uuid4())
    path = save_payload(
        f"tenants/{tenant_id}/files/{file_id}/politicas.txt",
        POLICY.encode(),
        "text/plain",
    )
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            """INSERT INTO tenants (id, tenant_key, name)
               VALUES (%s, %s, 'RAG Test') ON CONFLICT DO NOTHING""",
            (tenant_id, f"rag-{tenant_id[:8]}"),
        )
        conn.execute(
            """INSERT INTO files (id, tenant_id, name, content_type, size_bytes, storage_path)
               VALUES (%s, %s, 'politicas.txt', 'text/plain', %s, %s)""",
            (file_id, tenant_id, len(POLICY), path),
        )
        conn.commit()
    return file_id


async def test_ingestion_chunks_and_marks_ready():
    tenant_id = str(uuid.uuid4())
    file_id = _seed_file(tenant_id)
    await ingest_file(file_id, STUB_EMBED)

    with psycopg.connect(settings.database_url) as conn:
        status, chunk_count = conn.execute(
            "SELECT status, chunk_count FROM files WHERE id = %s", (file_id,)
        ).fetchone()
        chunks = conn.execute(
            "SELECT count(*), count(embedding) FROM file_chunks WHERE file_id = %s",
            (file_id,),
        ).fetchone()
    assert status == "ready"
    assert chunk_count == chunks[0] > 0
    assert chunks[1] == chunks[0]  # every chunk embedded


async def test_search_finds_the_relevant_chunk():
    tenant_id = str(uuid.uuid4())
    file_id = _seed_file(tenant_id)
    await ingest_file(file_id, STUB_EMBED)

    results = await search_chunks(
        STUB_EMBED, "qual o limite diário de alimentação?", [file_id], tenant_id
    )
    assert results
    assert any("R$ 120" in r["content"] for r in results[:2])


async def test_search_is_tenant_scoped():
    tenant_a, tenant_b = str(uuid.uuid4()), str(uuid.uuid4())
    file_a = _seed_file(tenant_a)
    await ingest_file(file_a, STUB_EMBED)

    # Tenant B searching tenant A's file id gets nothing.
    results = await search_chunks(STUB_EMBED, "reembolso", [file_a], tenant_b)
    assert results == []


async def test_agent_answers_from_documents_via_rag_tool(client):
    tenant_id = str(uuid.uuid4())
    file_id = _seed_file(tenant_id)
    await ingest_file(file_id, STUB_EMBED)

    await client.post(
        "/stub/script",
        json={
            "rules": [
                [
                    "limite de alimentação",
                    'TOOL:policies_agent:{"task": "consulte o limite de alimentacao"}',
                ],
                [
                    "consulte o limite de alimentacao",
                    'TOOL:query_agent_rag:{"question": "limite diário de alimentação"}',
                ],
                ["R$ 120", "o limite diário de alimentação é R$ 120"],
            ],
            "default": "não achei",
        },
    )
    payload = {
        "thread_id": f"t-{uuid.uuid4().hex[:8]}",
        "message": "qual o limite de alimentação?",
        "supervisor": {"prompt": "Coordene.", "model": STUB},
        "agents": [
            {
                "name": "policies_agent",
                "description": "políticas da empresa",
                "prompt": "Você responde políticas.",
                "model": STUB,
                "tools": ["query_agent_rag"],
                "file_ids": [file_id],
            }
        ],
        "max_steps": 4,
        "tenant_id": tenant_id,
        "embedding": STUB_EMBED,
    }
    r = await client.post("/v1/runs", json=payload)
    events = _events(r.text)
    tool_events = [d for e, d in events if e == "tool"]
    assert any(t["tool"] == "query_agent_rag" and t["status"] == "ok" for t in tool_events)
    done = next(d for e, d in events if e == "done")
    assert "R$ 120" in done["text"]
