"""SQL tool + artifact pipeline over the HTTP seam (sqlite datasource,
stub provider, local artifact storage)."""

import json
import uuid

import psycopg
import pytest

from app.config import settings
from app.datasources import make_temp_sqlite
from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STUB = {"provider": "stub", "model": "stub-1"}

SEED = """
CREATE TABLE vendas (mes TEXT, total REAL);
INSERT INTO vendas VALUES ('jan', 1000), ('fev', 1500), ('mar', 900);
"""


def _payload(message: str, db_path: str) -> dict:
    return {
        "thread_id": f"t-{uuid.uuid4().hex[:8]}",
        "message": message,
        "supervisor": {"prompt": "Coordene.", "model": STUB},
        "agents": [
            {
                "name": "dados_agent",
                "description": "consultas a dados",
                "prompt": "Você consulta dados.",
                "model": STUB,
                "tools": ["run_sql_query", "describe_datasources"],
            }
        ],
        "max_steps": 4,
        "tenant_id": None,
        "datasources": [
            {"name": "erp", "kind": "sqlite", "config": {"path": db_path}}
        ],
    }


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def test_sql_query_materializes_a_dataset_artifact(client):
    db = make_temp_sqlite(SEED)
    await _script(
        client,
        [
            ["total de vendas", 'TOOL:dados_agent:{"task": "some as vendas"}'],
            [
                "some as vendas",
                'TOOL:run_sql_query:{"datasource": "erp",'
                ' "query": "SELECT mes, total FROM vendas ORDER BY mes"}',
            ],
            ["artifact_id", "consultei: 3 meses de vendas"],
        ],
        default="feito",
    )
    payload = _payload("qual o total de vendas?", db)
    r = await client.post("/v1/runs", json=payload)
    events = _events(r.text)

    artifact_events = [d for e, d in events if e == "artifact"]
    assert len(artifact_events) == 1
    assert artifact_events[0]["kind"] == "dataset"
    artifact_id = artifact_events[0]["artifact_id"]

    # Metadata row + retrievable payload with ALL rows (not just the preview).
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT kind, row_count, storage_path, preview_json FROM artifacts WHERE id = %s",
            (artifact_id,),
        ).fetchone()
    assert row[0] == "dataset" and row[1] == 3

    from app.storage import load_payload

    stored = json.loads(load_payload(row[2]))
    assert len(stored["rows"]) == 3
    assert [c["name"] for c in stored["columns"]] == ["mes", "total"]

    # The model saw only id+schema+preview, never the payload dump format.
    tool_events = [d for e, d in events if e == "tool"]
    assert any(t["tool"] == "run_sql_query" and t["status"] == "ok" for t in tool_events)


async def test_write_statements_are_refused(client):
    db = make_temp_sqlite(SEED)
    await _script(
        client,
        [
            ["apague", 'TOOL:dados_agent:{"task": "apague tudo"}'],
            [
                "apague tudo",
                'TOOL:run_sql_query:{"datasource": "erp", "query": "DELETE FROM vendas"}',
            ],
            ["erro", "não posso apagar"],
        ],
        default="terminei",
    )
    r = await client.post("/v1/runs", json=_payload("apague as vendas", db))
    events = _events(r.text)
    assert any(e == "done" for e, _ in events)

    # The table is intact.
    import sqlite3

    conn = sqlite3.connect(db)
    count = conn.execute("SELECT count(*) FROM vendas").fetchone()[0]
    conn.close()
    assert count == 3


async def test_describe_datasources_lists_tables(client):
    db = make_temp_sqlite(SEED)
    await _script(
        client,
        [
            ["o que temos", 'TOOL:dados_agent:{"task": "descreva os dados"}'],
            ["descreva os dados", "TOOL:describe_datasources:{}"],
            ["vendas", "temos a tabela vendas com mes e total"],
        ],
        default="nada",
    )
    r = await client.post("/v1/runs", json=_payload("o que temos de dados?", db))
    events = _events(r.text)
    done = next(d for e, d in events if e == "done")
    assert "vendas" in done["text"]


async def test_unknown_datasource_is_a_tool_error_not_a_crash(client):
    db = make_temp_sqlite(SEED)
    await _script(
        client,
        [
            ["consulte", 'TOOL:dados_agent:{"task": "consulte o outro banco"}'],
            [
                "consulte o outro banco",
                'TOOL:run_sql_query:{"datasource": "nao_existe", "query": "SELECT 1"}',
            ],
            ["não existe", "essa fonte não existe"],
        ],
        default="segui",
    )
    r = await client.post("/v1/runs", json=_payload("consulte o outro banco", db))
    events = _events(r.text)
    assert any(e == "done" for e, _ in events)
    assert not any(e == "error" for e, _ in events)
