"""Output tools consuming dataset artifacts by reference (stub provider)."""

import json
import uuid

import psycopg
import pytest
from app.config import settings
from app.datasources import make_temp_sqlite
from app.storage import load_payload
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
                "name": "analista_agent",
                "description": "análises e gráficos",
                "prompt": "Você analisa dados.",
                "model": STUB,
                "tools": ["run_sql_query", "generate_chart", "export_xlsx", "generate_pdf"],
            }
        ],
        "max_steps": 6,
        "datasources": [{"name": "erp", "kind": "sqlite", "config": {"path": db_path}}],
    }


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def _run_and_get_artifacts(client, message, db, rules) -> list[dict]:
    await _script(client, rules, default="terminei a análise")
    r = await client.post("/v1/runs", json=_payload(message, db))
    events = _events(r.text)
    assert any(e == "done" for e, _ in events), r.text[-500:]
    return [d for e, d in events if e == "artifact"]


async def test_chart_chains_from_dataset_without_requerying(client):
    db = make_temp_sqlite(SEED)
    # The stub cannot see artifact ids, so the specialist first materializes
    # the dataset, then the SECOND round's tool result (with artifact_id) is
    # in its transcript; we script it to chain via the returned id marker.
    # Instead we exercise chaining directly at the tool layer through two
    # scripted rounds: query -> chart(previous id). The chart tool re-reads
    # the artifact id from the previous tool output using the stub's echo.
    await _script(
        client,
        [
            ["gráfico de vendas", 'TOOL:analista_agent:{"task": "faça o gráfico"}'],
            [
                "faça o gráfico",
                'TOOL:run_sql_query:{"datasource": "erp",'
                ' "query": "SELECT mes, total FROM vendas ORDER BY mes"}',
            ],
        ],
        default="terminei",
    )
    # First run materializes the dataset.
    r = await client.post("/v1/runs", json=_payload("gráfico de vendas", db))
    events = _events(r.text)
    dataset_id = next(d for e, d in events if e == "artifact")["artifact_id"]

    # Second run: chart from that dataset id.
    await _script(
        client,
        [
            ["agora o gráfico", 'TOOL:analista_agent:{"task": "plote o dataset"}'],
            [
                "plote o dataset",
                'TOOL:generate_chart:{"artifact_id": "' + dataset_id
                + '", "chart_type": "bar", "x_column": "mes", "y_columns": "total",'
                ' "title": "Vendas"}',
            ],
        ],
        default="gráfico pronto",
    )
    r2 = await client.post("/v1/runs", json=_payload("agora o gráfico", db))
    events2 = _events(r2.text)
    charts = [d for e, d in events2 if e == "artifact" and d["kind"] == "chart"]
    assert len(charts) == 1

    figure = json.loads(load_payload(_storage_path(charts[0]["artifact_id"])))
    assert figure["data"][0]["type"] == "bar"
    assert figure["data"][0]["x"] == ["fev", "jan", "mar"]
    assert figure["layout"]["title"]["text"] == "Vendas"


def _storage_path(artifact_id: str) -> str:
    with psycopg.connect(settings.database_url) as conn:
        return conn.execute(
            "SELECT storage_path FROM artifacts WHERE id = %s", (artifact_id,)
        ).fetchone()[0]


async def test_xlsx_and_pdf_generation(client):
    db = make_temp_sqlite(SEED)
    await _script(
        client,
        [
            [
                "materialize os dados",
                'TOOL:run_sql_query:{"datasource": "erp", "query": "SELECT * FROM vendas"}',
            ],
            ["quero a planilha", 'TOOL:analista_agent:{"task": "materialize os dados"}'],
        ],
        default="ok",
    )
    r = await client.post("/v1/runs", json=_payload("quero a planilha", db))
    dataset_id = next(d for e, d in _events(r.text) if e == "artifact")["artifact_id"]

    await _script(
        client,
        [
            ["exporte agora", 'TOOL:analista_agent:{"task": "gere os arquivos"}'],
            [
                "gere os arquivos",
                'TOOL:export_xlsx:{"artifact_id": "' + dataset_id + '", "title": "Vendas"}',
            ],
            [
                "vendas.xlsx",
                'TOOL:generate_pdf:{"title": "Relatório", "content_markdown":'
                ' "# Resumo\\n- tudo certo", "artifact_id": "' + dataset_id + '"}',
            ],
        ],
        default="arquivos prontos",
    )
    r2 = await client.post("/v1/runs", json=_payload("exporte agora", db))
    artifacts = [d for e, d in _events(r2.text) if e == "artifact"]
    kinds = {a["kind"] for a in artifacts}
    assert "file" in kinds
    files = [a for a in artifacts if a["kind"] == "file"]
    assert len(files) == 2

    xlsx = load_payload(_storage_path(files[0]["artifact_id"]))
    assert xlsx[:2] == b"PK"  # xlsx = zip container
    pdf = load_payload(_storage_path(files[1]["artifact_id"]))
    assert pdf[:4] == b"%PDF"


async def test_chart_refuses_foreign_tenant_artifact(client):
    db = make_temp_sqlite(SEED)
    await _script(
        client,
        [
            [
                "rode a consulta base",
                'TOOL:run_sql_query:{"datasource": "erp", "query": "SELECT * FROM vendas"}',
            ],
            ["materialize vendas", 'TOOL:analista_agent:{"task": "rode a consulta base"}'],
        ],
        default="ok",
    )
    tenant_a = str(uuid.uuid4())
    payload = _payload("materialize vendas", db)
    payload["tenant_id"] = tenant_a
    r = await client.post("/v1/runs", json=payload)
    dataset_id = next(d for e, d in _events(r.text) if e == "artifact")["artifact_id"]

    # Another tenant tries to chart it.
    await _script(
        client,
        [
            [
                "grafico do dataset alheio",
                'TOOL:generate_chart:{"artifact_id": "' + dataset_id
                + '", "chart_type": "bar", "x_column": "mes", "y_columns": "total"}',
            ],
            ["tente plotar", 'TOOL:analista_agent:{"task": "grafico do dataset alheio"}'],
            ["outra empresa", "não posso acessar"],
        ],
        default="bloqueado",
    )
    payload_b = _payload("tente plotar", db)
    payload_b["tenant_id"] = str(uuid.uuid4())
    r2 = await client.post("/v1/runs", json=payload_b)
    events = _events(r2.text)
    tool_events = [d for e, d in events if e == "tool"]
    chart_calls = [t for t in tool_events if t["tool"] == "generate_chart"]
    assert chart_calls and chart_calls[0]["status"] == "ok"  # tool ran, returned ERRO text
    assert not [d for e, d in events if e == "artifact" and d["kind"] == "chart"]
