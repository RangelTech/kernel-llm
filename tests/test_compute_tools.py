"""execute_python (sandbox) and generate_forecast via the MCP contract seam."""

import json
import uuid

import pytest
from app.storage import register_artifact
from app.tools import open_catalog_session, set_current_agent, set_run_context

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _tool_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


async def _seed_dataset(tenant_id=None, monthly=False) -> str:
    if monthly:
        rows = [
            [f"2025-{month:02d}-01", 100 + month * 10 + (5 if month % 2 else -5)]
            for month in range(1, 13)
        ] + [
            [f"2026-{month:02d}-01", 220 + month * 10 + (5 if month % 2 else -5)]
            for month in range(1, 7)
        ]
        columns = [{"name": "mes", "type": "date"}, {"name": "total", "type": "number"}]
    else:
        columns = [{"name": "produto", "type": "text"}, {"name": "preco", "type": "number"}]
        rows = [["parafuso", 2.5], ["porca", 1.2]]
    descriptor = await register_artifact(
        tenant_id=tenant_id,
        chat_id=f"t-{uuid.uuid4().hex[:8]}",
        agent_name="seed",
        kind="dataset",
        title="seed",
        schema_json=columns,
        preview_json=rows[:10],
        row_count=len(rows),
        payload=json.dumps({"columns": columns, "rows": rows}).encode(),
    )
    return descriptor["artifact_id"]


def _fresh_context(**kwargs):
    set_run_context(secrets={}, datasources=[], tenant_id=None, chat_id=None, **kwargs)
    set_current_agent("compute_agent", {"provider": "stub", "model": "stub-1"})


async def test_sandbox_runs_code_over_dataset_and_publishes(client):
    artifact_id = await _seed_dataset()
    _fresh_context()
    code = (
        "total = sum(r[1] for r in rows)\n"
        "print(f'total={total}')\n"
        "publish_dataset('dobrado', columns, [[r[0], r[1] * 2] for r in rows])\n"
    )
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "execute_python", {"code": code, "artifact_id": artifact_id}
        )
    body = json.loads(_tool_text(result))
    assert "total=3.7" in body["stdout"]
    assert len(body["artifacts"]) == 1
    assert body["artifacts"][0]["row_count"] == 2
    assert body["artifacts"][0]["preview"][0][1] == 5.0  # doubled


async def test_sandbox_timeout_kills_the_process(client, monkeypatch):
    from app import sandbox as sandbox_module

    original = sandbox_module.run_sandboxed

    async def fast_timeout(code, dataset, timeout_seconds=30):
        return await original(code, dataset, timeout_seconds=3)

    monkeypatch.setattr("app.sandbox.run_sandboxed", fast_timeout)
    _fresh_context()
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "execute_python", {"code": "import time\ntime.sleep(60)\n"}
        )
    assert "tempo limite" in _tool_text(result)


async def test_sandbox_blocks_network(client):
    _fresh_context()
    code = (
        "import socket\n"
        "try:\n"
        "    socket.socket()\n"
        "    print('REDE ABERTA')\n"
        "except Exception as exc:\n"
        "    print(f'bloqueado: {exc}')\n"
        "try:\n"
        "    socket.create_connection(('example.com', 80), timeout=3)\n"
        "    print('REDE ABERTA')\n"
        "except Exception as exc:\n"
        "    print(f'bloqueado: {exc}')\n"
    )
    async with open_catalog_session() as session:
        result = await session.call_tool("execute_python", {"code": code})
    body = json.loads(_tool_text(result))
    assert "REDE ABERTA" not in body["stdout"]
    assert "bloqueado" in body["stdout"]


async def test_forecast_generates_projection_and_chart(client):
    artifact_id = await _seed_dataset(monthly=True)
    _fresh_context()
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "generate_forecast",
            {
                "artifact_id": artifact_id,
                "date_column": "mes",
                "value_column": "total",
                "horizon": 4,
                "freq": "MS",
            },
        )
    body = json.loads(_tool_text(result))
    assert body["forecast"]["row_count"] == 4
    assert body["chart"]["kind"] == "chart"
    # Upward-trending series: the forecast should keep climbing past history.
    last_history_value = 220 + 6 * 10 - 5
    assert all(point[1] > last_history_value * 0.8 for point in body["future_preview"])


async def test_forecast_rejects_missing_columns(client):
    artifact_id = await _seed_dataset()
    _fresh_context()
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "generate_forecast",
            {"artifact_id": artifact_id, "date_column": "mes", "value_column": "total"},
        )
    assert "ERRO" in _tool_text(result)
