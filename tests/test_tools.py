"""Tools via MCP: catalog behaviour, specialist tool-loop, trace, secrets,
external MCP servers. HTTP seam + stub provider; Postgres required."""

import json
import threading
import uuid

import psycopg
import pytest
import uvicorn

from app.config import settings
from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


STUB = {"provider": "stub", "model": "stub-1"}


def _payload(message: str, agents: list, **kwargs) -> dict:
    return {
        "thread_id": f"t-{uuid.uuid4().hex[:8]}",
        "message": message,
        "supervisor": {"prompt": "Coordene.", "model": STUB},
        "agents": agents,
        "max_steps": 4,
        **kwargs,
    }


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def test_tools_catalog_lists_platform_tools(client):
    r = await client.get("/v1/tools")
    names = [t["name"] for t in r.json()]
    assert "calculate" in names and "call_http_api" in names


async def test_specialist_uses_calculate_and_the_call_is_traced(client):
    thread = f"t-{uuid.uuid4().hex[:8]}"
    await _script(
        client,
        [
            ["quanto é", 'TOOL:contador_agent:{"task": "calcule 17*23"}'],
            ["calcule 17*23", 'TOOL:calculate:{"expression": "17*23"}'],
            ["391", "o resultado é 391"],
        ],
        default="segue o resultado",
    )
    payload = _payload(
        "quanto é 17 vezes 23?",
        agents=[
            {
                "name": "contador_agent",
                "description": "faz contas",
                "prompt": "Você faz contas.",
                "model": STUB,
                "tools": ["calculate"],
            }
        ],
    )
    payload["thread_id"] = thread
    r = await client.post("/v1/runs", json=payload)
    events = _events(r.text)

    tool_events = [d for e, d in events if e == "tool"]
    assert any(t["tool"] == "calculate" and t["status"] == "ok" for t in tool_events)
    assert any(e == "done" for e, _ in events)

    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT tool_name, status, output FROM tool_calls WHERE chat_id = %s",
            (thread,),
        ).fetchall()
    assert any(r[0] == "calculate" and r[1] == "ok" and "391" in (r[2] or "") for r in rows)


async def test_specialist_without_the_tool_cannot_call_it(client):
    await _script(
        client,
        [
            ["conta", 'TOOL:sem_tool_agent:{"task": "calcule 2+2"}'],
            ["calcule 2+2", 'TOOL:calculate:{"expression": "2+2"}'],
        ],
        default="não consegui usar tools",
    )
    payload = _payload(
        "faça a conta",
        agents=[
            {
                "name": "sem_tool_agent",
                "description": "sem tools",
                "prompt": "Você não tem tools.",
                "model": STUB,
                "tools": [],  # calculate NOT allowed
            }
        ],
    )
    r = await client.post("/v1/runs", json=payload)
    events = _events(r.text)
    # The stub tries to call calculate, but with no tools offered the loop
    # records the attempt as an unavailable tool error — never a crash.
    tool_events = [d for e, d in events if e == "tool"]
    assert all(t["tool"] != "calculate" or t["status"] == "error" for t in tool_events)
    assert any(e == "done" for e, _ in events)


async def test_http_tool_resolves_secret_placeholders(client):
    """{{secret:TOKEN}} must reach the HTTP server resolved, without the
    value ever appearing in the model-visible transcript."""
    captured = {}
    from fastapi import FastAPI, Request

    echo = FastAPI()

    @echo.get("/ping")
    async def ping(request: Request):
        captured["auth"] = request.headers.get("authorization", "")
        return {"pong": True}

    config = uvicorn.Config(echo, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]

    await _script(
        client,
        [
            [
                "chame a api",
                "TOOL:integrador_agent:" + json.dumps({"task": "chame o ping"}),
            ],
            [
                "chame o ping",
                "TOOL:call_http_api:"
                + json.dumps(
                    {
                        "url": f"http://127.0.0.1:{port}/ping",
                        "headers_json": '{"Authorization": "Bearer {{secret:API_TOKEN}}"}',
                    }
                ),
            ],
            ["pong", "a api respondeu pong"],
        ],
        default="terminei",
    )
    payload = _payload(
        "chame a api",
        agents=[
            {
                "name": "integrador_agent",
                "description": "integra APIs",
                "prompt": "Você chama APIs.",
                "model": STUB,
                "tools": ["call_http_api"],
            }
        ],
        secrets={"API_TOKEN": "valor-super-secreto"},
    )
    r = await client.post("/v1/runs", json=payload)
    server.should_exit = True

    assert captured["auth"] == "Bearer valor-super-secreto"
    assert "valor-super-secreto" not in r.text  # never leaks into the stream


async def test_external_mcp_server_tool_is_callable(client):
    """A template-declared external MCP server (streamable HTTP) exposes its
    tools as ext_<name>_<tool> and they are callable end to end."""
    from mcp.server.fastmcp import FastMCP

    external = FastMCP("clima", host="127.0.0.1", port=0)

    @external.tool()
    def clima_atual(cidade: str) -> str:
        """Clima de uma cidade."""
        return f"25 graus e sol em {cidade}"

    starlette_app = external.streamable_http_app()
    config = uvicorn.Config(starlette_app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]

    await _script(
        client,
        [
            ["tempo em", 'TOOL:clima_agent:{"task": "qual o clima em Recife?"}'],
            [
                "qual o clima em recife",
                'TOOL:ext_clima_clima_atual:{"cidade": "Recife"}',
            ],
            ["25 graus", "faz 25 graus e sol em Recife"],
        ],
        default="sem clima",
    )
    payload = _payload(
        "como está o tempo em Recife?",
        agents=[
            {
                "name": "clima_agent",
                "description": "previsão do tempo",
                "prompt": "Você informa o clima.",
                "model": STUB,
                "tools": ["ext_clima_clima_atual"],
            }
        ],
        mcp_servers=[{"name": "clima", "url": f"http://127.0.0.1:{port}/mcp"}],
    )
    r = await client.post("/v1/runs", json=payload)
    server.should_exit = True

    events = _events(r.text)
    tool_events = [d for e, d in events if e == "tool"]
    assert any(
        t["tool"] == "ext_clima_clima_atual" and t["status"] == "ok" for t in tool_events
    )
    done = next(d for e, d in events if e == "done")
    assert "25 graus" in done["text"]
