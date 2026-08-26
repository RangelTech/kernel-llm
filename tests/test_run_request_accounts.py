"""Achado real 26/08/2026: `email_accounts`/`google_accounts`/
`microsoft_accounts` nunca tinham sido declarados em `RunRequest`
(`app/runs.py`) -- o backend sempre mandou no corpo do `POST /v1/runs`,
mas o Pydantic descarta campo não declarado por padrão, então a conta
real conectada no banco nunca chegava no contexto de execução da tool.
As tools sempre respondiam "ainda não conectou" pra qualquer conta real,
mesmo com o token certo salvo no Postgres -- nunca detectado porque os
testes de `test_google_tools.py`/`test_microsoft_tools.py` chamam
`set_run_context()` direto (in-process), pulando inteiramente a validação
Pydantic do `/v1/runs` que é onde o bug vivia. Este arquivo cobre
especificamente essa camada -- POST HTTP real, contrato de verdade."""

import uuid

import pytest

from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STUB = {"provider": "stub", "model": "stub-1"}


def _payload(message: str, **extra) -> dict:
    return {
        "thread_id": f"t-{uuid.uuid4().hex[:8]}",
        "message": message,
        "supervisor": {"prompt": "Coordene.", "model": STUB},
        "agents": [
            {
                "name": "secretaria",
                "description": "agenda",
                "prompt": "Você marca reuniões.",
                "model": STUB,
                "tools": ["google_calendar_list_events"],
            }
        ],
        "max_steps": 4,
        **extra,
    }


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def test_google_accounts_do_corpo_http_chegam_na_tool(client, monkeypatch):
    """A prova real: manda `google_accounts` no JSON do POST (exatamente
    como o backend manda), confirma que a tool recebe o token de verdade --
    não `set_run_context` chamado direto."""
    import app.tools as tools_module

    calls = []

    async def fake_get(url, token, params=None):
        calls.append(token)

        class Resp:
            status_code = 200

            def json(self_inner):
                return {"items": []}

        return Resp()

    monkeypatch.setattr(tools_module, "_google_get", fake_get)
    await _script(
        client,
        [
            ["reuniões", 'TOOL:secretaria:{"task": "veja a agenda"}'],
            ["veja a agenda", "TOOL:google_calendar_list_events:{}"],
            ["nenhuma reunião", "sem reuniões marcadas"],
        ],
        default="ok",
    )
    payload = _payload(
        "quais são minhas reuniões?",
        google_accounts=[
            {
                "label": "Clínica",
                "access_token": "TOKEN-DO-BACKEND-DE-VERDADE",
                "email_address": "clinica@example.com",
            }
        ],
    )
    r = await client.post("/v1/runs", json=payload)
    assert r.status_code == 200
    events = _events(r.text)
    tool_events = [d for e, d in events if e == "tool"]
    assert any(
        t["tool"] == "google_calendar_list_events" and t["status"] == "ok" for t in tool_events
    )
    # Prova real: a tool chegou a fazer a chamada HTTP com o token do corpo
    # do POST -- se o campo tivesse sido descartado pelo Pydantic (o bug),
    # `calls` ficaria vazio e a tool teria respondido "ainda não conectou"
    # sem nunca chamar `_google_get`.
    assert calls == ["TOKEN-DO-BACKEND-DE-VERDADE"]


async def test_sem_google_accounts_no_corpo_tool_explica_sem_travar(client):
    """Contraprova: sem o campo no corpo (comportamento antigo, quebrado),
    a tool deve continuar respondendo educadamente, nunca travar/500."""
    await _script(
        client,
        [
            ["reuniões", 'TOOL:secretaria:{"task": "veja a agenda"}'],
            ["veja a agenda", "TOOL:google_calendar_list_events:{}"],
        ],
        default="ok",
    )
    r = await client.post("/v1/runs", json=_payload("quais são minhas reuniões?"))
    assert r.status_code == 200
    events = _events(r.text)
    assert any(e == "done" for e, _ in events)
    tool_events = [d for e, d in events if e == "tool"]
    assert any(
        t["tool"] == "google_calendar_list_events" and t["status"] == "ok" for t in tool_events
    )
