"""Tools do Google (Calendar/Sheets), exercitadas pelo contrato MCP contra
uma API do Google falsa -- cobre em especial o gap de multi-conta
(produto-08 §9): antes só a conta mais recente era usável, agora cada tool
aceita `label` pra escolher entre N contas conectadas (ex.: 1 agenda por
médico de uma clínica)."""

import json
import threading

import httpx
import pytest
import uvicorn

from app.tools import open_catalog_session, set_current_agent, set_run_context

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _tool_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def _start_fake_google():
    from fastapi import FastAPI, Request

    fake = FastAPI()
    seen: dict = {}

    @fake.get("/calendar/v3/calendars/primary/events")
    async def list_events(request: Request):
        seen["auth"] = request.headers.get("authorization")
        evento = {
            "id": "evt-1",
            "summary": "Consulta",
            "start": {"dateTime": "2026-08-26T15:00:00-03:00"},
            "end": {"dateTime": "2026-08-26T15:30:00-03:00"},
        }
        return {"items": [evento]}

    config = uvicorn.Config(fake, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, port, seen


def _context(google_accounts: list[dict]) -> None:
    set_run_context(
        secrets={},
        datasources=[],
        tenant_id=None,
        chat_id=None,
        google_accounts=google_accounts,
    )
    set_current_agent("recepcao")


async def test_sem_conta_conectada_explica_em_vez_de_travar():
    _context([])
    async with open_catalog_session() as session:
        result = await session.call_tool("google_calendar_list_events", {})
    assert "ainda não conectou" in _tool_text(result)


async def test_sem_label_usa_a_unica_conta(monkeypatch):
    import app.tools as tools_module

    async def fake_get(url, token, params=None):
        class Resp:
            status_code = 200

            def json(self_inner):
                return {"items": []}

        fake_get.seen_token = token
        return Resp()

    monkeypatch.setattr(tools_module, "_google_get", fake_get)
    _context(
        [
            {
                "label": "Dr. Fulano",
                "access_token": "token-fulano",
                "email_address": "fulano@example.com",
            }
        ]
    )
    async with open_catalog_session() as session:
        result = await session.call_tool("google_calendar_list_events", {})
    payload = json.loads(_tool_text(result))
    assert payload == {"events": []}
    assert fake_get.seen_token == "token-fulano"


async def test_com_duas_contas_label_escolhe_a_certa(monkeypatch):
    """O caso real do dono: 3 Gmail, 1 por médico -- o LLM escolhe a agenda
    certa pelo `label`, nunca cai na conta errada."""
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
    _context(
        [
            {
                "label": "Dr. Fulano",
                "access_token": "token-fulano",
                "email_address": "fulano@example.com",
            },
            {
                "label": "Dra. Beltrana",
                "access_token": "token-beltrana",
                "email_address": "beltrana@example.com",
            },
        ]
    )
    async with open_catalog_session() as session:
        await session.call_tool("google_calendar_list_events", {"label": "Dra. Beltrana"})
    assert calls == ["token-beltrana"]


async def test_label_sem_correspondencia_nao_vaza_pra_outra_conta(monkeypatch):
    """Errar o nome não pode significar 'usa qualquer uma' -- tem que ficar
    claro que não achou aquela conta, nunca silenciosamente ler a agenda
    errada."""
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
    _context(
        [
            {
                "label": "Dr. Fulano",
                "access_token": "token-fulano",
                "email_address": "fulano@example.com",
            },
            {
                "label": "Dra. Beltrana",
                "access_token": "token-beltrana",
                "email_address": "beltrana@example.com",
            },
        ]
    )
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "google_calendar_list_events", {"label": "Dr. Inexistente"}
        )
    assert "ainda não conectou" in _tool_text(result)
    assert calls == []


async def test_ponta_a_ponta_contra_api_google_fake_com_label():
    """Prova a integração real de HTTP (não só o mock do _google_get):
    request chega no servidor fake com o Authorization certo."""
    server, port, seen = _start_fake_google()
    import app.tools as tools_module

    original_get = tools_module._google_get

    async def patched_get(url, token, params=None):
        url = url.replace("https://www.googleapis.com", f"http://127.0.0.1:{port}")
        return await original_get(url, token, params)

    try:
        tools_module._google_get = patched_get
        _context(
            [
                {
                    "label": "Clínica",
                    "access_token": "TOKEN-REAL",
                    "email_address": "clinica@example.com",
                }
            ]
        )
        async with open_catalog_session() as session:
            result = await session.call_tool("google_calendar_list_events", {})
        payload = json.loads(_tool_text(result))
    finally:
        tools_module._google_get = original_get
        server.should_exit = True

    assert seen["auth"] == "Bearer TOKEN-REAL"
    assert payload["events"][0]["summary"] == "Consulta"


async def test_criar_planilha_sem_conta_conectada_explica_em_vez_de_travar():
    """26/08/2026: gap real achado -- não existia jeito de CRIAR uma
    planilha nova, só ler/escrever numa que já existia. Tool nova."""
    _context([])
    async with open_catalog_session() as session:
        result = await session.call_tool("google_sheets_create", {"title": "Reuniões"})
    assert "ainda não conectou" in _tool_text(result)


async def test_criar_planilha_sem_valores(monkeypatch):
    import app.tools as tools_module

    seen = {}

    async def fake_post(url, token, json_body):
        seen["url"] = url
        seen["body"] = json_body

        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "spreadsheetId": "sheet-abc123",
                    "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/sheet-abc123",
                }

        return Resp()

    monkeypatch.setattr(tools_module, "_google_post", fake_post)
    _context(
        [{"label": "Clínica", "access_token": "TOKEN-REAL", "email_address": "clinica@example.com"}]
    )
    async with open_catalog_session() as session:
        result = await session.call_tool("google_sheets_create", {"title": "Reuniões"})
    payload = json.loads(_tool_text(result))
    assert payload["status"] == "ok"
    assert payload["spreadsheet_id"] == "sheet-abc123"
    assert seen["url"] == "https://sheets.googleapis.com/v4/spreadsheets"
    assert seen["body"] == {"properties": {"title": "Reuniões"}}


async def test_criar_planilha_com_valores_ja_preenche_a_primeira_aba(monkeypatch):
    """`values` opcional -- confirma que o preenchimento da 1a aba usa o
    `spreadsheet_id` recém-criado e a matriz certa."""
    import app.tools as tools_module

    async def fake_post(url, token, json_body):
        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "spreadsheetId": "sheet-novo",
                    "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/sheet-novo",
                }

        return Resp()

    put_chamado = {}

    async def fake_put(self, url, **kwargs):
        put_chamado["url"] = url
        put_chamado["json"] = kwargs.get("json")
        put_chamado["headers"] = kwargs.get("headers")

        class Resp:
            status_code = 200

            def json(self_inner):
                return {}

        return Resp()

    monkeypatch.setattr(tools_module, "_google_post", fake_post)
    monkeypatch.setattr(httpx.AsyncClient, "put", fake_put)
    _context(
        [{"label": "Clínica", "access_token": "TOKEN-REAL", "email_address": "clinica@example.com"}]
    )
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "google_sheets_create",
            {"title": "Reuniões", "values": '[["Reunião", "Data"], ["Alinhamento", "2026-08-26"]]'},
        )
    payload = json.loads(_tool_text(result))
    assert payload["status"] == "ok"
    assert payload["spreadsheet_id"] == "sheet-novo"
    assert put_chamado["url"] == "https://sheets.googleapis.com/v4/spreadsheets/sheet-novo/values/A1"
    assert put_chamado["json"] == {"values": [["Reunião", "Data"], ["Alinhamento", "2026-08-26"]]}
    assert put_chamado["headers"]["Authorization"] == "Bearer TOKEN-REAL"
