"""Tools do Outlook/Microsoft Graph, exercitadas pelo contrato MCP contra
uma API do Graph falsa -- mesmo padrão de test_google_tools.py, já cobrindo
multi-conta desde o início (produto-08 §12 nasce sem o gap que o Google
teve no §9)."""

import json

import pytest

from app.tools import open_catalog_session, set_current_agent, set_run_context

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _tool_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def _context(microsoft_accounts: list[dict]) -> None:
    set_run_context(
        secrets={},
        datasources=[],
        tenant_id=None,
        chat_id=None,
        microsoft_accounts=microsoft_accounts,
    )
    set_current_agent("recepcao")


async def test_sem_conta_conectada_explica_em_vez_de_travar():
    _context([])
    async with open_catalog_session() as session:
        result = await session.call_tool("outlook_calendar_list_events", {})
    assert "ainda não conectou" in _tool_text(result)


async def test_com_duas_contas_label_escolhe_a_certa(monkeypatch):
    """Caso real do dono: mais de uma agenda Outlook conectada -- o LLM
    escolhe pelo `label`, nunca cai na conta errada."""
    import app.tools as tools_module

    calls = []

    async def fake_get(url, token, params=None):
        calls.append(token)

        class Resp:
            status_code = 200

            def json(self_inner):
                return {"value": []}

        return Resp()

    monkeypatch.setattr(tools_module, "_ms_get", fake_get)
    _context(
        [
            {
                "label": "Recepção",
                "access_token": "token-recepcao",
                "email_address": "recepcao@empresa.com",
            },
            {
                "label": "Financeiro",
                "access_token": "token-financeiro",
                "email_address": "financeiro@empresa.com",
            },
        ]
    )
    async with open_catalog_session() as session:
        await session.call_tool("outlook_calendar_list_events", {"label": "Financeiro"})
    assert calls == ["token-financeiro"]


async def test_label_sem_correspondencia_nao_vaza_pra_outra_conta(monkeypatch):
    import app.tools as tools_module

    calls = []

    async def fake_get(url, token, params=None):
        calls.append(token)

        class Resp:
            status_code = 200

            def json(self_inner):
                return {"value": []}

        return Resp()

    monkeypatch.setattr(tools_module, "_ms_get", fake_get)
    _context(
        [
            {
                "label": "Recepção",
                "access_token": "token-recepcao",
                "email_address": "recepcao@empresa.com",
            }
        ]
    )
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "outlook_calendar_list_events", {"label": "Inexistente"}
        )
    assert "ainda não conectou" in _tool_text(result)
    assert calls == []


async def test_criar_evento_com_reuniao_teams_pede_o_campo_certo(monkeypatch):
    """Confirma que `criar_reuniao_teams=True` monta o corpo certo pro Graph
    gerar o link do Teams -- não é uma API separada."""
    import app.tools as tools_module

    seen = {}

    async def fake_post(url, token, json_body):
        seen["url"] = url
        seen["body"] = json_body

        class Resp:
            status_code = 200

            def json(self_inner):
                return {
                    "id": "evt-1",
                    "subject": json_body["subject"],
                    "webLink": "https://outlook.office.com/evt-1",
                    "onlineMeeting": {"joinUrl": "https://teams.microsoft.com/l/meetup-join/xyz"},
                }

        return Resp()

    monkeypatch.setattr(tools_module, "_ms_post", fake_post)
    _context(
        [
            {
                "label": "Recepção",
                "access_token": "token-recepcao",
                "email_address": "recepcao@empresa.com",
            }
        ]
    )
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "outlook_calendar_create_event",
            {
                "summary": "Reunião com cliente",
                "start": "2026-08-26T15:00:00-03:00",
                "end": "2026-08-26T15:30:00-03:00",
                "criar_reuniao_teams": True,
            },
        )
    payload = json.loads(_tool_text(result))
    assert seen["body"]["isOnlineMeeting"] is True
    assert seen["body"]["onlineMeetingProvider"] == "teamsForBusiness"
    assert payload["teams_join_url"] == "https://teams.microsoft.com/l/meetup-join/xyz"


async def test_criar_evento_sem_teams_nao_manda_campo_de_reuniao(monkeypatch):
    import app.tools as tools_module

    seen = {}

    async def fake_post(url, token, json_body):
        seen["body"] = json_body

        class Resp:
            status_code = 200

            def json(self_inner):
                return {"id": "evt-2", "subject": json_body["subject"], "webLink": "https://outlook.office.com/evt-2"}

        return Resp()

    monkeypatch.setattr(tools_module, "_ms_post", fake_post)
    _context(
        [
            {
                "label": "Recepção",
                "access_token": "token-recepcao",
                "email_address": "recepcao@empresa.com",
            }
        ]
    )
    async with open_catalog_session() as session:
        await session.call_tool(
            "outlook_calendar_create_event",
            {
                "summary": "Reunião interna",
                "start": "2026-08-26T15:00:00-03:00",
                "end": "2026-08-26T15:30:00-03:00",
            },
        )
    assert "isOnlineMeeting" not in seen["body"]
    assert "onlineMeetingProvider" not in seen["body"]
