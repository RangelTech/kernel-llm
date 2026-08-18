"""web_search (Serper + DuckDuckGo fallback) and analyze_pdf_pages, exercised
through the MCP contract seam."""

import json
import threading
import uuid

import pytest
import uvicorn

from app.config import settings
from app.storage import save_payload
from app.tools import open_catalog_session, set_current_agent, set_run_context

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _tool_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


async def test_web_search_uses_serper_when_configured():
    from fastapi import FastAPI

    fake = FastAPI()

    @fake.post("/search")
    async def search(body: dict):
        assert body["q"] == "cotação do dólar"
        return {
            "organic": [
                {"title": "Dólar hoje", "link": "https://ex.com/d", "snippet": "R$ 5,40"},
                {"title": "Câmbio", "link": "https://ex.com/c", "snippet": "estável"},
            ]
        }

    config = uvicorn.Config(fake, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]

    old_key, old_url = settings.serper_api_key, settings.serper_url
    settings.serper_api_key = "test-key"
    settings.serper_url = f"http://127.0.0.1:{port}/search"
    try:
        set_run_context(secrets={}, datasources=[], tenant_id=None, chat_id=None)
        async with open_catalog_session() as session:
            result = await session.call_tool("web_search", {"query": "cotação do dólar"})
    finally:
        settings.serper_api_key, settings.serper_url = old_key, old_url
        server.should_exit = True

    results = json.loads(_tool_text(result))
    assert results[0]["title"] == "Dólar hoje"
    assert results[0]["url"] == "https://ex.com/d"


async def test_web_search_reports_when_everything_fails(monkeypatch):
    old_key, old_url = settings.serper_api_key, settings.serper_url
    settings.serper_api_key = "test-key"
    settings.serper_url = "http://127.0.0.1:59998/search"  # dead
    # Kill the fallback too.
    import app.tools as tools_module

    def _boom(*_a, **_k):
        raise RuntimeError("ddg indisponível")

    monkeypatch.setitem(__import__("sys").modules, "ddgs", None)
    try:
        set_run_context(secrets={}, datasources=[], tenant_id=None, chat_id=None)
        async with open_catalog_session() as session:
            result = await session.call_tool("web_search", {"query": "qualquer"})
    finally:
        settings.serper_api_key, settings.serper_url = old_key, old_url

    assert "ERRO" in _tool_text(result)
    assert tools_module is not None


async def test_analyze_pdf_pages_runs_vision_per_page(client):
    import fitz

    document = fitz.open()
    for index, label in enumerate(["PRODUTO A CIRCULADO QTD 3", "NADA MARCADO"]):
        page = document.new_page()
        page.insert_text((72, 100), f"Pagina {index + 1}: {label}", fontsize=14)
    pdf_bytes = document.tobytes()
    document.close()

    path = save_payload(f"tests/pdf/{uuid.uuid4()}/catalogo.pdf", pdf_bytes, "application/pdf")

    # Vision is the stub: per-page scripted answers keyed on the page marker.
    await client.post(
        "/stub/script",
        json={
            "rules": [
                ["página 1]", '{"produtos": [{"nome": "PRODUTO A", "qtd": 3}]}'],
                ["página 2]", '{"produtos": []}'],
            ],
            "default": "{}",
        },
    )

    set_run_context(
        secrets={},
        datasources=[],
        tenant_id=None,
        chat_id=None,
        attachments=[
            {
                "kind": "file",
                "name": "catalogo.pdf",
                "content_type": "application/pdf",
                "storage_path": path,
            }
        ],
    )
    set_current_agent("ocr_agent", {"provider": "stub", "model": "stub-1"})

    async with open_catalog_session() as session:
        result = await session.call_tool(
            "analyze_pdf_pages",
            {
                "attachment_name": "catalogo.pdf",
                "instruction": "liste produtos circulados e quantidades",
            },
        )
    body = json.loads(_tool_text(result))
    assert body["pages_analyzed"] == 2 and body["total_pages"] == 2
    page1 = json.loads(body["pages"][0]["result"])
    assert page1["produtos"][0] == {"nome": "PRODUTO A", "qtd": 3}
    page2 = json.loads(body["pages"][1]["result"])
    assert page2["produtos"] == []


async def test_analyze_pdf_pages_unknown_attachment():
    set_run_context(secrets={}, datasources=[], tenant_id=None, chat_id=None)
    async with open_catalog_session() as session:
        result = await session.call_tool(
            "analyze_pdf_pages",
            {"attachment_name": "sumiu.pdf", "instruction": "x"},
        )
    assert "não encontrado" in _tool_text(result)
