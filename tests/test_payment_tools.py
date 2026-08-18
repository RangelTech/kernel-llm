"""generate_pix_charge / check_payment_status, exercitados pelo contrato MCP
contra um Mercado Pago falso."""

import json
import threading

import pytest
import uvicorn

from app.config import settings
from app.tools import open_catalog_session, set_current_agent, set_run_context

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _tool_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def _start_fake_gateway():
    from fastapi import FastAPI, Request

    fake = FastAPI()
    seen: dict = {}

    @fake.post("/v1/payments")
    async def create(request: Request):
        seen["body"] = await request.json()
        seen["auth"] = request.headers.get("authorization")
        return {
            "id": 1234567890,
            "status": "pending",
            "transaction_amount": seen["body"]["transaction_amount"],
            "point_of_interaction": {
                "transaction_data": {
                    "qr_code": "00020126PIX-COPIA-E-COLA",
                    "qr_code_base64": "aGVsbG8=",
                    "ticket_url": "https://mp.example/ticket/1234567890",
                }
            },
        }

    @fake.get("/v1/payments/{payment_id}")
    async def read(payment_id: str):
        # 999... = cobrança ainda pendente, como o gateway devolve de verdade:
        # com o copia-e-cola e o QR junto do status.
        if payment_id.startswith("999"):
            return {
                "id": int(payment_id),
                "status": "pending",
                "transaction_amount": 0.01,
                "point_of_interaction": {
                    "transaction_data": {
                        "qr_code": "00020126PIX-COPIA-E-COLA",
                        "qr_code_base64": "aGVsbG8=",
                    }
                },
            }
        return {"id": int(payment_id), "status": "approved", "transaction_amount": 48.9}

    config = uvicorn.Config(fake, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, port, seen


def _context(payment: dict) -> None:
    set_run_context(
        secrets={},
        datasources=[],
        tenant_id=None,
        chat_id=None,
        payment=payment,
    )
    set_current_agent("caixa")


async def test_charge_uses_the_requested_amount_not_a_test_value():
    """O valor cobrado é o valor pedido — sandbox não pode virar R$ 0,01."""
    server, port, seen = _start_fake_gateway()
    original = settings.mercado_pago_api
    settings.mercado_pago_api = f"http://127.0.0.1:{port}"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            result = await session.call_tool(
                "generate_pix_charge",
                {"amount": "48,90", "description": "Pedido 42", "reference_id": "42"},
            )
        payload = json.loads(_tool_text(result))
    finally:
        settings.mercado_pago_api = original
        server.should_exit = True

    assert seen["body"]["transaction_amount"] == 48.90
    assert seen["body"]["payment_method_id"] == "pix"
    assert seen["auth"] == "Bearer TEST-token"
    assert payload["status"] == "ok"
    assert payload["amount"] == "48.90"
    assert payload["pix_copia_e_cola"] == "00020126PIX-COPIA-E-COLA"
    assert payload["payment_id"] == "1234567890"


async def test_charge_without_credential_explains_instead_of_crashing():
    _context({})
    async with open_catalog_session() as session:
        result = await session.call_tool("generate_pix_charge", {"amount": "10.00"})
    assert "credencial de pagamento" in _tool_text(result)


async def test_charge_rejects_non_positive_amount():
    _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
    async with open_catalog_session() as session:
        result = await session.call_tool("generate_pix_charge", {"amount": "0"})
    assert "maior que zero" in _tool_text(result)


async def test_check_status_reports_paid():
    server, port, _ = _start_fake_gateway()
    original = settings.mercado_pago_api
    settings.mercado_pago_api = f"http://127.0.0.1:{port}"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            result = await session.call_tool("check_payment_status", {"payment_id": "1234567890"})
        payload = json.loads(_tool_text(result))
    finally:
        settings.mercado_pago_api = original
        server.should_exit = True

    assert payload["paid"] is True
    assert payload["status"] == "paid"


async def test_o_qr_publicado_volta_como_descriptor_no_retorno(monkeypatch):
    """Publicar o QR não basta: o descriptor precisa aparecer no retorno da tool.

    O kernel só emite o evento `artifact` para o cliente quando acha um
    `artifact_id` dentro do que a tool devolveu (`_find_artifact_descriptors`).
    Sem isso o QR era gravado no storage e nunca chegava à tela — e o modelo,
    não vendo imagem nenhuma, respondia ao usuário que não conseguia exibir QR
    Code, exatamente o oposto do que tinha acabado de acontecer."""
    publicados = []

    async def falso_register(**kwargs):
        publicados.append(kwargs)
        return {"artifact_id": "art-1", "kind": kwargs["kind"], "title": kwargs["title"]}

    import app.storage

    monkeypatch.setattr(app.storage, "register_artifact", falso_register)

    server, port, _ = _start_fake_gateway()
    original = settings.mercado_pago_api
    settings.mercado_pago_api = f"http://127.0.0.1:{port}"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            result = await session.call_tool("generate_pix_charge", {"amount": "0.01"})
        payload = json.loads(_tool_text(result))
    finally:
        settings.mercado_pago_api = original
        server.should_exit = True

    from app.graph import _find_artifact_descriptors

    assert publicados and publicados[0]["kind"] == "image"
    assert payload["qr_code_exibido"] is True
    assert [d["artifact_id"] for d in _find_artifact_descriptors(payload)] == ["art-1"]


async def test_check_status_devolve_o_copia_e_cola_da_cobranca_existente():
    """Consultar precisa entregar o código da cobrança que já existe.

    Sem isto, "me mostre o código de novo" só tinha uma saída: chamar
    generate_pix_charge outra vez — e cada repetição criava uma cobrança nova e
    pagável. Foi exatamente o que aconteceu no QA (três cobranças para exibir
    uma)."""
    server, port, _ = _start_fake_gateway()
    original = settings.mercado_pago_api
    settings.mercado_pago_api = f"http://127.0.0.1:{port}"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            result = await session.call_tool("check_payment_status", {"payment_id": "999123"})
        payload = json.loads(_tool_text(result))
    finally:
        settings.mercado_pago_api = original
        server.should_exit = True

    assert payload["status"] == "pending"
    assert payload["paid"] is False
    assert payload["pix_copia_e_cola"] == "00020126PIX-COPIA-E-COLA"


async def test_check_status_pago_nao_reexibe_o_qr():
    """Cobrança liquidada não pode voltar a mostrar QR — convidaria a pagar duas
    vezes. O copia-e-cola some junto porque o gateway não o devolve mais."""
    server, port, _ = _start_fake_gateway()
    original = settings.mercado_pago_api
    settings.mercado_pago_api = f"http://127.0.0.1:{port}"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            result = await session.call_tool("check_payment_status", {"payment_id": "1234567890"})
        payload = json.loads(_tool_text(result))
    finally:
        settings.mercado_pago_api = original
        server.should_exit = True

    assert payload["paid"] is True
    assert payload["qr_code_exibido"] is False


async def test_check_status_requires_an_identifier():
    _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
    async with open_catalog_session() as session:
        result = await session.call_tool("check_payment_status", {})
    assert "informe payment_id ou reference_id" in _tool_text(result).lower()


async def test_gateway_fora_do_ar_nao_vira_cobranca_fantasma():
    """Gateway inacessível precisa virar erro legível, nunca "cobrança gerada".

    É o modo de falha que custa dinheiro nos dois sentidos: se a tool responde
    algo ambíguo, o modelo anuncia uma cobrança que não existe e o cliente
    espera um PIX que nunca vai cair; se ela vaza o stacktrace, o modelo repete
    a chamada achando que foi problema de formato.

    A porta 9 é o descarte do TCP: conexão recusada na hora, sem espera.
    """
    original = settings.mercado_pago_api
    settings.mercado_pago_api = "http://127.0.0.1:9"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            resultado = await session.call_tool("generate_pix_charge", {"amount": "10.00"})
        texto = _tool_text(resultado)
    finally:
        settings.mercado_pago_api = original

    assert texto.startswith("ERRO")
    assert "cobrança" in texto.lower()
    # Nada que o modelo possa ler como sucesso.
    assert "payment_id" not in texto
    assert '"status": "ok"' not in texto


async def test_consulta_com_gateway_fora_do_ar_nao_inventa_status():
    """Pior que não responder é responder "pago" quando ninguém sabe."""
    original = settings.mercado_pago_api
    settings.mercado_pago_api = "http://127.0.0.1:9"
    try:
        _context({"provider": "mercado_pago", "access_token": "TEST-token", "sandbox": True})
        async with open_catalog_session() as session:
            resultado = await session.call_tool(
                "check_payment_status", {"payment_id": "1234567890"}
            )
        texto = _tool_text(resultado)
    finally:
        settings.mercado_pago_api = original

    assert texto.startswith("ERRO")
    assert "paid" not in texto
    assert "pix_copia_e_cola" not in texto
