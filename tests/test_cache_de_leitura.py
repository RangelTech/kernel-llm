"""Consulta idêntica repetida no mesmo turno não vai ao banco de novo.

Observado numa conversa real: um especialista rodou 45 consultas num único
turno, várias byte a byte idênticas, e o turno levou nove minutos. Nada no
retorno dizia que aquilo já tinha sido perguntado, então o modelo repetia.

O aviso no retorno é tão importante quanto o cache: devolver o mesmo resultado
em silêncio economiza a consulta, mas não quebra o laço.
"""

import json

import pytest

from app import tools

pytestmark = pytest.mark.anyio


@pytest.fixture
async def contexto_com_fonte(monkeypatch):
    """Precisa ser async: `set_run_context` grava num ContextVar, e um fixture
    síncrono roda fora da task onde o teste roda — o valor não chega lá. Rodando
    sozinho o arquivo passava; junto com qualquer outro teste async, `lake`
    sumia e a ferramenta devolvia "fonte não existe" antes de tocar o cache."""
    fonte = {"name": "lake", "kind": "postgresql"}
    tools.set_run_context(
        secrets={}, datasources=[fonte], tenant_id="t1", chat_id="c1"
    )

    chamadas = []

    async def _execute_query(source, query, limite):
        chamadas.append(query)
        return ["a"], [[1]]

    async def _register_artifact(**kwargs):
        return {"artifact_id": "art-1", "rows": kwargs.get("row_count")}

    monkeypatch.setattr("app.datasources.execute_query", _execute_query)
    monkeypatch.setattr("app.storage.register_artifact", _register_artifact)
    return chamadas


async def test_consulta_repetida_nao_vai_ao_banco(contexto_com_fonte):
    primeira = await tools.run_sql_query(datasource="lake", query="SELECT 1")
    segunda = await tools.run_sql_query(datasource="lake", query="SELECT 1")

    assert len(contexto_com_fonte) == 1, "a segunda consulta foi ao banco"
    assert "AVISO" in segunda, "o modelo precisa saber que repetiu"
    assert json.loads(primeira)["artifact_id"] in segunda


async def test_espaco_em_branco_nao_engana_o_cache(contexto_com_fonte):
    await tools.run_sql_query(datasource="lake", query="SELECT 1")
    await tools.run_sql_query(datasource="lake", query="SELECT   1\n")
    assert len(contexto_com_fonte) == 1


async def test_consulta_diferente_executa(contexto_com_fonte):
    await tools.run_sql_query(datasource="lake", query="SELECT 1")
    await tools.run_sql_query(datasource="lake", query="SELECT 2")
    assert len(contexto_com_fonte) == 2


async def test_escrita_invalida_a_leitura(contexto_com_fonte, monkeypatch):
    """Reaproveitar leitura feita antes de uma escrita mostraria dado velho."""

    async def _execute_write(source, statement, tabelas):
        return "pedidos", 1, []

    monkeypatch.setattr("app.datasources.execute_write", _execute_write)

    await tools.run_sql_query(datasource="lake", query="SELECT 1")
    await tools.execute_sql_write(
        datasource="lake", statement="INSERT INTO pedidos (x) VALUES (1)"
    )
    await tools.run_sql_query(datasource="lake", query="SELECT 1")

    assert len(contexto_com_fonte) == 2, "a leitura após a escrita usou cache velho"


async def test_grafico_no_sandbox_avisa_que_nao_aparece(monkeypatch):
    """O sandbox publica dataset e documento — imagem, não.

    Observado em conversa real: o agente plotou com matplotlib dentro do
    sandbox, o desenho se perdeu e ele anunciou "gráfico gerado com sucesso".
    O usuário ficou sem gráfico e sem saber disso.
    """
    tools.set_run_context(secrets={}, datasources=[], tenant_id="t1", chat_id="c1")

    async def _run_sandboxed(code, dataset, timeout_seconds=30):
        return ("", "", [], False)

    monkeypatch.setattr("app.sandbox.run_sandboxed", _run_sandboxed)

    saida = await tools.execute_python(
        code="import matplotlib.pyplot as plt\nplt.plot([1,2,3])"
    )
    assert "AVISO" in saida
    assert "generate_chart" in saida


async def test_sem_grafico_nao_avisa(monkeypatch):
    tools.set_run_context(secrets={}, datasources=[], tenant_id="t1", chat_id="c1")

    async def _run_sandboxed(code, dataset, timeout_seconds=30):
        return ("42", "", [], False)

    monkeypatch.setattr("app.sandbox.run_sandboxed", _run_sandboxed)

    saida = await tools.execute_python(code="print(6*7)")
    assert "AVISO" not in saida


async def test_catalogo_expoe_as_ferramentas_certas():
    """Helper interno não pode virar ferramenta, e ferramenta não pode sumir.

    Já aconteceu: um helper foi inserido logo acima de `describe_datasources`
    e ficou com o `@catalog.tool()` que era dela. O catálogo passou a oferecer
    o helper ao modelo e perdeu a ferramenta — sem erro nenhum, porque nada
    valida isso em tempo de importação.
    """
    # O catálogo só fica completo quando `app.main` importa `tools_output` — é
    # de lá que vêm gráfico, xlsx e pdf. Olhar só `app.tools` mede um catálogo
    # que nenhum modelo recebe.
    from app import main  # noqa: F401 — registra as ferramentas de saída

    nomes = {t.name for t in await tools.catalog.list_tools()}

    esperadas = {
        "describe_datasources",
        "run_sql_query",
        "execute_python",
        "generate_chart",
        "export_xlsx",
        "generate_pix_charge",
        "check_payment_status",
    }
    assert esperadas <= nomes, f"ferramentas faltando: {esperadas - nomes}"

    internos = {n for n in nomes if n.startswith("_")}
    assert not internos, f"helper interno exposto como ferramenta: {internos}"
