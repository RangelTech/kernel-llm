"""Nenhuma tabela pode sumir da lista sem o modelo saber.

Medido no QA: o dataset tinha 180 tabelas, `list_tables` devolvia as 50
primeiras e as tabelas do SIOPE — o assunto inteiro do template — estavam nas
posições 75 a 80. O modelo não sabia que existiam, então passou o turno
garimpando INFORMATION_SCHEMA: 54 consultas, mais de dez minutos, nenhuma
resposta.

Coluna de tudo estoura o contexto, e por isso continua limitada. Nome de tabela
é barato, e é o que faz o modelo pedir a coluna certa de uma vez.
"""

import pytest
from app import datasources

pytestmark = pytest.mark.anyio


def test_tabela_alem_do_limite_aparece_sem_as_colunas():
    nomes = [f"t{i:03d}" for i in range(180)]
    catalogo = datasources._catalogo(nomes, lambda n: [{"name": "x", "type": "INT64"}])

    assert len(catalogo) == 180, "tabela sumiu da lista"
    assert catalogo[0]["columns"] == [{"name": "x", "type": "INT64"}]
    assert catalogo[100]["table"] == "t100"
    assert "INFORMATION_SCHEMA" in catalogo[100]["columns"], (
        "sem coluna e sem dizer onde buscar, o modelo garimpa"
    )


def test_so_as_primeiras_pagam_a_consulta_de_colunas():
    """Detalhar coluna custa uma ida por tabela — não pode virar 180 idas."""
    pedidas = []

    def colunas_de(nome):
        pedidas.append(nome)
        return []

    datasources._catalogo([f"t{i}" for i in range(180)], colunas_de)
    assert len(pedidas) == datasources.TABELAS_COM_COLUNAS


async def test_sql_lista_tabela_que_ficou_sem_coluna(monkeypatch):
    """A consulta de colunas é limitada: uma tabela larga consome a cota e as
    seguintes voltariam sem nada. Elas continuam na lista."""
    chamadas = []

    async def _execute_query(datasource, query, max_rows=None):
        chamadas.append(query)
        if "information_schema.tables" in query:
            return ["t"], [["public.larga"], ["public.esquecida"]]
        return ["t", "c", "d"], [["public.larga", "id", "integer"]]

    monkeypatch.setattr(datasources, "execute_query", _execute_query)

    catalogo = await datasources.list_tables({"kind": "postgresql"})

    assert [t["table"] for t in catalogo] == ["public.larga", "public.esquecida"]
    assert catalogo[0]["columns"] == [{"name": "id", "type": "integer"}]
    assert "INFORMATION_SCHEMA" in catalogo[1]["columns"]


def test_fonte_pode_declarar_quais_tabelas_valem():
    """Num lake de 180 tabelas, quem configura sabe quais 7 interessam."""
    nomes = ["semantic_zone.obt_a", "semantic_zone.obt_siope", "semantic_zone.obt_z"]
    fonte = {"kind": "bigquery", "config": {"tables": ["obt_siope"]}}

    assert datasources._escolhidas(fonte, nomes) == ["semantic_zone.obt_siope"]


def test_nome_completo_tambem_serve():
    nomes = ["semantic_zone.obt_siope", "semantic_zone.obt_z"]
    fonte = {"config": {"tables": ["semantic_zone.obt_siope"]}}

    assert datasources._escolhidas(fonte, nomes) == ["semantic_zone.obt_siope"]


def test_sem_declaracao_nada_muda():
    nomes = ["a", "b"]
    assert datasources._escolhidas({"config": {}}, nomes) == nomes
    assert datasources._escolhidas({}, nomes) == nomes
