"""Quanto da conversa vai para o modelo, e o que acontece com o resto.

Antes disto a conversa inteira ia a cada turno: o custo crescia junto com ela
e, longa o bastante, o turno falhava no limite do modelo em vez de degradar.

As duas opções são do template porque a escolha afeta qualidade. Cortar é
barato e perde o começo; resumir preserva o sentido e custa uma chamada a mais.
"""

import pytest

from app import graph

pytestmark = pytest.mark.anyio


class _Msg:
    """Mensagem no formato que o grafo entrega (só o que o corte usa)."""

    def __init__(self, texto, tipo="human"):
        self.content = texto
        self.type = tipo


def _conversa(n: int) -> dict:
    return {
        "messages": [
            _Msg(f"m{i}", "human" if i % 2 == 0 else "ai") for i in range(n)
        ]
    }


async def test_conversa_curta_vai_inteira():
    saida = await graph._historico_para_o_modelo(
        _conversa(10), {"history_limit": 100}
    )
    assert len(saida) == 10
    assert all(marca is None for marca, _ in saida)


async def test_conversa_longa_corta_no_limite():
    """Sem compressão, sobram as mais recentes — o começo se perde."""
    saida = await graph._historico_para_o_modelo(
        _conversa(90), {"history_limit": 40, "compress_history": False}
    )
    assert len(saida) == 20
    assert saida[-1][1].content == "m89"
    assert saida[0][1].content == "m70"


async def test_compressao_troca_o_comeco_por_um_resumo(monkeypatch):
    """90 mensagens com limite 90 e compressão: 45 recentes + 1 resumo."""

    async def _resumir(mensagens, run_config):
        return f"resumo de {len(mensagens)} mensagens"

    monkeypatch.setattr(graph, "_resumir_trecho", _resumir)

    saida = await graph._historico_para_o_modelo(
        _conversa(91), {"history_limit": 90, "compress_history": True}
    )
    assert len(saida) == 46, "deveria ser 45 recentes + 1 resumo"
    assert saida[0][0] == "resumo"
    assert "46 mensagens" in saida[0][1]
    assert saida[-1][1].content == "m90"


async def test_resumo_que_falha_nao_derruba_o_turno(monkeypatch):
    """Sem resumo, degrada para o corte simples — não para a conversa."""

    async def _resumir(mensagens, run_config):
        return None

    monkeypatch.setattr(graph, "_resumir_trecho", _resumir)

    saida = await graph._historico_para_o_modelo(
        _conversa(90), {"history_limit": 40, "compress_history": True}
    )
    assert len(saida) == 20
    assert all(marca is None for marca, _ in saida)


async def test_limite_padrao_quando_o_template_nao_diz():
    saida = await graph._historico_para_o_modelo(_conversa(150), {})
    assert len(saida) == 50, "padrão 100 -> metade recente"


async def test_limite_de_um_nao_devolve_a_conversa_inteira():
    """`limite // 2` vira 0 e `mensagens[-0:]` é a lista toda.

    A API hoje recusa limite abaixo de 4, mas o kernel atende quem o chamar, e
    um corte que devolve tudo é pior que corte nenhum: o dono do template pediu
    o menor contexto possível e receberia o maior.
    """
    saida = await graph._historico_para_o_modelo(_conversa(30), {"history_limit": 1})
    assert len(saida) == 1
    assert saida[-1][1].content == "m29"
