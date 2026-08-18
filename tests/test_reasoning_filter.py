"""O rascunho do modelo não pode chegar ao usuário.

Modelos de raciocínio emitem `<think>…</think>` junto da resposta. Como um
combo mistura famílias de modelo, um provedor emite a tag e outro não — então
isso não é caso raro, é o caminho normal quando o cliente usa combo.

O que torna isso mais do que um `replace`: no streaming a tag chega partida
entre pedaços, e um pedaço que termina em `<thi` não pode ser mostrado antes de
o próximo provar o que ele é.
"""

import pytest
from app.providers import ReasoningFilter, strip_reasoning


@pytest.mark.parametrize(
    ("pedacos", "esperado"),
    [
        (["<think>rascunho</think>Resposta"], "Resposta"),
        (["<thi", "nk>oculto</thi", "nk>visível"], "visível"),
        (["antes <think>meio</think> depois"], "antes  depois"),
        (["sem tag nenhuma"], "sem tag nenhuma"),
        (["<think></think>Olá"], "Olá"),
        (["a<t", "hink>x</think>b"], "ab"),
        (["<think>um</think>meio<think>dois</think>fim"], "meiofim"),
        # `<` solto é texto comum e não pode sumir.
        (["custo < 100 reais"], "custo < 100 reais"),
    ],
)
def test_remove_o_rascunho_mesmo_partido_entre_pedacos(pedacos, esperado):
    filtro = ReasoningFilter()
    saida = "".join(filtro.feed(p) for p in pedacos) + filtro.flush()
    assert saida == esperado


def test_bloco_sem_fechamento_nao_vaza():
    """Se o modelo abriu o rascunho e a conexão morreu, mostrar o rascunho é
    pior do que não mostrar nada."""
    filtro = ReasoningFilter()
    assert filtro.feed("<think>pensando ainda") == ""
    assert filtro.flush() == ""


def test_resposta_inteira_de_uma_vez():
    assert strip_reasoning("<think>x</think>pronto") == "pronto"
    assert strip_reasoning("nada a remover") == "nada a remover"
