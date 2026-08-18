"""Conversation runtime.

Topology: one supervisor talks to the user and calls specialists as tools
(agents-as-tools). Each specialist is a single LLM call with its own prompt
and model. The internal tool-loop is ephemeral — only the user/assistant
exchange is checkpointed, which keeps history small and token-cheap.

Hard limits (max_steps per turn) make runaway loops impossible.
"""

import json
import logging
import time

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.providers import ModelConfig, complete
from app.tools import (
    ExternalServers,
    open_catalog_session,
    set_current_agent,
    set_run_context,
)

logger = logging.getLogger(__name__)


class RunState(MessagesState):
    """`run_config` rides along per run and is not part of the history."""

    run_config: dict


# Prompt do resumo. Pede fatos, não prosa: o resumo volta para o modelo como
# contexto, e adjetivo ocupa lugar de número.
_RESUMO_PROMPT = (
    "Resuma a conversa abaixo preservando o que a continuação precisa saber: "
    "decisões tomadas, números e valores citados, nomes, arquivos mencionados, "
    "preferências do usuário e pendências. Escreva em tópicos curtos, em "
    "português. Não invente nada e não comente — só o resumo."
)


async def _resumir_trecho(mensagens: list, run_config: dict) -> str | None:
    """Condensa as mensagens antigas numa só.

    Falhar aqui não pode derrubar o turno: se o resumo não sai, o trecho é
    descartado como se a compressão estivesse desligada — pior contexto, mas a
    conversa continua.
    """
    from app.providers import ModelConfig, complete

    transcricao = []
    for m in mensagens:
        papel = "assistente" if m.type == "ai" else "usuário"
        conteudo = m.content if isinstance(m.content, str) else json.dumps(m.content)
        transcricao.append(f"{papel}: {conteudo[:2000]}")

    async def engolir(_):
        return None

    try:
        resultado = await complete(
            ModelConfig(**run_config["supervisor"]["model"]),
            [
                {"role": "system", "content": _RESUMO_PROMPT},
                {"role": "user", "content": "\n".join(transcricao)},
            ],
            engolir,
        )
    except Exception:  # noqa: BLE001 — resumo é melhoria, não requisito
        logger.exception("falha ao resumir histórico")
        return None
    texto = (resultado.content or "").strip()
    return texto or None


async def _historico_para_o_modelo(state: RunState, run_config: dict) -> list:
    """As mensagens que cabem neste turno.

    A conversa inteira ia para o modelo a cada turno. O custo crescia junto com
    ela e, longa o bastante, o turno falhava no limite do modelo em vez de
    degradar. O corte é do template porque a escolha afeta qualidade: cortar é
    barato e perde o começo; resumir preserva o sentido e custa uma chamada.
    """
    mensagens = list(state["messages"])
    limite = int(run_config.get("history_limit") or 100)
    if len(mensagens) <= limite:
        return [(None, m) for m in mensagens]

    # Metade recente intacta; o resto vira resumo (ou é descartado). O piso de 1
    # importa: com limite 1, `limite // 2` é 0 e `mensagens[-0:]` devolveria a
    # conversa inteira — o oposto do que o corte pede. Hoje a API não aceita
    # limite abaixo de 4, mas o kernel não deve depender disso.
    metade = max(1, limite // 2)
    recentes = mensagens[-metade:]
    antigas = mensagens[:-metade]

    if not run_config.get("compress_history"):
        return [(None, m) for m in recentes]

    resumo = await _resumir_trecho(antigas, run_config)
    if resumo is None:
        return [(None, m) for m in recentes]
    cabecalho = (
        f"[Resumo automático das {len(antigas)} mensagens anteriores desta "
        f"conversa]\n{resumo}"
    )
    return [("resumo", cabecalho)] + [(None, m) for m in recentes]


async def _history_messages(
    state: RunState, system_prompt: str, run_config: dict
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for marca, item in await _historico_para_o_modelo(state, run_config):
        if marca == "resumo":
            messages.append({"role": "user", "content": item})
            continue
        role = "assistant" if item.type == "ai" else "user"
        messages.append({"role": role, "content": item.content})
    return messages


def _descricao_do_especialista(agent: dict) -> str:
    """A descrição que o supervisor lê para decidir a quem perguntar.

    Anexar documentos a um especialista não mudava nada aqui, então o supervisor
    continuava sem saber que aquele especialista podia responder a partir deles
    — e dizia "não tenho acesso a documentos internos" com o documento indexado
    e ligado ao agente. Quem monta o template teria que lembrar de reescrever a
    descrição à mão; agora a informação vai junto.
    """
    descricao = agent.get("description") or ""
    if agent.get("file_ids"):
        quantos = len(agent["file_ids"])
        plural = "s" if quantos > 1 else ""
        descricao = (
            f"{descricao} Também responde a partir de {quantos} documento{plural} "
            "interno da empresa (contratos, políticas, manuais): consulte este "
            "especialista quando a pergunta citar documento, arquivo ou "
            "informação interna."
        ).strip()
    return descricao


def _agent_tool_defs(agents: list[dict]) -> list[dict]:
    """Each specialist becomes one callable tool for the supervisor."""
    return [
        {
            "type": "function",
            "function": {
                "name": agent["name"],
                "description": _descricao_do_especialista(agent),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "Tarefa ou pergunta, completa e autocontida, "
                                "para este especialista resolver."
                            ),
                        }
                    },
                    "required": ["task"],
                },
            },
        }
        for agent in agents
    ]


async def _record_usage(run_config: dict, agent_name: str, config: ModelConfig, result) -> None:
    from app.trace import insert_usage

    await insert_usage(
        tenant_id=run_config.get("tenant_id"),
        user_id=run_config.get("user_id"),
        chat_id=run_config.get("thread_id"),
        agent_name=agent_name,
        provider=config.provider,
        model=config.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )


async def _record_tool_call(
    run_config: dict, agent: str, tool: str, arguments: dict, output: str,
    status: str, started: float, writer,
) -> None:
    duration_ms = int((time.monotonic() - started) * 1000)
    writer(
        {
            "type": "tool_call",
            "agent": agent,
            "tool": tool,
            "status": status,
            "duration_ms": duration_ms,
        }
    )
    from app.trace import insert_tool_call

    await insert_tool_call(
        tenant_id=run_config.get("tenant_id"),
        chat_id=run_config.get("thread_id"),
        agent_name=agent,
        tool_name=tool,
        input=arguments,
        output=output[:10_000],
        status=status,
        duration_ms=duration_ms,
    )


def _find_artifact_descriptors(parsed) -> list[dict]:
    """Artifact descriptors at the top level, one nesting level down, or in
    an 'artifacts' list — covers every catalog tool's return shape."""
    if not isinstance(parsed, dict):
        return []
    if "artifact_id" in parsed:
        return [parsed]
    found = []
    for value in parsed.values():
        if isinstance(value, dict) and "artifact_id" in value:
            found.append(value)
        elif isinstance(value, list):
            found.extend(v for v in value if isinstance(v, dict) and "artifact_id" in v)
    return found


async def _execute_tool(
    name: str, arguments: dict, catalog_session, external: ExternalServers
) -> str:
    if name in external.tools:
        return await external.call(name, arguments)
    result = await catalog_session.call_tool(name, arguments)
    parts = [c.text for c in result.content if getattr(c, "text", None)]
    return "\n".join(parts) or "(sem conteúdo)"


def _estourou_o_teto(run_config: dict) -> bool:
    """Conta as chamadas de ferramenta do turno inteiro e diz se passou.

    O contador mora no `run_config` porque ele é o único objeto que atravessa
    todos os especialistas de um mesmo turno — `specialist_max_tool_rounds`
    limita um especialista de cada vez, e o supervisor pode chamar vários.
    """
    teto = int(run_config.get("max_tool_calls_per_turn") or settings.max_tool_calls_per_turn)
    usadas = run_config.get("_tool_calls_no_turno", 0)
    if usadas >= teto:
        return True
    run_config["_tool_calls_no_turno"] = usadas + 1
    return False


def _limitar_saida(tool_output: str, descriptors: list[dict], limite: int) -> str:
    """Corta o que a ferramenta devolve antes de virar prompt do especialista.

    O resultado inteiro de uma consulta pode ter dezenas de milhares de linhas,
    e ele entra no contexto a cada rodada seguinte do especialista. Cortar não
    perde o dado: o payload continua materializado no artefato, e o artifact_id
    encadeia para gráfico, planilha e sandbox sem passar pelo modelo.

    O aviso é parte do corte. Truncar em silêncio faz o modelo somar uma coluna
    pela metade e apresentar o número como total.
    """
    if limite <= 0 or len(tool_output) <= limite:
        return tool_output
    ids = ", ".join(d["artifact_id"] for d in descriptors)
    onde = f" O resultado completo está no artefato {ids}." if ids else ""
    return (
        tool_output[:limite]
        + f"\n\n[CORTADO PELA PLATAFORMA: a saída tinha {len(tool_output)} caracteres "
        f"e só os primeiros {limite} chegam até você.{onde} NÃO some nem conte nada a "
        "partir do trecho acima — ele está incompleto. Para números sobre o conjunto "
        "todo, peça à consulta que agregue (SUM/COUNT/GROUP BY) ou encadeie o "
        "artefato em outra ferramenta.]"
    )


def _nota_dos_artefatos(descriptors: list[dict]) -> str:
    """Conta ao supervisor o que já apareceu na tela do usuário.

    O artefato é emitido como evento de stream direto ao cliente, então a
    imagem/planilha já está visível — mas quem escreve a resposta final é o
    supervisor, e ele só recebe o TEXTO do especialista. Sem esta nota ele
    respondia "não consigo exibir imagens" com o QR Code renderizado logo
    acima, no mesmo turno.
    """
    if not descriptors:
        return ""
    itens = "; ".join(
        f"{d.get('kind', 'arquivo')} \"{d.get('title', '')}\"" for d in descriptors
    )
    return (
        f"\n\n[NOTA DA PLATAFORMA] Já foram exibidos ao usuário nesta resposta: "
        f"{itens}. Ele está vendo isso na tela agora. "
        "NUNCA diga que não consegue mostrar imagens, gráficos ou arquivos: "
        "aponte o que está logo acima."
    )


CLAUSULA_DO_ESPECIALISTA = (
    "\n\nCOMO VOCÊ É CHAMADO: você não conversa com o usuário. Quem fala com ele "
    "é o supervisor, e você só recebe a tarefa já decidida por ele — você NÃO vê "
    "a conversa nem as respostas anteriores do usuário. Portanto NUNCA peça "
    "confirmação, esclarecimento ou dados ao usuário: perguntar aqui é o mesmo "
    "que não fazer nada, porque a pergunta volta para o supervisor e a próxima "
    "chamada chega igualmente sem histórico. Se a tarefa está clara, execute-a e "
    "devolva o resultado; a confirmação do usuário, quando é necessária, já foi "
    "colhida pelo supervisor antes de chamar você. Se algo indispensável faltar "
    "na tarefa, diga ao supervisor exatamente o que falta — não pergunte ao "
    "usuário."
)


async def _run_specialist(
    agent: dict, task: str, writer, run_config: dict, tool_defs: dict,
    catalog_session, external: ExternalServers,
) -> str:
    """One specialist turn: its own model, its own tools, a short bounded
    tool-loop. Output goes back to the supervisor, not to the user stream."""
    writer({"type": "agent_start", "name": agent["name"]})
    set_current_agent(agent["name"], agent.get("model"))
    config = ModelConfig(**agent["model"])
    limite_saida = int(
        run_config.get("tool_output_limit") or settings.tool_output_limit_default
    )
    allowed_names = {t for t in agent.get("tools", []) if t in tool_defs}
    allowed = [tool_defs[t] for t in allowed_names] or None
    messages = [
        {"role": "system", "content": agent["prompt"] + CLAUSULA_DO_ESPECIALISTA},
        {"role": "user", "content": task},
    ]

    async def swallow(_delta: str):
        return None

    output = ""
    exibidos: list[dict] = []
    exibidos_antes = 0
    try:
        for _round in range(settings.specialist_max_tool_rounds):
            result = await complete(config, messages, swallow, tools=allowed)
            await _record_usage(run_config, agent["name"], config, result)
            if not result.tool_calls:
                output = result.content or "(especialista não retornou conteúdo)"
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in result.tool_calls
                    ],
                }
            )
            for tc in result.tool_calls:
                try:
                    arguments = json.loads(tc.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": tc.arguments}
                started = time.monotonic()
                if _estourou_o_teto(run_config):
                    # O modelo precisa SABER que bateu no teto, senão tenta a
                    # mesma chamada de novo até esgotar as rodadas. Devolver
                    # como erro de ferramenta é o único canal que ele lê.
                    tool_output = (
                        "ERRO: teto de chamadas de ferramenta deste turno atingido. "
                        "Responda agora com o que você já tem e diga o que ficou sem "
                        "verificar — não tente outra consulta."
                    )
                    status = "error"
                    writer({"type": "limit", "detail": "max_tool_calls_per_turn"})
                elif tc.name not in allowed_names:
                    # A model may hallucinate a tool it was never offered;
                    # refuse without executing.
                    tool_output = f"ERRO: tool {tc.name} não disponível para este agente"
                    status = "error"
                else:
                    try:
                        tool_output = await _execute_tool(
                            tc.name, arguments, catalog_session, external
                        )
                        status = "ok"
                    except Exception as exc:  # noqa: BLE001 — reported into the loop
                        tool_output = f"ERRO na tool {tc.name}: {exc}"
                        status = "error"
                await _record_tool_call(
                    run_config, agent["name"], tc.name, arguments, tool_output,
                    status, started, writer,
                )
                # Materialized artifacts surface as their own stream event so
                # the frontend can render/download them. Tools may return one
                # descriptor or nest several (forecast, sandbox).
                descriptors: list[dict] = []
                if status == "ok" and '"artifact_id"' in tool_output:
                    try:
                        parsed = json.loads(tool_output)
                    except json.JSONDecodeError:
                        parsed = None
                    descriptors = _find_artifact_descriptors(parsed)
                    for descriptor in descriptors:
                        exibidos.append(descriptor)
                        writer(
                            {
                                "type": "artifact",
                                "artifact_id": descriptor["artifact_id"],
                                "kind": descriptor.get("kind", ""),
                                "title": descriptor.get("title", ""),
                            }
                        )
                # O corte vem DEPOIS de publicar o artefato e de registrar a
                # chamada: o que se limita é o que entra no prompt, não o que se
                # guarda nem o que o usuário recebe.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _limitar_saida(tool_output, descriptors, limite_saida),
                    }
                )
            # O aviso precisa entrar aqui, e não só no retorno ao supervisor: quem
            # redige primeiro é o especialista, e era ele quem escrevia "não
            # consigo exibir imagem" com o QR Code já na tela. Corrigir só o
            # supervisor deixava a negativa nascer e depois ser desmentida.
            if len(exibidos) > exibidos_antes:
                messages.append(
                    {"role": "user", "content": _nota_dos_artefatos(exibidos[exibidos_antes:])}
                )
                exibidos_antes = len(exibidos)
        else:
            result = await complete(
                config,
                messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Responda agora com o que você tem. Se alguma operação "
                            "não foi executada, diga claramente que NÃO foi feita — "
                            "nunca afirme sucesso de algo que você não executou."
                        ),
                    }
                ],
                swallow,
            )
            await _record_usage(run_config, agent["name"], config, result)
            output = result.content or "(sem resposta)"
    except Exception as exc:  # noqa: BLE001 — reported into the loop, not fatal
        logger.exception("specialist %s failed", agent["name"])
        output = f"ERRO no especialista {agent['name']}: {exc}"
    writer({"type": "agent_done", "name": agent["name"]})
    return output + _nota_dos_artefatos(exibidos)


async def _load_tool_defs(catalog_session, external: ExternalServers) -> dict:
    """All available tools as OpenAI-style function defs, keyed by name."""
    defs: dict[str, dict] = {}
    listed = await catalog_session.list_tools()
    for tool in listed.tools:
        defs[tool.name] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
    for public, (_server, _tool, schema) in external.tools.items():
        defs[public] = {
            "type": "function",
            "function": {
                "name": public,
                "description": f"Tool externa {public}",
                "parameters": schema or {"type": "object", "properties": {}},
            },
        }
    return defs


WRITE_CONFIRMATION_CLAUSE = (
    "\n\nREGRA DE ESCRITA: antes de qualquer operação que altere dados "
    "(criar pedido, efetuar venda, atualizar registro), apresente ao usuário um "
    "resumo claro da operação e SÓ execute depois que ele confirmar "
    "explicitamente na conversa (ex.: 'sim', 'confirmo'). Se a confirmação ainda "
    "não veio nesta conversa, pergunte e aguarde. Depois que ela vier, execute: "
    "não pergunte de novo. O especialista não vê a conversa, então ao delegar "
    "escreva na tarefa que o usuário JÁ confirmou, com os valores confirmados — "
    "sem isso ele pede confirmação que nunca chega ao usuário e nada é feito. "
    "Para criar um registro com filhos (um pedido e seus itens), use SEMPRE a "
    "tool execute_sql_transaction com TODOS os statements de uma vez — o pedido "
    "com RETURNING id e os itens referenciando {{returned:0}} — para que seja "
    "atômico. NUNCA crie o pedido e os itens em chamadas separadas e NUNCA repita "
    "um INSERT: se uma tool retornar status ok, a operação já foi gravada. Após "
    "executar, relate o resultado REAL retornado pela tool (affected_rows); "
    "NUNCA afirme que uma operação foi concluída sem tê-la executado, e NUNCA "
    "afirme falha se a tool retornou status ok."
)


def build_supervisor_prompt(
    base_prompt: str, memories: list[str], require_write_confirmation: bool
) -> str:
    prompt = base_prompt
    if memories:
        prompt += "\n\nO que você sabe sobre este usuário (memórias):\n" + "\n".join(
            f"- {m}" for m in memories
        )
    if require_write_confirmation:
        prompt += WRITE_CONFIRMATION_CLAUSE
    return prompt


async def _supervisor_node(state: RunState) -> dict:
    run_config = state["run_config"]
    supervisor = run_config["supervisor"]
    agents = {a["name"]: a for a in run_config.get("agents", [])}
    max_steps = int(run_config.get("max_steps", settings.max_steps_default))
    writer = get_stream_writer()

    set_run_context(
        secrets=run_config.get("secrets", {}),
        datasources=run_config.get("datasources", []),
        tenant_id=run_config.get("tenant_id"),
        chat_id=run_config.get("thread_id"),
        embedding=run_config.get("embedding") or None,
        agent_files={
            a["name"]: a.get("file_ids", []) for a in run_config.get("agents", [])
        },
        write_tables=run_config.get("write_tables", []),
        attachments=run_config.get("attachments", []),
        payment=run_config.get("payment") or {},
    )
    tool_defs = _agent_tool_defs(list(agents.values())) or None
    supervisor_config = ModelConfig(**supervisor["model"])
    system_prompt = build_supervisor_prompt(
        supervisor.get("prompt", ""),
        run_config.get("memories") or [],
        bool(run_config.get("require_write_confirmation"))
        and bool(run_config.get("write_tables")),
    )
    messages = await _history_messages(state, system_prompt, run_config)

    async def emit(delta: str):
        writer({"type": "token", "text": delta})

    final_text = ""
    async with (
        open_catalog_session() as catalog_session,
        ExternalServers(run_config.get("mcp_servers", [])) as external,
    ):
        platform_tools = await _load_tool_defs(catalog_session, external)

        for _step in range(max_steps):
            result = await complete(supervisor_config, messages, emit, tools=tool_defs)
            await _record_usage(run_config, "supervisor", supervisor_config, result)
            if not result.tool_calls:
                final_text = result.content
                break

            # Record the assistant turn that requested the calls, then answer them.
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in result.tool_calls
                    ],
                }
            )
            for tc in result.tool_calls:
                agent = agents.get(tc.name)
                if agent is None:
                    output = f"ERRO: especialista '{tc.name}' não existe."
                else:
                    try:
                        task = json.loads(tc.arguments or "{}").get("task", "")
                    except json.JSONDecodeError:
                        task = tc.arguments
                    output = await _run_specialist(
                        agent, task, writer, run_config, platform_tools,
                        catalog_session, external,
                    )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )
        else:
            # Step budget exhausted: force a final answer, no tools allowed.
            writer({"type": "limit", "detail": "max_steps"})
            result = await complete(
                supervisor_config,
                messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Limite de passos atingido. Responda agora ao usuário "
                            "com o que você já tem."
                        ),
                    }
                ],
                emit,
            )
            await _record_usage(run_config, "supervisor", supervisor_config, result)
            final_text = result.content

    return {"messages": [{"role": "assistant", "content": final_text}]}


def build_graph(checkpointer) -> object:
    graph = StateGraph(RunState)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile(checkpointer=checkpointer)


_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_graph = None


async def get_graph():
    """Lazily build the singleton graph with its Postgres checkpointer."""
    global _pool, _checkpointer, _graph
    if _graph is None:
        _pool = AsyncConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=settings.checkpoint_pool_size,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            # `check` descarta conexão morta antes de entregá-la: o banco fica
            # atrás do Traefik da VPS, e um restart lá deixa conexões ociosas
            # em estado zumbi que só falham na hora do uso.
            check=AsyncConnectionPool.check_connection,
            open=False,
        )
        await _pool.open()
        _checkpointer = AsyncPostgresSaver(_pool)
        await _checkpointer.setup()
        _graph = build_graph(_checkpointer)
    return _graph


async def close_graph() -> None:
    global _pool, _checkpointer, _graph
    if _pool is not None:
        await _pool.close()
    _pool = _checkpointer = _graph = None
