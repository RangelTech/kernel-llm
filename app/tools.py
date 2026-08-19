"""Platform tool catalog — a formal in-process MCP server.

Tools are defined once here (FastMCP); the runtime consumes them through a
real MCP client over the in-memory transport, so the contract is pure MCP and
the server can be split out later without touching callers. External MCP
servers declared on a template are reached with the streamable-HTTP client
and their tools are namespaced ext_<server>_<tool>.
"""

import base64
import json
import logging
import re
from contextlib import AsyncExitStack
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation

import httpx
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

logger = logging.getLogger(__name__)

catalog = FastMCP("agent-platform-tools")

_SECRET_REF = re.compile(r"\{\{secret:([A-Za-z0-9_.-]+)\}\}")

# Per-run execution context. A ContextVar (not a module global) because the
# kernel serves concurrent runs in one process — a global would leak one
# tenant's secrets/datasources into another tenant's run.
RUN_CONTEXT: ContextVar[dict] = ContextVar("run_context")


def _context() -> dict:
    try:
        return RUN_CONTEXT.get()
    except LookupError:
        return {}


def set_run_context(
    *,
    secrets: dict[str, str],
    datasources: list[dict],
    tenant_id,
    chat_id,
    embedding: dict | None = None,
    agent_files: dict[str, list[str]] | None = None,
    write_tables: list[str] | None = None,
    attachments: list[dict] | None = None,
    payment: dict | None = None,
) -> None:
    RUN_CONTEXT.set(
        {
            "attachments": attachments or [],
            "secrets": secrets,
            "datasources": {d["name"]: d for d in datasources},
            "tenant_id": tenant_id,
            "chat_id": chat_id,
            "agent": "",
            "embedding": embedding or {"provider": "stub"},
            "agent_files": agent_files or {},
            "write_tables": write_tables or [],
            "payment": payment or {},
            # Leituras já feitas neste turno. Ver `_cache_de_leitura`.
            "leituras": {},
        }
    )


def set_current_agent(agent_name: str, agent_model: dict | None = None) -> None:
    # Mutate the shared dict IN PLACE: the in-memory MCP server task snapshots
    # the ContextVar when the session opens, so a later .set() would be
    # invisible to tool handlers — the shared object reference is not.
    context = _context()
    context["agent"] = agent_name
    if agent_model is not None:
        context["agent_model"] = agent_model


def _resolve_secrets(text: str) -> str:
    secrets = _context().get("secrets", {})

    def sub(match: re.Match) -> str:
        return secrets.get(match.group(1), match.group(0))

    return _SECRET_REF.sub(sub, text)


@catalog.tool()
def calculate(expression: str) -> str:
    """Calculadora determinística para aritmética exata. Use SEMPRE que
    precisar de contas (somas, porcentagens, juros). Aceita + - * / % ** e
    parênteses. Exemplo: (1500 * 1.05) - 200"""
    cleaned = expression.replace(",", ".").replace(" ", "")
    if not re.fullmatch(r"[0-9.+\-*/%()eE]+", cleaned):
        return "ERRO: expressão contém caracteres não permitidos"
    try:
        # Decimal-friendly eval without names/builtins; the regex above blocks
        # identifiers so this cannot reach arbitrary code.
        result = eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307
        if isinstance(result, float):
            result = Decimal(str(result)).normalize()
        return str(result)
    except (SyntaxError, ZeroDivisionError, InvalidOperation, ArithmeticError) as exc:
        return f"ERRO: {exc}"


@catalog.tool()
async def call_http_api(
    url: str,
    method: str = "GET",
    headers_json: str = "{}",
    body_json: str = "",
    timeout_seconds: int = 30,
) -> str:
    """Chama uma API HTTP externa e retorna o corpo da resposta. Referências
    {{secret:NOME}} em url, headers ou body são substituídas por segredos da
    empresa sem passar pelo modelo. headers_json e body_json são strings JSON."""
    url = _resolve_secrets(url)
    try:
        headers = json.loads(_resolve_secrets(headers_json) or "{}")
    except json.JSONDecodeError:
        return "ERRO: headers_json não é JSON válido"
    body = _resolve_secrets(body_json) if body_json else None

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        return f"ERRO: método {method} não suportado"
    try:
        async with httpx.AsyncClient(timeout=min(timeout_seconds, 60)) as client:
            response = await client.request(method, url, headers=headers, content=body)
        text = response.text[:20_000]
        return f"HTTP {response.status_code}\n{text}"
    except httpx.HTTPError as exc:
        return f"ERRO: {exc}"


def _cache_de_leitura() -> dict:
    """Consultas de leitura já executadas neste turno.

    Observado em conversa real: um especialista rodou 45 consultas num único
    turno, várias delas byte a byte idênticas, porque nada no retorno indicava
    que aquilo já tinha sido perguntado. Cada repetição custava ~900 ms e uma
    cobrança do BigQuery, e o turno levou nove minutos.

    O cache vive no contexto do turno, então não atravessa conversas e não
    guarda dado de um tenant onde outro possa ver.
    """
    contexto = _context()
    if "leituras" not in contexto:
        contexto["leituras"] = {}
    return contexto["leituras"]


def _chave_de_leitura(datasource: str, query: str) -> str:
    return f"{datasource}\n{' '.join(query.split())}"


def _invalidar_leituras() -> None:
    """Uma escrita torna qualquer leitura anterior suspeita."""
    _cache_de_leitura().clear()


@catalog.tool()
async def describe_datasources() -> str:
    """Lista as fontes de dados disponíveis nesta conversa, com suas tabelas e
    colunas. Chame antes de escrever SQL para saber o que existe."""
    context = _context()
    datasources = context.get("datasources", {})
    if not datasources:
        return "Nenhuma fonte de dados vinculada a este template."

    cache = _cache_de_leitura()
    if "__catalogo__" in cache:
        return cache["__catalogo__"]

    from app.datasources import list_tables

    output = []
    for name, datasource in datasources.items():
        entry = {"datasource": name, "kind": datasource["kind"]}
        try:
            entry["tables"] = await list_tables(datasource)
        except Exception as exc:  # noqa: BLE001 — reported to the model
            entry["error"] = str(exc)[:300]
        output.append(entry)
    catalogo = json.dumps(output, ensure_ascii=False, default=str)
    cache["__catalogo__"] = catalogo
    return catalogo


@catalog.tool()
async def run_sql_query(datasource: str, query: str, title: str = "") -> str:
    """Executa uma consulta SQL de LEITURA (SELECT/WITH) na fonte de dados e
    materializa o resultado como um dataset artifact. Retorna o artifact_id,
    o schema e uma amostra das primeiras linhas — use o artifact_id para
    encadear com outras tools (gráfico, planilha) sem reexecutar a consulta.
    `datasource` é o nome da fonte (veja describe_datasources)."""
    from app.config import settings
    from app.datasources import execute_query
    from app.storage import register_artifact

    context = _context()
    source = context.get("datasources", {}).get(datasource)
    if source is None:
        available = ", ".join(context.get("datasources", {})) or "(nenhuma)"
        return f"ERRO: fonte '{datasource}' não existe. Disponíveis: {available}"
    cache = _cache_de_leitura()
    chave = _chave_de_leitura(datasource, query)
    if chave in cache:
        # Devolver só o resultado faria o modelo repetir de novo. O aviso é o
        # que quebra o laço.
        return (
            "AVISO: esta consulta já foi executada neste turno; resultado abaixo "
            "reaproveitado. Se ele não responde à pergunta, mude a consulta em vez "
            "de repeti-la.\n\n" + cache[chave]
        )

    try:
        columns, rows = await execute_query(source, query, settings.sql_max_rows)
    except Exception as exc:  # noqa: BLE001 — the model needs the error to retry
        return f"ERRO na consulta: {exc}"

    preview = rows[: settings.artifact_preview_rows]
    descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="dataset",
        title=title or f"Consulta em {datasource}",
        schema_json=columns,
        preview_json=preview,
        row_count=len(rows),
        payload=json.dumps(
            {"columns": columns, "rows": rows}, ensure_ascii=False, default=str
        ).encode(),
    )
    resultado = json.dumps(descriptor, ensure_ascii=False, default=str)
    cache[chave] = resultado
    return resultado


@catalog.tool()
async def query_mongo(
    datasource: str,
    collection: str,
    filter_json: str = "{}",
    projection_json: str = "",
    limit: int = 100,
    title: str = "",
) -> str:
    """Consulta uma fonte de dados MongoDB (NÃO use run_sql_query para Mongo —
    não é SQL). Sempre LEITURA: executa um `find` com filtro e, opcionalmente,
    projeção de campos; nunca grava, atualiza nem apaga. `filter_json` e
    `projection_json` são strings JSON no formato do MongoDB (ex.:
    filter_json='{"status": "ativo"}', projection_json='{"nome": 1, "_id": 0}').
    `collection` é o nome da coleção (veja describe_datasources). Retorna um
    dataset artifact com os documentos encontrados."""
    from app.config import settings
    from app.datasources import query_mongo as _run_query_mongo
    from app.storage import register_artifact

    context = _context()
    source = context.get("datasources", {}).get(datasource)
    if source is None:
        available = ", ".join(context.get("datasources", {})) or "(nenhuma)"
        return f"ERRO: fonte '{datasource}' não existe. Disponíveis: {available}"
    if source.get("kind") != "mongodb":
        return f"ERRO: '{datasource}' não é uma fonte MongoDB (kind={source.get('kind')})"

    try:
        filtro = json.loads(filter_json or "{}")
        if not isinstance(filtro, dict):
            return "ERRO: filter_json deve ser um objeto JSON"
        projecao = json.loads(projection_json) if projection_json else None
        if projecao is not None and not isinstance(projecao, dict):
            return "ERRO: projection_json deve ser um objeto JSON"
    except json.JSONDecodeError as exc:
        return f"ERRO: JSON inválido em filter_json/projection_json: {exc}"

    max_rows = min(int(limit) if limit else settings.sql_max_rows, settings.sql_max_rows)
    cache = _cache_de_leitura()
    chave = _chave_de_leitura(
        datasource, f"{collection}\n{json.dumps(filtro, sort_keys=True)}\n{max_rows}"
    )
    if chave in cache:
        return (
            "AVISO: esta consulta já foi executada neste turno; resultado abaixo "
            "reaproveitado. Se ele não responde à pergunta, mude a consulta em vez "
            "de repeti-la.\n\n" + cache[chave]
        )

    try:
        columns, rows = await _run_query_mongo(source, collection, filtro, projecao, max_rows)
    except Exception as exc:  # noqa: BLE001 — the model needs the error to retry
        return f"ERRO na consulta: {exc}"

    preview = rows[: settings.artifact_preview_rows]
    descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="dataset",
        title=title or f"Consulta em {datasource}.{collection}",
        schema_json=columns,
        preview_json=preview,
        row_count=len(rows),
        payload=json.dumps(
            {"columns": columns, "rows": rows}, ensure_ascii=False, default=str
        ).encode(),
    )
    resultado = json.dumps(descriptor, ensure_ascii=False, default=str)
    cache[chave] = resultado
    return resultado


async def _load_dataset_for(context: dict, artifact_id: str) -> dict | None:
    from app.storage import get_artifact, load_payload

    record = await get_artifact(artifact_id)
    if record is None or record["kind"] != "dataset":
        return None
    if record["tenant_id"] and context.get("tenant_id") and str(record["tenant_id"]) != str(
        context["tenant_id"]
    ):
        return None
    return json.loads(load_payload(record["storage_path"]))


# Bibliotecas de gráfico: usá-las aqui não gera nada que o usuário veja.
_BIBLIOTECAS_DE_GRAFICO = ("matplotlib", "pyplot", "plotly", "seaborn", "plt.")


def _plotou_sem_publicar(code: str, outputs: list[dict]) -> bool:
    """O código desenhou um gráfico e não publicou dataset nenhum?

    Só avisa quando não há dataset publicado: quem plota *e* publica os dados
    ainda pode chamar generate_chart depois, e nesse caso o aviso seria ruído.
    """
    if any(o.get("kind") == "dataset" for o in outputs):
        return False
    return any(termo in code for termo in _BIBLIOTECAS_DE_GRAFICO)


@catalog.tool()
async def execute_python(code: str, artifact_id: str = "") -> str:
    """Executa código Python em sandbox isolado (sem rede, com timeout).
    Se artifact_id de um dataset for passado, o código recebe `columns`,
    `rows` e `df` (pandas). Use publish_dataset(titulo, columns, rows) e
    publish_document(titulo, texto) para materializar resultados. stdout é
    retornado — use print() para responder valores.
    NÃO desenhe gráfico aqui: matplotlib/plotly dentro do sandbox não produzem
    nada visível para o usuário. Para gráfico, publique um dataset e depois
    chame generate_chart com o artifact_id devolvido."""
    from app.sandbox import run_sandboxed
    from app.storage import register_artifact

    context = _context()
    dataset = None
    if artifact_id:
        dataset = await _load_dataset_for(context, artifact_id)
        if dataset is None:
            return f"ERRO: artifact {artifact_id} não é um dataset acessível"

    stdout, stderr, outputs, timed_out = await run_sandboxed(code, dataset)
    if timed_out:
        return "ERRO: tempo limite excedido — o código foi interrompido"

    # O sandbox só publica dataset e documento. Um gráfico desenhado aqui não
    # chega a lugar nenhum — e o modelo, sem saber disso, anuncia ao usuário um
    # gráfico que não existe. Melhor falar antes que ele responda.
    aviso_grafico = ""
    if _plotou_sem_publicar(code, outputs):
        aviso_grafico = (
            "AVISO: código de gráfico detectado, mas o sandbox não produz imagem. "
            "Nada foi mostrado ao usuário. Publique os dados com publish_dataset e "
            "chame generate_chart com o artifact_id para o gráfico aparecer.\n\n"
        )

    published = []
    for output in outputs[:5]:
        if output.get("kind") == "dataset":
            # publish_dataset() is called by model-generated code, which
            # commonly passes a bare list of column-name strings (e.g. from
            # df.columns.tolist()) instead of the platform's canonical
            # [{"name": ..., "type": ...}] shape. Normalize here so every
            # downstream consumer (generate_chart, export_xlsx, further
            # execute_python calls) can rely on a single consistent format
            # instead of crashing with "string indices must be integers".
            raw_columns = output.get("columns", []) or []
            normalized_columns = [
                {"name": col} if isinstance(col, str) else col for col in raw_columns
            ]
            descriptor = await register_artifact(
                tenant_id=context.get("tenant_id"),
                chat_id=context.get("chat_id"),
                agent_name=context.get("agent", ""),
                kind="dataset",
                title=output.get("title", "Dataset do sandbox"),
                schema_json=normalized_columns,
                preview_json=(output.get("rows") or [])[:10],
                row_count=len(output.get("rows") or []),
                payload=json.dumps(
                    {"columns": normalized_columns, "rows": output.get("rows", [])},
                    ensure_ascii=False,
                    default=str,
                ).encode(),
            )
        else:
            descriptor = await register_artifact(
                tenant_id=context.get("tenant_id"),
                chat_id=context.get("chat_id"),
                agent_name=context.get("agent", ""),
                kind="file",
                title=(output.get("title", "documento") or "documento") + ".txt",
                schema_json=None,
                preview_json=None,
                row_count=None,
                payload=str(output.get("text", "")).encode(),
                content_type="text/plain",
                extension="txt",
            )
        published.append(descriptor)

    return aviso_grafico + json.dumps(
        {"stdout": stdout, "stderr": stderr, "artifacts": published},
        ensure_ascii=False,
        default=str,
    )


@catalog.tool()
async def generate_forecast(
    artifact_id: str,
    date_column: str,
    value_column: str,
    horizon: int = 6,
    freq: str = "MS",
) -> str:
    """Gera previsão temporal a partir de um dataset artifact (coluna de data
    + coluna numérica). freq: MS=mensal, W=semanal, D=diário, QS=trimestral.
    Publica o dataset de projeção e um gráfico com histórico + previsão."""
    from app.forecast import forecast_series
    from app.storage import register_artifact

    context = _context()
    dataset = await _load_dataset_for(context, artifact_id)
    if dataset is None:
        return f"ERRO: artifact {artifact_id} não é um dataset acessível"

    try:
        history, future = await forecast_series(
            dataset["columns"], dataset["rows"], date_column, value_column,
            horizon=min(horizon, 36), freq=freq,
        )
    except Exception as exc:  # noqa: BLE001 — the model needs the reason
        return f"ERRO na previsão: {exc}"

    forecast_descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="dataset",
        title=f"Previsão de {value_column} ({horizon} períodos)",
        schema_json=[{"name": "data", "type": "date"}, {"name": "previsao", "type": "number"}],
        preview_json=future[:10],
        row_count=len(future),
        payload=json.dumps(
            {
                "columns": [{"name": "data"}, {"name": "previsao"}],
                "rows": future,
            },
            ensure_ascii=False,
        ).encode(),
    )

    figure = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "histórico",
                "x": [p[0] for p in history],
                "y": [p[1] for p in history],
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "previsão",
                "line": {"dash": "dash"},
                "x": [p[0] for p in future],
                "y": [p[1] for p in future],
            },
        ],
        "layout": {"title": f"Previsão de {value_column}"},
    }
    chart_descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="chart",
        title=f"Previsão de {value_column}",
        schema_json=None,
        preview_json=None,
        row_count=None,
        payload=json.dumps(figure, ensure_ascii=False).encode(),
    )
    return json.dumps(
        {
            "forecast": forecast_descriptor,
            "chart": chart_descriptor,
            "future_preview": future[: min(6, len(future))],
        },
        ensure_ascii=False,
        default=str,
    )


@catalog.tool()
async def web_search(query: str) -> str:
    """Busca na web e retorna resultados com título, URL e resumo. Use para
    informações atuais (notícias, preços de mercado, fatos recentes)."""
    from app.config import settings

    results: list[dict] = []
    if settings.serper_api_key:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    settings.serper_url,
                    json={"q": query, "num": settings.web_search_max_results},
                    headers={"X-API-KEY": settings.serper_api_key},
                )
                response.raise_for_status()
                for item in response.json().get("organic", [])[: settings.web_search_max_results]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                        }
                    )
        except Exception:  # noqa: BLE001 — fall through to DuckDuckGo
            results = []

    if not results:
        # Fallback: DuckDuckGo via ddgs. Documented risk: scraping-based,
        # may break without notice.
        try:
            from ddgs import DDGS

            def _ddg():
                with DDGS() as ddg:
                    return list(ddg.text(query, max_results=settings.web_search_max_results))

            import anyio as _anyio

            for item in await _anyio.to_thread.run_sync(_ddg):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            return f"ERRO: busca indisponível (Serper e fallback falharam): {exc}"

    if not results:
        return "Nenhum resultado encontrado."
    return json.dumps(results, ensure_ascii=False)


@catalog.tool()
async def analyze_pdf_pages(attachment_name: str, instruction: str, max_pages: int = 10) -> str:
    """Analisa um PDF anexado na conversa página a página usando visão de IA.
    Passe o nome do anexo e a instrução de extração (ex.: 'liste os produtos
    circulados e as quantidades escritas à mão'). Retorna JSON por página."""
    from app.config import settings as kernel_settings
    from app.providers import ModelConfig, complete
    from app.storage import load_payload

    context = _context()
    attachment = next(
        (a for a in context.get("attachments", []) if a.get("name") == attachment_name),
        None,
    )
    if attachment is None:
        names = ", ".join(a.get("name", "?") for a in context.get("attachments", []))
        return f"ERRO: anexo '{attachment_name}' não encontrado. Anexos: {names or '(nenhum)'}"

    try:
        data = load_payload(attachment["storage_path"])
    except Exception as exc:  # noqa: BLE001
        return f"ERRO ao carregar o PDF: {exc}"

    import base64

    import fitz  # pymupdf

    model = context.get("agent_model") or {"provider": "stub", "model": "stub-1"}
    config = ModelConfig(**model)
    limit = min(max_pages, kernel_settings.pdf_vision_max_pages)

    async def swallow(_delta: str):
        return None

    pages_output: list[dict] = []
    with fitz.open(stream=data, filetype="pdf") as document:
        total = len(document)
        for index in range(min(total, limit)):
            pixmap = document[index].get_pixmap(dpi=110)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode()
            content = [
                {"type": "text", "text": f"[página {index + 1}] {instruction}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
            try:
                result = await complete(
                    config, [{"role": "user", "content": content}], swallow
                )
                pages_output.append({"page": index + 1, "result": result.content})
            except Exception as exc:  # noqa: BLE001
                pages_output.append({"page": index + 1, "error": str(exc)[:300]})

    return json.dumps(
        {"pages_analyzed": len(pages_output), "total_pages": total, "pages": pages_output},
        ensure_ascii=False,
    )


@catalog.tool()
async def execute_sql_write(datasource: str, statement: str) -> str:
    """Executa UMA escrita SQL (INSERT, UPDATE ou DELETE) na fonte de dados,
    somente em tabelas autorizadas pelo template. UPDATE/DELETE exigem WHERE.
    Use apenas após o usuário confirmar a operação quando o template exigir
    confirmação. Use INSERT ... RETURNING id para obter o id gerado (campo
    'returned') e encadear inserts pai->filho. NUNCA repita um INSERT já
    executado com sucesso. Retorna tabela, linhas afetadas e RETURNING."""
    context = _context()
    source = context.get("datasources", {}).get(datasource)
    if source is None:
        available = ", ".join(context.get("datasources", {})) or "(nenhuma)"
        return f"ERRO: fonte '{datasource}' não existe. Disponíveis: {available}"

    from app.datasources import execute_write

    try:
        table, rows, returned = await execute_write(
            source, statement, context.get("write_tables", [])
        )
    except Exception as exc:  # noqa: BLE001 — the model needs the reason
        return f"ERRO na escrita: {exc}"
    _invalidar_leituras()
    return json.dumps(
        {"status": "ok", "table": table, "affected_rows": rows, "returned": returned},
        ensure_ascii=False,
        default=str,
    )


@catalog.tool()
async def execute_sql_transaction(datasource: str, statements_json: str) -> str:
    """Executa VÁRIAS escritas SQL numa ÚNICA transação atômica (tudo ou nada).
    Use SEMPRE que criar um registro com filhos — por exemplo um pedido e seus
    itens — para não duplicar nem deixar dados pela metade. `statements_json` é
    um array JSON de statements na ordem de execução. Para usar o id gerado por
    um INSERT anterior, termine-o com RETURNING id e referencie com
    {{returned:0}} (0 = primeiro statement). Exemplo:
    ["INSERT INTO pedidos (cliente_id, total) VALUES (1, 100) RETURNING id",
     "INSERT INTO itens_pedido (pedido_id, produto_id, quantidade) VALUES ({{returned:0}}, 5, 2)"].
    Só tabelas autorizadas; UPDATE/DELETE exigem WHERE. Use após a confirmação
    do usuário quando o template exigir. Se falhar, NADA é gravado."""
    context = _context()
    source = context.get("datasources", {}).get(datasource)
    if source is None:
        available = ", ".join(context.get("datasources", {})) or "(nenhuma)"
        return f"ERRO: fonte '{datasource}' não existe. Disponíveis: {available}"
    try:
        statements = json.loads(statements_json)
        if not isinstance(statements, list) or not all(isinstance(s, str) for s in statements):
            return "ERRO: statements_json deve ser um array JSON de strings SQL"
    except json.JSONDecodeError as exc:
        return f"ERRO: statements_json inválido: {exc}"

    from app.datasources import execute_transaction

    try:
        results = await execute_transaction(
            source, statements, context.get("write_tables", [])
        )
    except Exception as exc:  # noqa: BLE001 — the model needs the reason to retry
        return f"ERRO na transação (nada foi gravado): {exc}"
    _invalidar_leituras()
    return json.dumps(
        {"status": "ok", "statements": len(results), "results": results},
        ensure_ascii=False,
        default=str,
    )


@catalog.tool()
async def query_agent_rag(question: str) -> str:
    """Busca semântica nos documentos da empresa vinculados a você. Use para
    responder perguntas sobre conteúdo de arquivos (políticas, manuais,
    contratos). Retorna os trechos mais relevantes com a fonte."""
    context = _context()
    file_ids = context.get("agent_files", {}).get(context.get("agent", ""), [])
    if not file_ids:
        return "Nenhum documento vinculado a este agente."

    from app.ingestion import search_chunks

    results = await search_chunks(
        context.get("embedding", {"provider": "stub"}),
        question,
        file_ids,
        context.get("tenant_id"),
    )
    if not results:
        return "Nenhum trecho relevante encontrado nos documentos."
    parts = [
        f"[{r['file']} — trecho {r['chunk']} — similaridade {r['similarity']}]\n{r['content']}"
        for r in results
    ]
    return "\n\n".join(parts)


async def _publicar_qr(context: dict, qr_base64: str | None, valor) -> object | None:
    """Publica a imagem do QR Code na conversa e devolve o artefato.

    Falhar aqui não pode derrubar a cobrança: o copia-e-cola sozinho já paga.
    Por isso o except engole o erro e só registra — quem chama trata `None`
    como "não deu para mostrar a imagem".
    """
    if not qr_base64:
        return None
    try:
        from app.storage import register_artifact

        return await register_artifact(
            tenant_id=context.get("tenant_id"),
            chat_id=context.get("chat_id"),
            agent_name=context.get("agent", ""),
            kind="image",
            title=f"QR Code PIX — R$ {valor}",
            schema_json=None,
            preview_json=None,
            row_count=None,
            payload=base64.b64decode(qr_base64),
            content_type="image/png",
            extension="png",
        )
    except Exception:  # noqa: BLE001
        logger.exception("falha ao publicar o QR Code como artefato")
        return None


def _dados_do_pix(payment: dict) -> dict:
    """Extrai copia-e-cola e QR de um pagamento já buscado no gateway.

    O Mercado Pago devolve os dois em point_of_interaction.transaction_data; a
    consulta traz os mesmos campos da criação. É isto que permite *mostrar* uma
    cobrança existente em vez de criar outra só para exibir o código.
    """
    interacao = payment.get("point_of_interaction") or {}
    return interacao.get("transaction_data") or {}


@catalog.tool()
async def generate_pix_charge(
    amount: str,
    description: str = "",
    reference_id: str = "",
    payer_email: str = "",
) -> str:
    """Gera uma cobrança PIX real para o cliente pagar (Mercado Pago).

    `amount` é o valor a cobrar em reais (ex.: "48.90") e deve vir do total do
    pedido — NUNCA invente nem arredonde. `reference_id` é o identificador do
    pedido no sistema, para conseguir consultar o pagamento depois.
    Retorna o código copia-e-cola do PIX e o payment_id da cobrança. Entregue o
    copia-e-cola ao cliente e use check_payment_status para saber se caiu.
    A imagem do QR Code já é publicada na conversa automaticamente: quando
    `qr_code_exibido` vier verdadeiro, o cliente já está vendo o QR — diga
    isso em vez de afirmar que não consegue mostrar imagem."""
    context = _context()
    credential = context.get("payment") or {}
    if not credential.get("access_token"):
        return (
            "ERRO: esta empresa ainda não configurou a credencial de pagamento. "
            "Peça ao administrador para cadastrá-la em Pagamentos."
        )

    from app.payments import create_pix_charge, parse_amount

    try:
        value = parse_amount(amount)
    except ValueError as exc:
        return f"ERRO: {exc}"

    try:
        charge = await create_pix_charge(
            credential=credential,
            tenant_id=context.get("tenant_id"),
            chat_id=context.get("chat_id"),
            amount=value,
            description=description or "Cobrança PIX",
            reference_id=reference_id,
            payer_email=payer_email or "comprador@example.com",
            notification_url=credential.get("notification_url"),
        )
    except httpx.HTTPStatusError as exc:
        return f"ERRO do gateway ({exc.response.status_code}): {exc.response.text[:500]}"
    except Exception as exc:  # noqa: BLE001 — o modelo precisa do motivo
        return f"ERRO ao gerar cobrança: {exc}"

    qr_artifact = await _publicar_qr(context, charge.get("qr_code_base64"), charge["amount"])

    return json.dumps(
        {
            "status": "ok",
            "payment_id": charge["external_id"],
            "amount": str(charge["amount"]),
            "payment_status": charge["status"],
            "pix_copia_e_cola": charge["qr_code"],
            "ticket_url": charge["ticket_url"],
            # O descriptor precisa VOLTAR no retorno da tool: o kernel só emite o
            # evento de artefato para o cliente quando encontra um artifact_id
            # aqui dentro. Publicar sem devolver gravava o QR e nunca o mostrava
            # — e o modelo, sem ver imagem alguma, dizia ao usuário que não
            # conseguia exibir QR Code.
            "qr_code_artifact": qr_artifact,
            "qr_code_exibido": bool(qr_artifact),
            "sandbox": bool(credential.get("sandbox", True)),
        },
        ensure_ascii=False,
    )


@catalog.tool()
async def check_payment_status(payment_id: str = "", reference_id: str = "") -> str:
    """Consulta uma cobrança PIX já existente: status, valor e o código
    copia-e-cola dela. Informe o `payment_id` devolvido por generate_pix_charge
    ou o `reference_id` do pedido.

    Use SEMPRE que o cliente pedir de novo o código, o QR ou os dados de uma
    cobrança que você já criou — NUNCA chame generate_pix_charge para isso, pois
    aquilo cria uma cobrança nova e pagável. generate_pix_charge é só para uma
    cobrança que ainda não existe. Também use aqui para saber se o pagamento
    caiu — a confirmação automática pode demorar alguns segundos. A imagem do QR
    é republicada na conversa: com `qr_code_exibido` verdadeiro, o cliente já
    está vendo o QR."""
    context = _context()
    credential = context.get("payment") or {}
    if not credential.get("access_token"):
        return "ERRO: esta empresa ainda não configurou a credencial de pagamento."

    from app.payments import STATUS_MAP, fetch_payment, lookup_charge, sync_charge_status

    external_id = payment_id.strip()
    if not external_id and reference_id.strip():
        row = await lookup_charge(
            tenant_id=context.get("tenant_id"), reference_id=reference_id.strip()
        )
        if row is None:
            return f"Nenhuma cobrança encontrada para a referência '{reference_id}'."
        external_id = row["external_id"]
    if not external_id:
        return "ERRO: informe payment_id ou reference_id."

    try:
        payment = await fetch_payment(credential=credential, external_id=external_id)
    except httpx.HTTPStatusError as exc:
        return f"ERRO do gateway ({exc.response.status_code}): {exc.response.text[:500]}"
    except Exception as exc:  # noqa: BLE001
        return f"ERRO ao consultar cobrança: {exc}"

    status = STATUS_MAP.get(payment.get("status", ""), "pending")
    await sync_charge_status(
        tenant_id=context.get("tenant_id"),
        external_id=external_id,
        status=status,
        payment=payment,
    )
    pix = _dados_do_pix(payment)
    # Só republica o QR enquanto dá para pagar: reexibir QR de cobrança paga ou
    # cancelada convida um segundo pagamento.
    qr_artifact = (
        await _publicar_qr(
            context, pix.get("qr_code_base64"), payment.get("transaction_amount")
        )
        if status == "pending"
        else None
    )
    return json.dumps(
        {
            "payment_id": external_id,
            "status": status,
            "gateway_status": payment.get("status"),
            "amount": payment.get("transaction_amount"),
            "paid": status == "paid",
            "pix_copia_e_cola": pix.get("qr_code") or "",
            "qr_code_artifact": qr_artifact,
            "qr_code_exibido": bool(qr_artifact),
        },
        ensure_ascii=False,
    )


def open_catalog_session():
    """Async context manager yielding a live MCP ClientSession over the
    in-memory transport, connected to the platform catalog."""
    return create_connected_server_and_client_session(catalog._mcp_server)


class ExternalServers:
    """Connects a template's external MCP servers (streamable HTTP) and maps
    their tools under ext_<server>_<tool>."""

    def __init__(self, servers: list[dict]):
        self._servers = servers
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, tuple[str, str, dict]] = {}  # public -> (server, tool, schema)

    async def __aenter__(self):
        from mcp.client.streamable_http import streamablehttp_client

        self._stack = AsyncExitStack()
        for server in self._servers:
            name = server["name"]
            headers = (
                {"Authorization": f"Bearer {server['auth_token']}"}
                if server.get("auth_token")
                else None
            )
            try:
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(server["url"], headers=headers)
                )
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.sessions[name] = session
                listed = await session.list_tools()
                for tool in listed.tools:
                    public = f"ext_{name}_{tool.name}"
                    self.tools[public] = (name, tool.name, tool.inputSchema or {})
            except Exception:  # noqa: BLE001 — a dead external server must not kill the run
                continue
        return self

    async def __aexit__(self, *exc):
        if self._stack:
            await self._stack.aclose()

    async def call(self, public_name: str, arguments: dict) -> str:
        server, tool, _ = self.tools[public_name]
        result = await self.sessions[server].call_tool(tool, arguments)
        parts = [c.text for c in result.content if getattr(c, "text", None)]
        return "\n".join(parts) or "(sem conteúdo)"
