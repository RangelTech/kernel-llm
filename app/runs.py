"""POST /v1/runs — execute one conversation turn, streaming SSE.

Events:
  event: token        data: {"text": "..."}      (supervisor deltas)
  event: agent        data: {"name": "...", "status": "start|done"}
  event: limit        data: {"detail": "max_steps"}
  event: done         data: {"text": "<full reply>"}
  event: error        data: {"detail": "..."}

Dois limites, e a diferença importa. `turn_timeout_seconds` é silêncio entre
eventos; `turn_total_timeout_seconds` é o turno inteiro. Só o primeiro existia,
e ele sozinho não limita nada: um turno que chama ferramenta a cada poucos
segundos reinicia a espera a cada evento e segue indefinidamente. Medido numa
conversa real, um turno passou de oito minutos e 66 chamadas assim — e quem ia
cortá-lo era o Cloud Run aos 600 s, cortando a conexão em vez de emitir o
evento de timeout. Qualquer um dos dois emite `error` e encerra o run. Cliente
que desconecta cancela o run limpo.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import get_graph

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])


class ModelSpec(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict = Field(default_factory=dict)


class AgentSpec(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    description: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    model: ModelSpec
    tools: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)


class SupervisorSpec(BaseModel):
    prompt: str = ""
    model: ModelSpec


class McpServerSpec(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[a-z0-9_]+$")
    url: str
    auth_token: str | None = None


class DatasourceSpec(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    kind: str
    config: dict = Field(default_factory=dict)
    secret: str | None = None


class RunRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1)
    supervisor: SupervisorSpec
    agents: list[AgentSpec] = Field(default_factory=list)
    max_steps: int = Field(default=6, ge=1, le=20)
    # Quantas mensagens recentes seguem para o modelo.
    history_limit: int = Field(default=100, ge=4, le=500)
    # Resumir o começo em vez de descartá-lo.
    compress_history: bool = False
    # Teto do que uma ferramenta devolve para dentro do prompt do especialista.
    tool_output_limit: int = Field(default=24_000, ge=2_000, le=200_000)
    # Teto de chamadas de ferramenta no turno inteiro, somando especialistas.
    max_tool_calls_per_turn: int = Field(default=40, ge=5, le=200)
    tenant_id: str | None = None
    user_id: str | None = None
    # name -> value, resolved into {{secret:NAME}} references inside tools;
    # never exposed to the model context.
    secrets: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
    datasources: list[DatasourceSpec] = Field(default_factory=list)
    # Embedding provider for RAG lookups: {provider, model, api_key}.
    embedding: dict = Field(default_factory=dict)
    # SQL write allowlist + conversational confirmation flag.
    write_tables: list[str] = Field(default_factory=list)
    require_write_confirmation: bool = True
    # Credencial de gateway de pagamento do tenant: {provider, access_token,
    # sandbox}. Igual aos secrets, nunca entra no contexto do modelo.
    payment: dict = Field(default_factory=dict)
    # Uploaded attachments already in object storage.
    attachments: list[dict] = Field(default_factory=list)
    # Whisper provider for audio attachments: {provider, model, api_key}.
    transcription: dict = Field(default_factory=dict)
    # Set only by the backend while resolving the RAgentes system template.
    tenant_guide_enabled: bool = False


def require_internal_auth(request: Request) -> None:
    """Service-to-service gate. In Cloud Run the platform enforces OIDC; this
    shared-token check covers dev and doubles as defense in depth."""
    if not settings.internal_token:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {settings.internal_token}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/v1/runs", dependencies=[Depends(require_internal_auth)])
async def create_run(payload: RunRequest):
    graph = await get_graph()

    run_config = {
        "supervisor": payload.supervisor.model_dump(),
        "agents": [a.model_dump() for a in payload.agents],
        "max_steps": payload.max_steps,
        "history_limit": payload.history_limit,
        "compress_history": payload.compress_history,
        "tool_output_limit": payload.tool_output_limit,
        "max_tool_calls_per_turn": payload.max_tool_calls_per_turn,
        "tenant_id": payload.tenant_id,
        "user_id": payload.user_id,
        "thread_id": payload.thread_id,
        "secrets": payload.secrets,
        "mcp_servers": [s.model_dump() for s in payload.mcp_servers],
        "datasources": [d.model_dump() for d in payload.datasources],
        "embedding": payload.embedding,
        "payment": payload.payment,
        "attachments": payload.attachments,
        "tenant_guide_enabled": payload.tenant_guide_enabled,
        "write_tables": payload.write_tables,
        "require_write_confirmation": payload.require_write_confirmation,
    }

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def produce():
            final_text = ""
            try:
                from app.attachments import build_user_content

                user_content = await build_user_content(
                    payload.message,
                    payload.attachments,
                    payload.transcription or {"provider": "stub"},
                )
                # Long-term memories relevant to this turn ride in run_config
                # and are injected into the supervisor's system prompt.
                if payload.user_id and payload.embedding:
                    from app.memories import recall

                    run_config["memories"] = await recall(
                        tenant_id=payload.tenant_id,
                        user_id=payload.user_id,
                        query=payload.message,
                        embedding=payload.embedding,
                    )
                async for mode, chunk in graph.astream(
                    {
                        "messages": [{"role": "user", "content": user_content}],
                        "run_config": run_config,
                    },
                    config={"configurable": {"thread_id": payload.thread_id}},
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        kind = chunk.get("type")
                        if kind == "token":
                            await queue.put(("token", {"text": chunk["text"]}))
                        elif kind in ("agent_start", "agent_done"):
                            await queue.put(
                                (
                                    "agent",
                                    {
                                        "name": chunk["name"],
                                        "status": "start" if kind == "agent_start" else "done",
                                    },
                                )
                            )
                        elif kind == "tool_call":
                            await queue.put(
                                (
                                    "tool",
                                    {
                                        "agent": chunk["agent"],
                                        "tool": chunk["tool"],
                                        "status": chunk["status"],
                                        "duration_ms": chunk["duration_ms"],
                                    },
                                )
                            )
                        elif kind == "artifact":
                            await queue.put(
                                (
                                    "artifact",
                                    {
                                        "artifact_id": chunk["artifact_id"],
                                        "kind": chunk["kind"],
                                        "title": chunk["title"],
                                    },
                                )
                            )
                        elif kind == "limit":
                            await queue.put(("limit", {"detail": chunk["detail"]}))
                    elif mode == "values" and chunk.get("messages"):
                        final_text = chunk["messages"][-1].content
                if not (final_text or "").strip():
                    # Turno que termina sem texto é o pior jeito de falhar num
                    # chat: o cliente manda a pergunta, não recebe nada e nada
                    # dispara alarme. Aconteceu em campo (C1 t11) e passou como
                    # sucesso, porque `done` com texto vazio é indistinguível de
                    # resposta entregue. Aqui vira erro explícito, e o log
                    # carrega a thread para dar para investigar depois.
                    logger.error("EMPTY_ANSWER thread=%s", payload.thread_id)
                    await queue.put(("error", {"detail": "empty_answer"}))
                else:
                    await queue.put(("done", {"text": final_text}))
            except Exception as exc:  # noqa: BLE001 — reported to the client as an event
                logger.exception("run failed thread=%s", payload.thread_id)
                await queue.put(("error", {"detail": str(exc)}))
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        prazo = asyncio.get_running_loop().time() + settings.turn_total_timeout_seconds
        try:
            while True:
                restante = prazo - asyncio.get_running_loop().time()
                if restante <= 0:
                    producer.cancel()
                    yield _sse("error", {"detail": "timeout"})
                    return
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=min(settings.turn_timeout_seconds, restante),
                    )
                except TimeoutError:
                    producer.cancel()
                    yield _sse("error", {"detail": "timeout"})
                    return
                if item is None:
                    return
                event, data = item
                yield _sse(event, data)
        finally:
            producer.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
