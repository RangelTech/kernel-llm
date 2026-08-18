from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app import providers, tools_output  # noqa: F401 — registers output tools
from app.config import settings
from app.graph import close_graph
from app.runs import require_internal_auth
from app.runs import router as runs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    from app.trace import close_trace_pool

    await close_trace_pool()
    await close_graph()


app = FastAPI(title="agent-platform kernel", lifespan=lifespan)
app.include_router(runs_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "kernel"}


@app.get("/v1/tools", dependencies=[Depends(require_internal_auth)])
async def list_platform_tools():
    """Tool catalog via MCP list_tools — feeds the template editor."""
    from app.tools import open_catalog_session

    async with open_catalog_session() as session:
        listed = await session.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.inputSchema or {},
        }
        for t in listed.tools
    ]


class IngestFileIn(BaseModel):
    file_id: str
    embedding: dict = {}


@app.post("/v1/ingest-file", dependencies=[Depends(require_internal_auth)])
async def ingest_file_endpoint(payload: IngestFileIn):
    """Chunk + embed one uploaded file. Called by Cloud Tasks (prod) or the
    backend's background task (dev). Errors land in files.status."""
    from app.ingestion import ingest_file

    await ingest_file(payload.file_id, payload.embedding or {"provider": "stub"})
    return {"status": "ok"}


class ExtractMemoriesIn(BaseModel):
    thread_id: str
    tenant_id: str
    user_id: str
    model: dict
    embedding: dict = {}


@app.post("/v1/extract-memories", dependencies=[Depends(require_internal_auth)])
async def extract_memories_endpoint(payload: ExtractMemoriesIn):
    """Post-conversation fact extraction (async via Cloud Tasks / dev inline)."""
    from app.memories import extract_memories

    saved = await extract_memories(
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
        model=payload.model,
        embedding=payload.embedding or {"provider": "stub"},
    )
    return {"saved": saved}


class TestDatasourceIn(BaseModel):
    kind: str
    config: dict = {}
    secret: str | None = None


@app.post("/v1/test-datasource", dependencies=[Depends(require_internal_auth)])
async def test_datasource(payload: TestDatasourceIn):
    from app.datasources import test_connection

    ok, detail = await test_connection(payload.model_dump())
    return {"ok": ok, "detail": detail}


class TestModelIn(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None


@app.post("/v1/test-model", dependencies=[Depends(require_internal_auth)])
async def test_model(payload: TestModelIn):
    """One tiny completion to prove the credentials/model work."""
    from app.providers import ModelConfig, stream_completion

    config = ModelConfig(
        provider=payload.provider,
        model=payload.model,
        api_key=payload.api_key,
        api_base=payload.api_base,
        max_tokens=5,
    )
    try:
        async for _ in stream_completion(
            config, [{"role": "user", "content": "responda: ok"}]
        ):
            break  # first delta is proof enough
        return {"ok": True, "detail": ""}
    except Exception as exc:  # noqa: BLE001 — the point is reporting it
        return {"ok": False, "detail": str(exc)[:500]}


class StubScriptIn(BaseModel):
    rules: list[tuple[str, str]] = []
    default: str = "ok"


if settings.enable_stub_control:

    @app.post("/stub/script")
    def set_stub_script(payload: StubScriptIn):
        """Test-only: program the deterministic stub provider."""
        providers.stub_script = providers.StubScript(payload.rules, payload.default)
        return {"status": "ok"}
