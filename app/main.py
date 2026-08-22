from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
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
        async for _ in stream_completion(config, [{"role": "user", "content": "responda: ok"}]):
            break  # first delta is proof enough
        return {"ok": True, "detail": ""}
    except Exception as exc:  # noqa: BLE001 — the point is reporting it
        return {"ok": False, "detail": str(exc)[:500]}


_MAX_CUSTOM_ARTIFACT_BYTES = 250 * 1024 * 1024


class ArtifactRegisterInitIn(BaseModel):
    tenant_id: str
    kind: str = "file"
    extension: str = "bin"
    content_type: str = "application/octet-stream"


class ArtifactRegisterCompleteIn(ArtifactRegisterInitIn):
    artifact_id: str
    chat_id: str | None = None
    agent_name: str = "custom_tool"
    title: str = "Arquivo gerado"
    schema_json: dict | list | None = None
    preview_json: dict | list | None = None
    row_count: int | None = None


@app.post("/v1/artifacts/register-init", dependencies=[Depends(require_internal_auth)])
async def artifact_register_init(payload: ArtifactRegisterInitIn):
    """Create a short-lived direct-upload URL; bytes never traverse the kernel."""
    import uuid

    from app.storage import artifact_upload_target

    artifact_id = str(uuid.uuid4())
    path, upload_url = artifact_upload_target(
        tenant_id=payload.tenant_id,
        kind=payload.kind,
        artifact_id=artifact_id,
        extension=payload.extension,
        content_type=payload.content_type,
    )
    if upload_url is None:
        return {"artifact_id": artifact_id, "storage_path": path, "upload_url": None}
    return {"artifact_id": artifact_id, "storage_path": path, "upload_url": upload_url}


@app.post("/v1/artifacts/register-complete", dependencies=[Depends(require_internal_auth)])
async def artifact_register_complete(payload: ArtifactRegisterCompleteIn):
    """Verify direct upload size then atomically publish its artifact metadata."""
    from app.storage import (
        artifact_payload_size,
        artifact_upload_target,
        delete_payload,
        register_uploaded_artifact,
    )

    expected_path, _ = artifact_upload_target(
        tenant_id=payload.tenant_id,
        kind=payload.kind,
        artifact_id=payload.artifact_id,
        extension=payload.extension,
        content_type=payload.content_type,
    )
    try:
        size = artifact_payload_size(expected_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Upload de artifact não encontrado") from exc
    if size > _MAX_CUSTOM_ARTIFACT_BYTES:
        delete_payload(expected_path)
        raise HTTPException(status_code=413, detail="Artifact excede o limite de 250MB")
    return await register_uploaded_artifact(
        artifact_id=payload.artifact_id,
        tenant_id=payload.tenant_id,
        chat_id=payload.chat_id,
        agent_name=payload.agent_name,
        kind=payload.kind,
        title=payload.title,
        schema_json=payload.schema_json,
        preview_json=payload.preview_json,
        row_count=payload.row_count,
        storage_path=expected_path,
        content_type=payload.content_type,
    )


class StubScriptIn(BaseModel):
    rules: list[tuple[str, str]] = []
    default: str = "ok"


if settings.enable_stub_control:

    @app.post("/stub/script")
    def set_stub_script(payload: StubScriptIn):
        """Test-only: program the deterministic stub provider."""
        providers.stub_script = providers.StubScript(payload.rules, payload.default)
        return {"status": "ok"}
