"""Kernel test fixtures.

The kernel keeps process-global connection pools (checkpointer, trace), so
the whole test session must share ONE event loop — anyio gives us that by
scoping the backend fixture to the session. Pools are closed once at the end.
"""

import asyncio
import json
import sys
from pathlib import Path

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

# psycopg's async support cannot run on Windows' default ProactorEventLoop.
# Production runs on Linux; this only affects the local dev test run.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _apply_schema_migrations():
    """Kernel integration tests need only the tables the kernel itself uses —
    ver `kernel/tests/schema.sql` (autocontido, sem depender de `backend/`)."""
    from app.config import settings
    from db_guard import exigir_banco_descartavel

    # Aplicar schema é escrita: vale a mesma recusa que protege o backend.
    exigir_banco_descartavel(settings.database_url)

    schema_path = Path(__file__).resolve().parent / "schema.sql"
    with psycopg.connect(settings.database_url, connect_timeout=15) as conn:
        conn.execute(schema_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def _isolate_local_storage(tmp_path_factory):
    """Keep artifacts created by tests out of kernel/artifacts."""
    from app.config import settings

    settings.storage_backend = "local"
    settings.s3_bucket = ""
    settings.gcs_bucket = ""
    settings.artifacts_local_dir = str(tmp_path_factory.mktemp("kernel-artifacts"))


@pytest.fixture(scope="session")
async def client(anyio_backend):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://kernel") as c:
        yield c

    from app.graph import close_graph
    from app.trace import close_trace_pool

    await close_trace_pool()
    await close_graph()


def sse_events(sse_text: str) -> list[tuple[str, dict]]:
    events = []
    for block in sse_text.strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event:
            events.append((event, data))
    return events
