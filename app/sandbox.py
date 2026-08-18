"""Isolated Python execution for agents.

The user code runs in a separate interpreter process with:
- a hard wall-clock timeout (killed, not asked);
- network access blocked (socket is neutered before user code runs);
- input dataset exposed as `columns`/`rows` (and `df` when pandas exists);
- `publish_dataset(title, columns, rows)` / `publish_document(title, text)`
  collected by the host and registered as artifacts.

Memory limits rely on the container in production (Cloud Run); Windows dev
has no rlimit equivalent, which is acceptable for local work.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anyio

_RUNNER_TEMPLATE = '''
import json, sys

# Block network before any user code runs.
import socket as _socket
def _no_net(*_a, **_k):
    raise RuntimeError("acesso à rede é bloqueado no sandbox")
_socket.socket = _no_net
_socket.create_connection = _no_net

_IN = json.load(open(sys.argv[1], encoding="utf-8"))
columns = _IN.get("columns", [])
rows = _IN.get("rows", [])
try:
    import pandas as pd
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns]) if columns else None
except Exception:
    df = None

_OUTPUTS = []
def publish_dataset(title, columns, rows):
    _OUTPUTS.append({"kind": "dataset", "title": title,
                     "columns": columns, "rows": rows})
def publish_document(title, text):
    _OUTPUTS.append({"kind": "document", "title": title, "text": str(text)})

try:
    exec(compile(open(sys.argv[3], encoding="utf-8").read(), "<agent-code>", "exec"))
except Exception as exc:
    print(f"ERRO NO CÓDIGO: {type(exc).__name__}: {exc}", file=sys.stderr)

json.dump(_OUTPUTS, open(sys.argv[2], "w", encoding="utf-8"), default=str)
'''


async def run_sandboxed(
    code: str, dataset: dict | None, timeout_seconds: int = 30
) -> tuple[str, str, list[dict], bool]:
    """(stdout, stderr, published_outputs, timed_out)."""
    workdir = Path(tempfile.mkdtemp(prefix="sandbox-"))
    input_path = workdir / "input.json"
    output_path = workdir / "output.json"
    code_path = workdir / "code.py"
    runner_path = workdir / "runner.py"

    input_path.write_text(
        json.dumps(dataset or {}, ensure_ascii=False, default=str), encoding="utf-8"
    )
    code_path.write_text(code, encoding="utf-8")
    runner_path.write_text(_RUNNER_TEMPLATE, encoding="utf-8")

    # subprocess.run in a worker thread: asyncio subprocess support is absent
    # on Windows' SelectorEventLoop (which psycopg requires in dev).
    def _run() -> tuple[bytes, bytes, bool]:
        try:
            completed = subprocess.run(  # noqa: S603 — sandboxed by design
                [
                    sys.executable,
                    str(runner_path),
                    str(input_path),
                    str(output_path),
                    str(code_path),
                ],
                capture_output=True,
                cwd=str(workdir),
                timeout=timeout_seconds,
            )
            return completed.stdout, completed.stderr, False
        except subprocess.TimeoutExpired as exc:
            return exc.stdout or b"", exc.stderr or b"", True

    stdout_data, stderr_data, timed_out = await anyio.to_thread.run_sync(_run)

    outputs: list[dict] = []
    if output_path.exists():
        try:
            outputs = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return (
        stdout_data.decode("utf-8", errors="replace")[:8_000],
        stderr_data.decode("utf-8", errors="replace")[:4_000],
        outputs,
        timed_out,
    )
