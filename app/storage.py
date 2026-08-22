"""Artifact payload storage: S3-compatible/MinIO, GCS, or local filesystem.

Metadata lives in the artifacts table (written here too, same pattern as the
tool-call trace); only the full payload goes to object storage.
"""

import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

from app.config import settings

logger = logging.getLogger(__name__)

_gcs_client = None
_s3_client = None


def _storage_backend() -> str:
    forced = (settings.storage_backend or "").strip().lower()
    if forced:
        return forced
    if settings.s3_bucket:
        return "s3"
    if settings.gcs_bucket:
        return "gcs"
    return "local"


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3

        kwargs = {
            "service_name": "s3",
            "region_name": settings.s3_region,
            "aws_access_key_id": settings.s3_access_key_id,
            "aws_secret_access_key": settings.s3_secret_access_key,
        }
        if settings.s3_endpoint_url:
            kwargs["endpoint_url"] = settings.s3_endpoint_url
        _s3_client = boto3.client(**kwargs)
    return _s3_client


def _gcs():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage as gcs

        _gcs_client = gcs.Client()
    return _gcs_client


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _join_key(prefix: str, relative_path: str) -> str:
    clean_prefix = _normalize_prefix(prefix)
    clean_relative = relative_path.lstrip("/")
    return f"{clean_prefix}/{clean_relative}" if clean_prefix else clean_relative


def _split_storage_uri(storage_path: str, scheme: str) -> tuple[str, str]:
    _, _, rest = storage_path.partition(f"{scheme}://")
    bucket_name, _, object_key = rest.partition("/")
    return bucket_name, object_key


def save_payload(relative_path: str, data: bytes, content_type: str) -> str:
    """Store bytes; returns the storage path (s3://..., gs://..., or local path)."""
    backend = _storage_backend()
    if backend == "s3":
        object_key = _join_key(settings.s3_prefix, relative_path)
        _s3().put_object(
            Bucket=settings.s3_bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{settings.s3_bucket}/{object_key}"
    if backend == "gcs":
        blob_name = _join_key(settings.gcs_prefix, relative_path)
        bucket = _gcs().bucket(settings.gcs_bucket)
        bucket.blob(blob_name).upload_from_string(data, content_type=content_type)
        return f"gs://{settings.gcs_bucket}/{blob_name}"
    base = Path(settings.artifacts_local_dir)
    path = base / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def load_payload(storage_path: str) -> bytes:
    if storage_path.startswith("s3://"):
        bucket_name, object_key = _split_storage_uri(storage_path, "s3")
        response = _s3().get_object(Bucket=bucket_name, Key=object_key)
        return response["Body"].read()
    if storage_path.startswith("gs://"):
        bucket_name, blob_name = _split_storage_uri(storage_path, "gs")
        return _gcs().bucket(bucket_name).blob(blob_name).download_as_bytes()
    return Path(storage_path).read_bytes()


def signed_url(storage_path: str, expires_seconds: int = 3600) -> str | None:
    """Browser-usable URL for a stored payload (None for local dev paths)."""
    if storage_path.startswith("s3://"):
        bucket_name, object_key = _split_storage_uri(storage_path, "s3")
        public_base = (settings.s3_public_base_url or "").rstrip("/")
        if public_base:
            return f"{public_base}/{quote(object_key)}"
        try:
            return _s3().generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_key},
                ExpiresIn=expires_seconds or settings.s3_presign_expires_seconds,
            )
        except Exception:  # noqa: BLE001 — degrade to no URL if presign fails
            logger.exception("failed to sign s3 url for %s", storage_path)
            return None
    if not storage_path.startswith("gs://"):
        return None
    bucket_name, blob_name = _split_storage_uri(storage_path, "gs")
    try:
        return (
            _gcs()
            .bucket(bucket_name)
            .blob(blob_name)
            .generate_signed_url(expiration=timedelta(seconds=expires_seconds))
        )
    except Exception:  # noqa: BLE001 — SA without signing perms: degrade to no URL
        logger.exception("failed to sign url for %s", storage_path)
        return None


def artifact_upload_target(
    *, tenant_id: str, kind: str, artifact_id: str, extension: str, content_type: str
) -> tuple[str, str | None]:
    """Return the final path and one-time PUT URL without carrying file bytes through Cloud Run."""
    safe_kind = kind if kind in {"file", "image", "dataset", "document"} else "file"
    safe_extension = "".join(ch for ch in extension.lower() if ch.isalnum())[:16] or "bin"
    relative = f"tenants/{tenant_id}/artifacts/{safe_kind}/{artifact_id}.{safe_extension}"
    backend = _storage_backend()
    if backend == "s3":
        key = _join_key(settings.s3_prefix, relative)
        path = f"s3://{settings.s3_bucket}/{key}"
        return (
            path,
            _s3().generate_presigned_url(
                "put_object",
                Params={"Bucket": settings.s3_bucket, "Key": key, "ContentType": content_type},
                ExpiresIn=900,
            ),
        )
    if backend == "gcs":
        name = _join_key(settings.gcs_prefix, relative)
        path = f"gs://{settings.gcs_bucket}/{name}"
        url = (
            _gcs()
            .bucket(settings.gcs_bucket)
            .blob(name)
            .generate_signed_url(
                expiration=timedelta(minutes=15), method="PUT", content_type=content_type
            )
        )
        return path, url
    return str(Path(settings.artifacts_local_dir) / relative), None


def artifact_payload_size(storage_path: str) -> int:
    if storage_path.startswith("s3://"):
        bucket, key = _split_storage_uri(storage_path, "s3")
        return int(_s3().head_object(Bucket=bucket, Key=key)["ContentLength"])
    if storage_path.startswith("gs://"):
        bucket, name = _split_storage_uri(storage_path, "gs")
        blob = _gcs().bucket(bucket).blob(name)
        blob.reload()
        return int(blob.size or 0)
    return Path(storage_path).stat().st_size


def delete_payload(storage_path: str) -> None:
    if storage_path.startswith("s3://"):
        bucket, key = _split_storage_uri(storage_path, "s3")
        _s3().delete_object(Bucket=bucket, Key=key)
    elif storage_path.startswith("gs://"):
        bucket, name = _split_storage_uri(storage_path, "gs")
        _gcs().bucket(bucket).blob(name).delete()
    else:
        Path(storage_path).unlink(missing_ok=True)


async def register_uploaded_artifact(
    *,
    artifact_id: str,
    tenant_id: str,
    chat_id: str | None,
    agent_name: str,
    kind: str,
    title: str,
    schema_json: dict | list | None,
    preview_json: dict | list | None,
    row_count: int | None,
    storage_path: str,
    content_type: str,
) -> dict:
    """Commit metadata only after the direct object-storage upload is verified."""
    from app.trace import _get_pool

    pool = await _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO artifacts
               (id, tenant_id, chat_id, agent_name, kind, title, schema_json,
                preview_json, row_count, storage_path, content_type)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                artifact_id,
                tenant_id,
                chat_id,
                agent_name,
                kind,
                title,
                json.dumps(schema_json, ensure_ascii=False, default=str),
                json.dumps(preview_json, ensure_ascii=False, default=str),
                row_count,
                storage_path,
                content_type,
            ),
        )
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "title": title,
        "schema": schema_json,
        "preview": preview_json,
        "row_count": row_count,
    }


async def register_artifact(
    *,
    tenant_id,
    chat_id,
    agent_name: str,
    kind: str,
    title: str,
    schema_json: dict | list | None,
    preview_json: dict | list | None,
    row_count: int | None,
    payload: bytes,
    content_type: str = "application/json",
    extension: str = "json",
) -> dict:
    """Persist payload + metadata row. Returns the artifact descriptor the
    tools hand back to the model (id + schema + preview, never the payload)."""
    artifact_id = str(uuid.uuid4())
    tenant_segment = str(tenant_id) if tenant_id else "shared"
    relative = f"tenants/{tenant_segment}/artifacts/{kind}/{artifact_id}.{extension}"
    storage_path = save_payload(relative, payload, content_type)

    from app.trace import _get_pool  # same tiny pool, artifacts are trace-adjacent

    pool = await _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO artifacts
                   (id, tenant_id, chat_id, agent_name, kind, title, schema_json,
                    preview_json, row_count, storage_path, content_type)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                artifact_id,
                tenant_id,
                chat_id,
                agent_name,
                kind,
                title,
                json.dumps(schema_json, ensure_ascii=False, default=str),
                json.dumps(preview_json, ensure_ascii=False, default=str),
                row_count,
                storage_path,
                content_type,
            ),
        )
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "title": title,
        "schema": schema_json,
        "preview": preview_json,
        "row_count": row_count,
    }


async def get_artifact(artifact_id: str) -> dict | None:
    from app.trace import _get_pool

    # O supervisor é instruído (graph.py:_nota_dos_artefatos) a embutir links
    # markdown no formato "[titulo](artifact:UUID)" nas respostas — é assim
    # que o artefato aparece pro modelo no histórico da conversa. Quando um
    # turno seguinte pede pra encadear esse artefato (ex.: "exporte esse
    # resultado"), o modelo naturalmente copia o texto que já viu, prefixo
    # "artifact:" incluso, como argumento da tool. Sem isso o cast pra uuid
    # no WHERE explode (psycopg.errors.InvalidTextRepresentation) e o erro
    # cru do driver chega ao usuário em vez de um retry limpo.
    if artifact_id.startswith("artifact:"):
        artifact_id = artifact_id[len("artifact:") :]

    pool = await _get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT * FROM artifacts WHERE id = %s", (artifact_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    columns = [d.name for d in cursor.description]
    return dict(zip(columns, row, strict=True))
