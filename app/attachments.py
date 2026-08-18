"""Chat attachment preprocessing: images become multimodal content blocks,
audio is transcribed (Whisper API), documents are inlined as text context.

Runs before the graph so the checkpointer stores the final user content.
"""

import base64
import io
import logging

from app.ingestion import extract_text
from app.storage import load_payload

logger = logging.getLogger(__name__)

_DOC_CONTEXT_CAP = 30_000  # chars of inlined document text per message


async def transcribe(spec: dict, data: bytes, filename: str) -> str:
    """spec: {provider, model, api_key}. Stub decodes bytes (tests)."""
    if spec.get("provider") == "stub":
        return data.decode("utf-8", errors="replace")

    import litellm

    buffer = io.BytesIO(data)
    buffer.name = filename or "audio.webm"
    response = await litellm.atranscription(
        model=spec.get("model") or "whisper-1",
        file=buffer,
        api_key=spec.get("api_key"),
    )
    return response.text


async def build_user_content(
    message: str, attachments: list[dict], transcription_spec: dict
) -> str | list[dict]:
    """The user message enriched with attachments.

    Returns plain text when there are no images, else OpenAI-style content
    blocks (litellm translates for Gemini/Anthropic).
    """
    if not attachments:
        return message

    text_parts: list[str] = []
    image_blocks: list[dict] = []

    for attachment in attachments:
        kind = attachment.get("kind", "file")
        name = attachment.get("name", "anexo")
        try:
            data = load_payload(attachment["storage_path"])
        except Exception as exc:  # noqa: BLE001 — reported inline, not fatal
            logger.exception("failed to load attachment %s", name)
            text_parts.append(f"[anexo {name}: falha ao carregar — {exc}]")
            continue

        if kind == "image":
            encoded = base64.b64encode(data).decode()
            content_type = attachment.get("content_type", "image/png")
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{encoded}"},
                }
            )
            text_parts.append(f"[imagem anexada: {name}]")
        elif kind == "audio":
            try:
                transcript = await transcribe(transcription_spec, data, name)
                text_parts.append(f"[áudio {name} transcrito]: {transcript}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("transcription failed for %s", name)
                text_parts.append(f"[áudio {name}: falha na transcrição — {exc}]")
        else:
            try:
                text = extract_text(
                    data, attachment.get("content_type", ""), name
                )[:_DOC_CONTEXT_CAP]
                text_parts.append(f"[conteúdo do arquivo {name}]:\n{text}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("extraction failed for %s", name)
                text_parts.append(f"[arquivo {name}: falha na leitura — {exc}]")

    full_text = "\n\n".join(filter(None, [*text_parts, message]))
    if image_blocks:
        return [{"type": "text", "text": full_text}, *image_blocks]
    return full_text
