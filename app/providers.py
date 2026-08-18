"""LLM provider layer.

Everything the graph needs from a model goes through `stream_completion`, an
async generator of text deltas. Two implementations:

- litellm: the real path — provider-agnostic via litellm.acompletion(stream=True)
- stub: deterministic scripted responses for tests. The stub is part of the
  product's test seam, not test code: suites drive the whole HTTP surface with
  it and never spend a token.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Resolved model configuration for one completion call."""

    provider: str  # litellm provider prefix, e.g. "gemini", "openai", "stub"
    model: str  # model name, e.g. "gemini-2.5-flash"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def litellm_model(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else self.model


class StubScript:
    """Deterministic responses keyed by substring of the last user message.

    Falls back to `default`. Exposed via kernel settings for the test suites.
    Reply grammar:
      "texto qualquer"                 → plain text answer
      "__RAISE__"                      → scripted provider failure
      "TOOL:<name>:<json-args>"        → emit one tool call (agents-as-tools)
    """

    def __init__(self, rules: list[tuple[str, str]] | None = None, default: str = "ok"):
        self.rules = rules or []
        self.default = default

    def reply_for(self, prompt: str) -> str:
        for needle, reply in self.rules:
            if needle.lower() in prompt.lower():
                return reply
        return self.default


# Mutable so tests can swap scripts at runtime through the /stub endpoint.
stub_script = StubScript(
    rules=[
        ("erro proposital", "__RAISE__"),
    ],
    default="Resposta simulada do stub.",
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as providers emit it


@dataclass
class Completion:
    """Result of one model call: streamed text and/or requested tool calls."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Accounting: token counts are estimates (litellm token_counter) and cost
    # comes from litellm's price table — good enough for tenant reporting.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


def _scripted_reply(messages: list[dict]) -> str:
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] in ("user", "tool")),
        "",
    )
    if isinstance(last_user, list):  # multimodal content blocks
        last_user = " ".join(b.get("text", "") for b in last_user if isinstance(b, dict))
    return stub_script.reply_for(str(last_user))


async def _stream_stub(messages: list[dict], on_delta) -> Completion:
    reply = _scripted_reply(messages)
    if reply == "__RAISE__":
        raise RuntimeError("stub provider error (scripted)")
    if reply.startswith("TOOL:"):
        _, name, args = reply.split(":", 2)
        return Completion(tool_calls=[ToolCall(id="stub-call-1", name=name, arguments=args)])
    words = reply.split(" ")
    parts = []
    for i, word in enumerate(words):
        delta = word if i == 0 else " " + word
        parts.append(delta)
        await on_delta(delta)
    return Completion(content="".join(parts))


class ReasoningFilter:
    """Drops `<think>…</think>` blocks from model output.

    Reasoning models emit their scratchpad inline with the answer. It is not
    part of the reply and must never reach the user — a combo that mixes model
    families makes this unavoidable, since one provider emits the tags and
    another does not.

    Streaming is what makes this more than a `replace`: a tag arrives split
    across chunks, so a partial tag at the end of a chunk has to be held back
    until the next one proves what it is.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._held = ""
        self._inside = False

    def feed(self, text: str) -> str:
        """Returns the part of `text` that is safe to show right now."""
        self._held += text
        out: list[str] = []
        while self._held:
            if self._inside:
                fim = self._held.find(self.CLOSE)
                if fim == -1:
                    # Keep only what could still be the closing tag.
                    self._held = self._held[-(len(self.CLOSE) - 1) :]
                    break
                self._held = self._held[fim + len(self.CLOSE) :]
                self._inside = False
                continue
            inicio = self._held.find(self.OPEN)
            if inicio == -1:
                seguro = self._maior_prefixo_de_tag(self._held)
                if seguro:
                    out.append(self._held[:-seguro])
                    self._held = self._held[-seguro:]
                else:
                    out.append(self._held)
                    self._held = ""
                break
            out.append(self._held[:inicio])
            self._held = self._held[inicio + len(self.OPEN) :]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        """Whatever is left once the stream ends.

        Text held back inside an unterminated block is dropped: an answer that
        never closed its scratchpad has no reply to show, and printing the
        scratchpad would be worse than printing nothing.
        """
        if self._inside:
            self._held = ""
            return ""
        resto, self._held = self._held, ""
        return resto

    def _maior_prefixo_de_tag(self, texto: str) -> int:
        """Quantos caracteres finais podem ser o começo de `<think>`."""
        for tamanho in range(min(len(self.OPEN) - 1, len(texto)), 0, -1):
            if self.OPEN.startswith(texto[-tamanho:]):
                return tamanho
        return 0


def strip_reasoning(text: str) -> str:
    """Same rule applied to a complete, non-streamed answer."""
    filtro = ReasoningFilter()
    return filtro.feed(text) + filtro.flush()


async def _stream_litellm(
    config: ModelConfig, messages: list[dict], on_delta, tools: list[dict] | None
) -> Completion:
    import litellm

    kwargs = dict(
        model=config.litellm_model,
        messages=messages,
        api_key=config.api_key,
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
        **config.extra,
    )
    if tools:
        kwargs["tools"] = tools

    response = await litellm.acompletion(**kwargs)

    content_parts: list[str] = []
    # Streamed tool calls arrive as fragments indexed by position.
    calls: dict[int, dict] = {}
    reasoning = ReasoningFilter()
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if delta.content:
            visivel = reasoning.feed(delta.content)
            if visivel:
                content_parts.append(visivel)
                await on_delta(visivel)
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(
                tc.index, {"id": "", "name": "", "arguments": ""}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments

    resto = reasoning.flush()
    if resto:
        content_parts.append(resto)
        await on_delta(resto)

    tool_calls = [
        ToolCall(id=c["id"] or f"call-{i}", name=c["name"], arguments=c["arguments"] or "{}")
        for i, c in sorted(calls.items())
    ]
    return Completion(content="".join(content_parts), tool_calls=tool_calls)


def _estimate_usage(config: ModelConfig, messages: list[dict], result: Completion) -> None:
    """Fill token/cost estimates in place. Never raises."""
    try:
        if config.provider == "stub":
            result.prompt_tokens = sum(
                len(str(m.get("content") or "").split()) for m in messages
            )
            result.completion_tokens = len(result.content.split())
            result.cost_usd = 0.0
            return
        import litellm

        text_messages = [
            {"role": m.get("role", "user"), "content": str(m.get("content") or "")}
            for m in messages
        ]
        result.prompt_tokens = litellm.token_counter(
            model=config.litellm_model, messages=text_messages
        )
        result.completion_tokens = litellm.token_counter(
            model=config.litellm_model, text=result.content or " "
        )
        result.cost_usd = float(
            litellm.completion_cost(
                model=config.litellm_model,
                prompt=" ",
                completion=result.content or " ",
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        )
    except Exception:  # noqa: BLE001 — accounting must never break a call
        logger.debug("usage estimation failed for %s", config.litellm_model, exc_info=True)


async def complete(
    config: ModelConfig,
    messages: list[dict],
    on_delta,
    tools: list[dict] | None = None,
) -> Completion:
    """One model call. Text deltas go to `on_delta` as they stream; the full
    result (content + any tool calls + usage estimates) is returned at the end."""
    import time as _time

    started = _time.monotonic()
    if config.provider == "stub":
        result = await _stream_stub(messages, on_delta)
    else:
        result = await _stream_litellm(config, messages, on_delta, tools)
    result.latency_ms = int((_time.monotonic() - started) * 1000)
    _estimate_usage(config, messages, result)
    return result


def stream_completion(config: ModelConfig, messages: list[dict]) -> AsyncIterator[str]:
    """Text-only compatibility wrapper over `complete` (used by /v1/test-model)."""

    async def generator():
        queue: list[str] = []

        async def collect(delta: str):
            queue.append(delta)

        result = await complete(config, messages, collect)
        for d in queue:
            yield d
        if not queue and result.content:
            yield result.content

    return generator()
