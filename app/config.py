from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Kernel settings. Dev defaults only; production values come from the
    environment (Cloud Run). Tenant/provider config lives in the database."""

    database_url: str = "postgresql://agent:agent@localhost:5433/agent_llm"
    port: int = 8080
    # Base da API do gateway de pagamento. Só muda em teste, para apontar para
    # um servidor falso em vez do Mercado Pago real.
    mercado_pago_api: str = "https://api.mercadopago.com"
    # Máximo de silêncio entre dois eventos do turno. Não é o teto do turno: um
    # turno que emite uma ferramenta a cada poucos segundos nunca encosta nisto.
    turn_timeout_seconds: float = 120.0
    # Teto do turno inteiro. Precisa ficar ABAIXO do timeout de requisição do
    # Cloud Run (600 s em infra/deploy.sh): passando dele, quem corta é a borda,
    # e o cliente recebe a conexão cortada no lugar do evento de timeout.
    turn_total_timeout_seconds: float = 540.0
    # Default supervisor step budget per turn when the template omits it.
    max_steps_default: int = 6
    # Tool rounds a specialist may take before being forced to answer.
    specialist_max_tool_rounds: int = 7
    # Teto, em caracteres, do que uma ferramenta devolve para dentro do prompt
    # do especialista. É aqui que o contexto realmente estoura: numa conversa de
    # 30 turnos o especialista `despesas` acumulou 889 mil tokens de prompt,
    # contra 10 mil do supervisor — o histórico da conversa é o problema menor.
    # O dataset inteiro não precisa voltar: o artifact_id encadeia para as
    # ferramentas seguintes sem passar pelo modelo.
    tool_output_limit_default: int = 24_000
    # Teto de chamadas de ferramenta no turno inteiro, somando todos os
    # especialistas. `specialist_max_tool_rounds` limita rodadas de UM
    # especialista; com o supervisor delegando várias vezes, um único turno já
    # disparou 30 chamadas e publicou 26 datasets para responder "faça um
    # gráfico de linha". Nada errava — só custava demais e demorava demais.
    max_tool_calls_per_turn: int = 40

    # Web search: Serper platform key (Secret Manager) + DuckDuckGo fallback.
    serper_api_key: str = ""
    serper_url: str = "https://google.serper.dev/search"
    web_search_max_results: int = 6
    # Pages a single analyze_pdf_pages call may send to the vision model.
    pdf_vision_max_pages: int = 20

    # Artifact payload storage. Priority: S3-compatible (MinIO) -> GCS -> local
    # dir. Must match the backend's settings so both sides read the same paths.
    storage_backend: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = "us-east-1"
    s3_bucket: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_prefix: str = "agent-llm"
    s3_public_base_url: str = ""
    s3_presign_expires_seconds: int = 3600
    gcs_bucket: str = ""
    gcs_prefix: str = "agent llm"
    artifacts_local_dir: str = "./artifacts"
    # Rows shown to the model as a dataset preview.
    artifact_preview_rows: int = 10
    # Hard cap on rows a SQL tool call may materialize.
    sql_max_rows: int = 50_000
    checkpoint_pool_size: int = 5
    # Shared secret for backend->kernel calls. Empty disables the check (dev).
    internal_token: str = ""
    # Exposes POST /stub/script so test suites can program the stub provider.
    enable_stub_control: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
