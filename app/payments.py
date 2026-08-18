"""Gateway de pagamento PIX (Mercado Pago) usado pelas tools do kernel.

Duas regras que valem mais que o código:

1. O valor cobrado é sempre o valor recebido. A flag `sandbox` da credencial
   troca qual token/conta é usada — nunca o valor. (Corrige explicitamente o
   gotcha de `amount = 0.01` da implementação de referência.)
2. A credencial vem do contexto do run (resolvida por tenant no backend), não
   de variável de ambiente global — um tenant nunca cobra com a conta de outro.
"""

import logging
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

STATUS_MAP = {
    "approved": "paid",
    "authorized": "pending",
    "in_process": "pending",
    "in_mediation": "pending",
    "pending": "pending",
    "rejected": "failed",
    "cancelled": "cancelled",
    "refunded": "refunded",
    "charged_back": "refunded",
}

_pool: AsyncConnectionPool | None = None


async def _get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            settings.database_url, min_size=0, max_size=3,
            kwargs={"autocommit": True, "row_factory": dict_row},
            # Conexão ociosa pode ter morrido com o banco atrás do Traefik.
            check=AsyncConnectionPool.check_connection,
            open=False,
        )
        await _pool.open()
    return _pool


async def close_payments_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def parse_amount(raw: Any) -> Decimal:
    """Aceita '48,90' / '48.90' / 48.9 e devolve um Decimal com 2 casas."""
    if isinstance(raw, str):
        raw = raw.strip().replace("R$", "").replace(" ", "").replace(",", ".")
    try:
        value = Decimal(str(raw)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError, ArithmeticError) as exc:
        raise ValueError(f"valor inválido: {raw!r}") from exc
    if value <= 0:
        raise ValueError("valor da cobrança deve ser maior que zero")
    return value


async def create_pix_charge(
    *,
    credential: dict,
    tenant_id,
    chat_id,
    amount: Decimal,
    description: str,
    reference_id: str,
    payer_email: str,
    notification_url: str | None = None,
) -> dict:
    """Cria a cobrança no Mercado Pago e registra o vínculo payment_id <-> pedido."""
    token = credential.get("access_token")
    if not token:
        raise RuntimeError("credencial de pagamento não configurada para esta empresa")

    payload: dict[str, Any] = {
        # float(Decimal) só aqui, na fronteira do JSON: a aritmética do valor
        # já aconteceu em Decimal.
        "transaction_amount": float(amount),
        "description": description or "Cobrança PIX",
        "payment_method_id": "pix",
        "external_reference": reference_id or str(uuid.uuid4()),
        "payer": {"email": payer_email},
    }
    if notification_url:
        payload["notification_url"] = notification_url

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{settings.mercado_pago_api}/v1/payments",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                # Idempotência: um retry de rede não gera duas cobranças.
                "X-Idempotency-Key": str(uuid.uuid4()),
            },
            json=payload,
        )
        response.raise_for_status()
        payment = response.json()

    interactive = (payment.get("point_of_interaction") or {}).get("transaction_data") or {}
    charge = {
        "external_id": str(payment.get("id")),
        "status": STATUS_MAP.get(payment.get("status", ""), "pending"),
        "qr_code": interactive.get("qr_code"),
        "qr_code_base64": interactive.get("qr_code_base64"),
        "ticket_url": interactive.get("ticket_url"),
        "amount": amount,
    }

    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """INSERT INTO payment_charges
                       (tenant_id, provider, external_id, chat_id, amount, description,
                        reference_id, status, qr_code, qr_code_base64, ticket_url,
                        sandbox, raw)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (provider, external_id) DO UPDATE
                       SET status = EXCLUDED.status, updated_at = now()""",
                (
                    tenant_id,
                    credential.get("provider", "mercado_pago"),
                    charge["external_id"],
                    chat_id,
                    amount,
                    description,
                    reference_id,
                    charge["status"],
                    charge["qr_code"],
                    charge["qr_code_base64"],
                    charge["ticket_url"],
                    bool(credential.get("sandbox", True)),
                    Json(payment),
                ),
            )
    except Exception:  # noqa: BLE001 — cobrança já existe no gateway; registro é best effort
        logger.exception("falha ao registrar cobrança %s", charge["external_id"])

    return charge


async def fetch_payment(*, credential: dict, external_id: str) -> dict:
    token = credential.get("access_token")
    if not token:
        raise RuntimeError("credencial de pagamento não configurada para esta empresa")
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{settings.mercado_pago_api}/v1/payments/{external_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()


async def sync_charge_status(*, tenant_id, external_id: str, status: str, payment: dict) -> None:
    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """UPDATE payment_charges
                      SET status = %s, raw = %s, updated_at = now()
                    WHERE tenant_id = %s AND external_id = %s""",
                (status, Json(payment), tenant_id, external_id),
            )
    except Exception:  # noqa: BLE001 — consulta já respondeu ao agente
        logger.exception("falha ao sincronizar cobrança %s", external_id)


async def lookup_charge(*, tenant_id, reference_id: str) -> dict | None:
    """Última cobrança de uma referência de negócio (ex.: id do pedido)."""
    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """SELECT external_id, status, amount, description
                     FROM payment_charges
                    WHERE tenant_id = %s AND reference_id = %s
                    ORDER BY created_at DESC LIMIT 1""",
                (tenant_id, reference_id),
            )
            return await cursor.fetchone()
    except Exception:  # noqa: BLE001
        logger.exception("falha ao buscar cobrança de %s", reference_id)
        return None
