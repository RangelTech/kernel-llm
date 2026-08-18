"""Recusa rodar a suíte contra um banco que não seja descartável.

Cópia deliberada de `backend/tests/guardas.py::exigir_banco_descartavel` — o
kernel vira repo próprio (`kernel-llm`) e não pode mais importar do backend
(nem em teste). É a mesma regra de segurança, duplicada de propósito: as duas
cópias protegem bancos diferentes (kernel e backend têm cada um o seu),
divergir aqui não corrompe nada do outro lado.
"""

import os
from urllib.parse import urlparse

HOSTS_DESCARTAVEIS = {"localhost", "127.0.0.1", "::1", "postgres", "db", ""}
ESCAPE = "ALLOW_DESTRUCTIVE_TESTS"


def exigir_banco_descartavel(dsn: str) -> None:
    """Levanta se o DSN não parece ser de teste. Chamado antes de qualquer
    fixture que escreva no banco."""
    if os.environ.get(ESCAPE) == "1":
        return
    host = (urlparse(dsn).hostname or "").lower()
    if host in HOSTS_DESCARTAVEIS:
        return
    raise RuntimeError(
        f"a suíte apaga dados e o DATABASE_URL aponta para '{host}', que não é "
        "um banco local de teste. Se isto é mesmo um banco descartável, rode com "
        f"{ESCAPE}=1. Se é produção, você acabou de evitar apagar os tenants."
    )
