import pytest
from pydantic import ValidationError

from app.config import Settings


def make(**kw):
    return Settings(_env_file=None, **kw)


def test_prod_sem_internal_token_recusa_subir():
    with pytest.raises(ValidationError, match="INTERNAL_TOKEN"):
        make(environment="prod", internal_token="")


def test_prod_aceita_alias_production_e_exige_token():
    with pytest.raises(ValidationError):
        make(environment="production", internal_token="  ")


def test_prod_via_env_rt_env(monkeypatch):
    monkeypatch.setenv("RT_ENV", "prod")
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("INTERNAL_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        make()


def test_prod_com_token_sobe_e_desliga_stub_control(monkeypatch):
    monkeypatch.delenv("ENABLE_STUB_CONTROL", raising=False)
    s = make(environment="prod", internal_token="x")
    assert s.enable_stub_control is False


def test_prod_permite_ligar_stub_control_explicitamente():
    s = make(environment="prod", internal_token="x", enable_stub_control=True)
    assert s.enable_stub_control is True


def test_dev_mantem_stub_control_ligado_e_token_opcional(monkeypatch):
    for k in ("ENVIRONMENT", "RT_ENV", "INTERNAL_TOKEN", "ENABLE_STUB_CONTROL"):
        monkeypatch.delenv(k, raising=False)
    s = make()
    assert s.enable_stub_control is True
    assert s.internal_token == ""
