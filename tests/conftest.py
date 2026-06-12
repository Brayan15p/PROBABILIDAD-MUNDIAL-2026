"""Fixtures compartidas: aísla caché y credenciales en cada test."""

from __future__ import annotations

import pytest

from src.cache import CacheManager, SingletonMeta

_CREDENTIAL_VARS = (
    "FOOTBALL_DATA_TOKEN", "API_FOOTBALL_KEY", "NEWSAPI_KEY", "GNEWS_KEY",
    "ZAFRONIX_BASE_URL", "ZAFRONIX_KEY", "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "WC26_MARKET_WEIGHT",
)


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Caché en tmpdir + entorno sin credenciales (cero red en tests)."""
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    SingletonMeta._instances.pop(CacheManager, None)
    CacheManager(base_dir=tmp_path / "cache")
    yield
    SingletonMeta._instances.pop(CacheManager, None)
