"""Tests del Caché Local Estricto (Singleton + decorador)."""

from __future__ import annotations

import json
import time

from src.cache import CacheManager, cached


def test_singleton_identity() -> None:
    assert CacheManager() is CacheManager()


def test_roundtrip_and_namespacing() -> None:
    cm = CacheManager()
    cm.set("team_metrics", "ARG", {"gf": 15}, ttl_seconds=60)
    assert cm.get("team_metrics", "ARG") == {"gf": 15}
    assert cm.get("news", "ARG") is None  # otro namespace no colisiona


def test_ttl_expiry(monkeypatch) -> None:
    cm = CacheManager()
    cm.set("market", "ARG-FRA", {"p": 0.5}, ttl_seconds=10)
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 11)
    assert cm.get("market", "ARG-FRA") is None


def test_corrupted_entry_is_discarded() -> None:
    cm = CacheManager()
    cm.set("news", "messi", [1, 2], ttl_seconds=60)
    path = cm._path_for("news", "messi")
    path.write_text("{esto no es json", encoding="utf-8")
    assert cm.get("news", "messi") is None
    assert not path.exists()  # la entrada corrupta se purga


def test_envelope_schema_on_disk() -> None:
    cm = CacheManager()
    cm.set("squads", "MEX", ["a", "b"], ttl_seconds=5)
    envelope = json.loads(cm._path_for("squads", "MEX").read_text())
    assert {"version", "written_at", "ttl_seconds", "payload"} <= set(envelope)


def test_cached_decorator_avoids_second_call() -> None:
    calls = {"n": 0}

    class Provider:
        @cached("news", ttl_seconds=60)
        def fetch(self, player: str) -> list[str]:
            calls["n"] += 1
            return [f"articulo-{player}"]

    p = Provider()
    assert p.fetch("Messi") == ["articulo-Messi"]
    assert p.fetch("Messi") == ["articulo-Messi"]
    assert calls["n"] == 1  # segunda lectura: disco, no función


def test_cached_decorator_force_refresh_bypasses() -> None:
    calls = {"n": 0}

    class Provider:
        @cached("market", ttl_seconds=60)
        def fetch(self, key: str) -> dict[str, int]:
            calls["n"] += 1
            return {"v": calls["n"]}

    p = Provider()
    assert p.fetch("x") == {"v": 1}
    assert p.fetch("x", force_refresh=True) == {"v": 2}
    assert calls["n"] == 2


def test_clear_by_namespace() -> None:
    cm = CacheManager()
    cm.set("a", "k1", 1, 60)
    cm.set("b", "k2", 2, 60)
    assert cm.clear("a") == 1
    assert cm.get("b", "k2") == 2
