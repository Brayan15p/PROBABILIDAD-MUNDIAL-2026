"""Infraestructura de Caché Local Estricto en JSON (``data/cache/``).

Implementa el patrón **Singleton** (thread-safe) para el gestor y un
**decorador** ``@cached`` que intercepta cualquier método de proveedor:
si la clave ya existe en disco y no expiró su TTL, se lee inmediatamente
del disco y NUNCA se toca la red — requisito crítico para no agotar las
cuotas de los planes gratuitos.

Diseño:
    - Una entrada = un archivo ``data/cache/<namespace>/<sha256>.json``.
    - Envelope con metadatos: timestamp de escritura, TTL, clave humana.
    - Corrupción tolerada: una entrada ilegible se descarta y regenera
      (se loggea ``CacheCorruptionError`` sin abortar el pipeline).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from src import config
from src.exceptions import CacheCorruptionError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

#: Versión del esquema del envelope; invalida cachés viejas si cambia.
_ENVELOPE_VERSION = 1


class SingletonMeta(type):
    """Metaclase Singleton con doble verificación de bloqueo (thread-safe)."""

    _instances: dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:  # double-checked locking
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class CacheManager(metaclass=SingletonMeta):
    """Gestor único de caché JSON en disco.

    Attributes:
        base_dir: Directorio raíz de la caché (``data/cache/`` por defecto).
        hits: Contador de lecturas servidas desde disco en este proceso.
        misses: Contador de fallos de caché (red/fallback requeridos).
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path = base_dir or config.CACHE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hits: int = 0
        self.misses: int = 0
        self._io_lock = threading.Lock()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def get(self, namespace: str, key: str) -> Any | None:
        """Lee una entrada vigente de la caché.

        Args:
            namespace: Agrupador lógico (p. ej. ``team_metrics``, ``news``).
            key: Clave humana determinista (se hashea para el nombre de archivo).

        Returns:
            El payload deserializado si existe y su TTL no expiró; ``None``
            en cualquier otro caso (ausente, expirado o corrupto).
        """
        path = self._path_for(namespace, key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            envelope = self._read_envelope(path)
        except CacheCorruptionError as exc:
            logger.warning("Cache corrupta descartada (%s): %s", path.name, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

        age = time.time() - envelope["written_at"]
        if age > envelope["ttl_seconds"]:
            self.misses += 1
            return None
        self.hits += 1
        return envelope["payload"]

    def set(self, namespace: str, key: str, payload: Any, ttl_seconds: int) -> None:
        """Persiste una entrada con su TTL en un archivo JSON atómico.

        Args:
            namespace: Agrupador lógico.
            key: Clave humana determinista.
            payload: Objeto JSON-serializable a guardar.
            ttl_seconds: Vigencia en segundos antes de considerarse rancia.
        """
        path = self._path_for(namespace, key)
        envelope = {
            "version": _ENVELOPE_VERSION,
            "namespace": namespace,
            "key": key,
            "written_at": time.time(),
            "ttl_seconds": ttl_seconds,
            "payload": payload,
        }
        tmp = path.with_suffix(".tmp")
        with self._io_lock:
            tmp.write_text(json.dumps(envelope, ensure_ascii=False, default=str),
                           encoding="utf-8")
            tmp.replace(path)  # rename atómico en POSIX

    def invalidate(self, namespace: str, key: str) -> bool:
        """Elimina una entrada específica.

        Args:
            namespace: Agrupador lógico.
            key: Clave humana.

        Returns:
            True si existía y fue eliminada.
        """
        path = self._path_for(namespace, key)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    def clear(self, namespace: str | None = None) -> int:
        """Vacía la caché completa o un namespace.

        Args:
            namespace: Si se indica, solo borra ese subdirectorio.

        Returns:
            Número de archivos eliminados.
        """
        target = self.base_dir / namespace if namespace else self.base_dir
        if not target.exists():
            return 0
        count = 0
        for f in target.rglob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """Resume el estado de la caché para observabilidad de la CLI.

        Returns:
            Dict con hits/misses del proceso y entradas por namespace.
        """
        per_ns: dict[str, int] = {}
        for sub in sorted(self.base_dir.iterdir()) if self.base_dir.exists() else []:
            if sub.is_dir():
                per_ns[sub.name] = sum(1 for _ in sub.glob("*.json"))
        return {"hits": self.hits, "misses": self.misses,
                "entries_by_namespace": per_ns,
                "base_dir": str(self.base_dir)}

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _path_for(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        ns_dir = self.base_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{digest}.json"

    @staticmethod
    def _read_envelope(path: Path) -> dict[str, Any]:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CacheCorruptionError(str(exc)) from exc
        required = {"version", "written_at", "ttl_seconds", "payload"}
        if not required.issubset(envelope):
            raise CacheCorruptionError(
                f"Envelope inválido, faltan claves: {required - set(envelope)}")
        if envelope["version"] != _ENVELOPE_VERSION:
            raise CacheCorruptionError(
                f"Versión de envelope incompatible: {envelope['version']}")
        return envelope


def cached(namespace: str, ttl_seconds: int,
           key_fn: Callable[..., str] | None = None,
           bypass_kwarg: str = "force_refresh") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorador de Caché Local Estricto para métodos de proveedores.

    Política: *cache-first*. Si la clave existe y está vigente se devuelve
    el disco sin tocar la red. El kwarg ``force_refresh=True`` (consumido
    por el decorador) permite el bypass explícito, usado por ``--live-odds``.

    Args:
        namespace: Namespace de la caché donde persistir.
        ttl_seconds: TTL de las entradas generadas por la función.
        key_fn: Constructor de clave a partir de ``(*args, **kwargs)``.
            Por defecto: nombre de la función + repr posicional/nombrado.
        bypass_kwarg: Nombre del kwarg booleano que fuerza refresco.

    Returns:
        Decorador que envuelve la función con la política cache-first.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            force = bool(kwargs.pop(bypass_kwarg, False))
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # args[0] suele ser self: se excluye su identidad de la clave.
                positional = args[1:] if args and hasattr(args[0], "__dict__") else args
                key = f"{func.__qualname__}|{positional!r}|{sorted(kwargs.items())!r}"
            manager = CacheManager()
            if not force:
                hit = manager.get(namespace, key)
                if hit is not None:
                    return hit  # type: ignore[return-value]
            result = func(*args, **kwargs)
            if result is not None:
                manager.set(namespace, key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
