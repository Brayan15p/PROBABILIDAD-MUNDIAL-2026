"""Jerarquía de excepciones controladas del motor predictivo.

Toda falla de infraestructura (red, autenticación, cuota, datos corruptos)
se traduce a una excepción de esta jerarquía. Los componentes superiores
capturan ``ProviderError`` para activar la degradación elegante hacia el
``FallbackDataStore`` sin abortar el pipeline.
"""

from __future__ import annotations


class WorldCupEngineError(Exception):
    """Raíz de todas las excepciones del motor."""


class ConfigurationError(WorldCupEngineError):
    """Configuración inválida o credenciales ausentes para una operación."""


class ProviderError(WorldCupEngineError):
    """Falla genérica de un proveedor externo de datos.

    Attributes:
        provider: Nombre lógico del proveedor que falló.
        status_code: Código HTTP devuelto, si aplica.
    """

    def __init__(self, message: str, provider: str = "unknown",
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class ProviderAuthError(ProviderError):
    """Token inválido, expirado o ausente (HTTP 401/403)."""


class ProviderQuotaError(ProviderError):
    """Cuota del plan gratuito agotada o rate-limit excedido (HTTP 429)."""


class ProviderTimeoutError(ProviderError):
    """El proveedor no respondió dentro del timeout configurado."""


class DataUnavailableError(WorldCupEngineError):
    """Ni el proveedor remoto ni el Fallback Data Store tienen el dato."""


class CacheCorruptionError(WorldCupEngineError):
    """Una entrada de caché en disco no es JSON válido o viola su esquema."""


class TeamResolutionError(WorldCupEngineError):
    """El nombre de equipo recibido no se pudo resolver a un código FIFA.

    Attributes:
        query: Texto original que falló la resolución.
        suggestions: Códigos/nombres candidatos más cercanos.
    """

    def __init__(self, query: str, suggestions: list[str] | None = None) -> None:
        self.query = query
        self.suggestions = suggestions or []
        hint = f" Sugerencias: {', '.join(self.suggestions)}" if self.suggestions else ""
        super().__init__(f"Equipo no reconocido: '{query}'.{hint}")


class SimulationError(WorldCupEngineError):
    """Parámetros de simulación inválidos (lambdas no positivos, sims < mínimo)."""
