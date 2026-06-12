"""Configuración central del motor (12-factor: todo sobre-escribible por entorno).

Ningún parámetro estadístico vive aquí: los parámetros del modelo (mu del
torneo, fuerzas de ataque/defensa, covarianza de goles) se ESTIMAN siempre
desde datos en ``src.simulator``. Este módulo solo contiene:

1. Credenciales y endpoints de proveedores (variables de entorno).
2. Parámetros OPERACIONALES (TTLs de caché, timeouts, throttling) cuya
   derivación cuantitativa se documenta junto a cada constante — están
   anclados a las cuotas publicadas de los planes gratuitos, no a gusto.
3. Cotas contractuales fijadas por especificación del producto (rango del
   TIF, mínimo de iteraciones Monte Carlo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - import opcional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = Path(os.getenv("WC26_CACHE_DIR", str(DATA_DIR / "cache")))
FALLBACK_DIR: Path = Path(os.getenv("WC26_FALLBACK_DIR", str(DATA_DIR / "fallback")))

HISTORY_FALLBACK_FILE: Path = FALLBACK_DIR / "world_cup_history.json"
SQUADS_FALLBACK_FILE: Path = FALLBACK_DIR / "squads.json"
MARKET_FALLBACK_FILE: Path = FALLBACK_DIR / "market_baseline.json"

# ---------------------------------------------------------------------------
# Contratos de producto (especificación funcional, no heurísticas)
# ---------------------------------------------------------------------------
#: Rango contractual del Tactical Impact Factor exigido por especificación:
#: multiplicador simétrico en [0.80, 1.20] alrededor del rendimiento neutro.
TIF_LOWER_BOUND: float = 0.80
TIF_UPPER_BOUND: float = 1.20
#: Semiamplitud derivada algebraicamente del contrato anterior:
#: (UPPER - LOWER) / 2 = 0.20 → TIF = 1 + 0.20 * polaridad_ajustada.
TIF_HALF_RANGE: float = (TIF_UPPER_BOUND - TIF_LOWER_BOUND) / 2.0

#: Mínimo contractual de iteraciones Monte Carlo exigido por especificación.
#: Con n=15000 el error estándar máximo de una probabilidad estimada es
#: sqrt(0.25/15000) ≈ 0.41 puntos porcentuales (cota de varianza binomial).
MIN_MONTE_CARLO_SIMS: int = 15_000

#: Masa de cola máxima tolerada al truncar la malla de marcadores para el
#: cálculo analítico de la pmf bivariada. La cota K se busca con la CDF de
#: Poisson hasta que P(X > K) < TAIL_EPSILON; no es un límite de goles fijo.
TAIL_EPSILON: float = 1e-9

# ---------------------------------------------------------------------------
# Parámetros operacionales de red (derivados de límites publicados)
# ---------------------------------------------------------------------------
#: Timeout HTTP. P99 de latencia observada en las APIs usadas < 10 s;
#: 15 s evita falsos ProviderTimeoutError sin colgar la CLI.
HTTP_TIMEOUT_SECONDS: float = float(os.getenv("WC26_HTTP_TIMEOUT", "15"))

#: Reintentos con backoff exponencial base 2 (2s, 4s, 8s) ante errores 5xx.
HTTP_MAX_RETRIES: int = int(os.getenv("WC26_HTTP_RETRIES", "3"))

#: football-data.org plan gratuito publica 10 req/min ⇒ intervalo mínimo
#: entre llamadas = 60/10 = 6 s para no recibir HTTP 429.
FOOTBALL_DATA_MIN_INTERVAL_S: float = 60.0 / 10.0

# ---------------------------------------------------------------------------
# TTLs de caché (en segundos). Derivación documentada por cuota publicada.
# ---------------------------------------------------------------------------
#: NewsAPI.org free = 100 req/día. Peor caso por ejecución completa:
#: 2 equipos × 11 titulares × 2 proveedores = 44 requests. Con TTL = 12 h
#: se garantizan ≤ 2 barridos frescos/día ⇒ 88 req ≤ 100 (margen 12%).
CACHE_TTL_NEWS_S: int = int(os.getenv("WC26_TTL_NEWS", str(12 * 3600)))

#: football-data.org / API-Football: las métricas agregadas de selecciones
#: solo cambian cuando se juega un partido nuevo (frecuencia ≥ 24 h en un
#: Mundial: cada selección juega como máximo 1 partido/día).
CACHE_TTL_TEAM_S: int = int(os.getenv("WC26_TTL_TEAM", str(24 * 3600)))

#: Datos históricos de ediciones cerradas (2014/2018/2022): inmutables.
CACHE_TTL_HISTORY_S: int = int(os.getenv("WC26_TTL_HISTORY", str(365 * 24 * 3600)))

#: Mercado (Polymarket): TTL corto; la bandera --live-odds fuerza bypass.
#: 15 min ≈ frecuencia de actualización de los snapshots públicos de la
#: Gamma API sin autenticación; configurable por entorno.
CACHE_TTL_MARKET_S: int = int(os.getenv("WC26_TTL_MARKET", str(15 * 60)))

# ---------------------------------------------------------------------------
# Parámetros NLP operacionales
# ---------------------------------------------------------------------------
#: Vida media (horas) del decaimiento exponencial de relevancia de noticias.
#: Anclada al ciclo competitivo real de un Mundial: hay partido del mismo
#: equipo cada 96 h como máximo en fase de grupos; media vida = 72 h hace
#: que una noticia del partido anterior pese 0.5 y una de hace dos ciclos
#: pese 0.25 — decaimiento geométrico coherente con el calendario FIFA.
NEWS_HALF_LIFE_HOURS: float = float(os.getenv("WC26_NEWS_HALF_LIFE_H", "72"))

#: Máximo de artículos por jugador y proveedor (pageSize). Cota de cuota:
#: con 10 artículos/jugador el barrido completo cabe en una respuesta HTTP
#: por jugador (1 request), sin paginación extra.
NEWS_PAGE_SIZE: int = int(os.getenv("WC26_NEWS_PAGE_SIZE", "10"))


@dataclass(frozen=True)
class Credentials:
    """Credenciales de proveedores leídas del entorno.

    Attributes:
        football_data_token: Token de api.football-data.org (header X-Auth-Token).
        api_football_key: Key de API-Football (api-sports.io o RapidAPI).
        api_football_host: Host RapidAPI si se usa esa pasarela.
        newsapi_key: API key de newsapi.org.
        gnews_key: API key de gnews.io.
        zafronix_base_url: Endpoint opcional de la API histórica Zafronix.
        zafronix_key: Credencial opcional de Zafronix.
    """

    football_data_token: str | None = field(
        default_factory=lambda: os.getenv("FOOTBALL_DATA_TOKEN"))
    api_football_key: str | None = field(
        default_factory=lambda: os.getenv("API_FOOTBALL_KEY"))
    api_football_host: str = field(
        default_factory=lambda: os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io"))
    newsapi_key: str | None = field(default_factory=lambda: os.getenv("NEWSAPI_KEY"))
    gnews_key: str | None = field(default_factory=lambda: os.getenv("GNEWS_KEY"))
    zafronix_base_url: str | None = field(
        default_factory=lambda: os.getenv("ZAFRONIX_BASE_URL"))
    zafronix_key: str | None = field(default_factory=lambda: os.getenv("ZAFRONIX_KEY"))
    supabase_url: str | None = field(default_factory=lambda: os.getenv("SUPABASE_URL"))
    supabase_key: str | None = field(
        default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY"))


@dataclass(frozen=True)
class Endpoints:
    """Endpoints base de los proveedores soportados."""

    football_data: str = "https://api.football-data.org/v4"
    api_football: str = field(
        default_factory=lambda: "https://" + os.getenv(
            "API_FOOTBALL_HOST", "v3.football.api-sports.io"))
    newsapi: str = "https://newsapi.org/v2"
    gnews: str = "https://gnews.io/api/v4"
    polymarket_gamma: str = "https://gamma-api.polymarket.com"


def get_credentials() -> Credentials:
    """Construye el snapshot inmutable de credenciales desde el entorno.

    Returns:
        Instancia congelada de :class:`Credentials`.
    """
    return Credentials()


def get_endpoints() -> Endpoints:
    """Construye el snapshot inmutable de endpoints.

    Returns:
        Instancia congelada de :class:`Endpoints`.
    """
    return Endpoints()
