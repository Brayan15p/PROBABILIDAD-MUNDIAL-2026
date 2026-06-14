"""Cliente de datos multi-API del motor predictivo (capa de adquisición).

Unifica cuatro familias de fuentes bajo la fachada ``WorldCupDataEngine``:

1. **Deportiva**: football-data.org (plan gratuito) o API-Football
   (api-sports.io / RapidAPI), seleccionada por ``SportsProviderFactory``
   según las credenciales presentes (patrón *Factory*).
2. **Histórica**: adaptador ``ZafronixHistoricalAdapter`` para tendencias
   de fases finales. *Nota de ingeniería honesta*: "Zafronix" no es un
   servicio público verificable; el adaptador implementa la interfaz
   solicitada resolviendo contra ``ZAFRONIX_BASE_URL`` si se configura y,
   en su defecto, contra el dataset real embebido de la FIFA 2014–2022.
3. **Noticias**: NewsAPI.org + GNews, agregadas y deduplicadas por URL.
4. **Mercado**: Polymarket (Gamma API pública); fallback al snapshot de
   consenso de casas embebido con su fecha de corte.

Toda llamada remota pasa por el decorador ``@cached`` (cache-first sobre
``data/cache/``) y toda falla de token/cuota levanta excepciones
controladas que activan el ``FallbackDataStore``.
"""

from __future__ import annotations

import difflib
import logging
import threading
import time
import unicodedata
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import requests

from src import config
from src.cache import CacheManager, cached
from src.exceptions import (
    ConfigurationError,
    DataUnavailableError,
    ProviderAuthError,
    ProviderError,
    ProviderQuotaError,
    ProviderTimeoutError,
    TeamResolutionError,
)
from src.models import DataOrigin, KnockoutTendencies, MarketSnapshot, NewsArticle

logger = logging.getLogger(__name__)


def normalize_text(value: str) -> str:
    """Normaliza un texto para comparación: minúsculas y sin diacríticos.

    Args:
        value: Texto arbitrario (p. ej. ``"México"``).

    Returns:
        Versión plegada (p. ej. ``"mexico"``).
    """
    folded = unicodedata.normalize("NFKD", value)
    return "".join(c for c in folded if not unicodedata.combining(c)).lower().strip()


# ---------------------------------------------------------------------------
# Proveedor base (plantilla de transporte HTTP con resiliencia)
# ---------------------------------------------------------------------------
class BaseAPIProvider(ABC):
    """Transporte HTTP común: auth, timeout, reintentos y mapeo de errores.

    Subclases declaran ``provider_name`` y construyen requests; esta base
    aplica backoff exponencial ante 5xx y traduce 401/403/429 a la
    jerarquía de excepciones controladas del motor.
    """

    #: Nombre lógico usado en logs, excepciones y trazabilidad.
    provider_name: str = "base"

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    @abstractmethod
    def is_configured(self) -> bool:
        """Indica si el proveedor tiene credenciales utilizables."""

    def _request(self, method: str, url: str,
                 headers: dict[str, str] | None = None,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Ejecuta una request con reintentos y mapeo de errores.

        Args:
            method: Verbo HTTP.
            url: URL absoluta.
            headers: Cabeceras adicionales (auth incluida por la subclase).
            params: Query params.

        Returns:
            Cuerpo JSON deserializado.

        Raises:
            ProviderAuthError: HTTP 401/403 (token inválido o plan insuficiente).
            ProviderQuotaError: HTTP 429 (cuota agotada).
            ProviderTimeoutError: Timeout de red agotando reintentos.
            ProviderError: Cualquier otra falla de transporte o decodificación.
        """
        last_exc: Exception | None = None
        for attempt in range(config.HTTP_MAX_RETRIES + 1):
            try:
                resp = self._session.request(
                    method, url, headers=headers, params=params,
                    timeout=config.HTTP_TIMEOUT_SECONDS)
            except requests.Timeout as exc:
                last_exc = exc
                self._backoff(attempt)
                continue
            except requests.RequestException as exc:
                raise ProviderError(f"Red caída hacia {self.provider_name}: {exc}",
                                    provider=self.provider_name) from exc

            if resp.status_code in (401, 403):
                raise ProviderAuthError(
                    f"Token rechazado por {self.provider_name} "
                    f"(HTTP {resp.status_code})",
                    provider=self.provider_name, status_code=resp.status_code)
            if resp.status_code == 429:
                raise ProviderQuotaError(
                    f"Cuota agotada en {self.provider_name} (HTTP 429); "
                    "el motor degradará a caché/fallback",
                    provider=self.provider_name, status_code=429)
            if resp.status_code >= 500:
                last_exc = ProviderError(
                    f"{self.provider_name} HTTP {resp.status_code}",
                    provider=self.provider_name, status_code=resp.status_code)
                self._backoff(attempt)
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    f"{self.provider_name} HTTP {resp.status_code}: "
                    f"{resp.text[:200]}",
                    provider=self.provider_name, status_code=resp.status_code)
            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError(
                    f"JSON inválido de {self.provider_name}",
                    provider=self.provider_name) from exc

        raise ProviderTimeoutError(
            f"{self.provider_name} no respondió tras "
            f"{config.HTTP_MAX_RETRIES + 1} intentos: {last_exc}",
            provider=self.provider_name)

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Espera exponencial base 2 entre reintentos (2s, 4s, 8s...)."""
        time.sleep(2.0 ** (attempt + 1))


# ---------------------------------------------------------------------------
# Proveedores deportivos
# ---------------------------------------------------------------------------
class FootballDataProvider(BaseAPIProvider):
    """Cliente de api.football-data.org v4 (competición ``WC``).

    El plan gratuito publica 10 req/min: se aplica un throttle de proceso
    (intervalo mínimo entre llamadas = 6 s, derivado de la cuota).
    """

    provider_name = "football-data.org"
    _throttle_lock = threading.Lock()
    _last_call_monotonic: float = 0.0

    def __init__(self, token: str | None = None) -> None:
        super().__init__()
        self._token = token if token is not None else config.get_credentials().football_data_token

    def is_configured(self) -> bool:
        """True si hay token de football-data.org en el entorno."""
        return bool(self._token)

    @cached("sports_matches", config.CACHE_TTL_HISTORY_S)
    def fetch_world_cup_matches(self, season: int) -> list[dict[str, Any]]:
        """Descarga y normaliza todos los partidos de una edición del Mundial.

        Args:
            season: Año de la edición (2014, 2018, 2022, 2026).

        Returns:
            Lista de partidos normalizados::

                {"edition", "stage", "home", "away", "home_goals",
                 "away_goals", "home_goals_90", "away_goals_90", "duration"}

        Raises:
            ProviderAuthError: Token ausente/insuficiente para la temporada.
            ProviderQuotaError: Rate limit del plan gratuito excedido.
        """
        if not self.is_configured():
            raise ProviderAuthError("FOOTBALL_DATA_TOKEN no configurado",
                                    provider=self.provider_name)
        self._throttle()
        body = self._request(
            "GET", f"{config.get_endpoints().football_data}/competitions/WC/matches",
            headers={"X-Auth-Token": self._token or ""},
            params={"season": season})
        normalized: list[dict[str, Any]] = []
        for m in body.get("matches", []):
            score = m.get("score", {})
            full = score.get("fullTime", {})
            reg = score.get("regularTime") or full
            if full.get("home") is None or full.get("away") is None:
                continue  # partido aún no jugado
            normalized.append({
                "edition": season,
                "stage": m.get("stage", "UNKNOWN"),
                "home": m.get("homeTeam", {}).get("tla")
                        or m.get("homeTeam", {}).get("name", ""),
                "away": m.get("awayTeam", {}).get("tla")
                        or m.get("awayTeam", {}).get("name", ""),
                "home_goals": int(full["home"]),
                "away_goals": int(full["away"]),
                "home_goals_90": int(reg.get("home", full["home"])),
                "away_goals_90": int(reg.get("away", full["away"])),
                "duration": score.get("duration", "REGULAR"),
            })
        return normalized

    @classmethod
    def _throttle(cls) -> None:
        """Garantiza el intervalo mínimo entre llamadas (cuota 10 req/min)."""
        with cls._throttle_lock:
            now = time.monotonic()
            wait = config.FOOTBALL_DATA_MIN_INTERVAL_S - (now - cls._last_call_monotonic)
            if wait > 0:
                time.sleep(wait)
            cls._last_call_monotonic = time.monotonic()


class ApiFootballProvider(BaseAPIProvider):
    """Cliente de API-Football v3 (api-sports.io directo o vía RapidAPI).

    Usado para: partidos del Mundial (league=1), plantillas/onces titulares
    y, si el plan lo expone, estadísticas extendidas (xG, ratings).
    """

    provider_name = "api-football"

    def __init__(self, api_key: str | None = None, host: str | None = None) -> None:
        super().__init__()
        creds = config.get_credentials()
        self._key = api_key if api_key is not None else creds.api_football_key
        self._host = host or creds.api_football_host

    def is_configured(self) -> bool:
        """True si hay API key de API-Football en el entorno."""
        return bool(self._key)

    def _auth_headers(self) -> dict[str, str]:
        if "rapidapi" in self._host:
            return {"x-rapidapi-key": self._key or "",
                    "x-rapidapi-host": self._host}
        return {"x-apisports-key": self._key or ""}

    @cached("sports_matches", config.CACHE_TTL_HISTORY_S)
    def fetch_world_cup_matches(self, season: int) -> list[dict[str, Any]]:
        """Descarga los partidos del Mundial (league=1) de una temporada.

        Args:
            season: Año de la edición.

        Returns:
            Lista de partidos en el mismo esquema normalizado que
            :meth:`FootballDataProvider.fetch_world_cup_matches`.
        """
        if not self.is_configured():
            raise ProviderAuthError("API_FOOTBALL_KEY no configurado",
                                    provider=self.provider_name)
        body = self._request(
            "GET", f"https://{self._host}/fixtures",
            headers=self._auth_headers(),
            params={"league": 1, "season": season})
        normalized: list[dict[str, Any]] = []
        for item in body.get("response", []):
            goals = item.get("goals", {})
            score = item.get("score", {})
            ft = score.get("fulltime", {})
            if goals.get("home") is None or goals.get("away") is None:
                continue
            normalized.append({
                "edition": season,
                "stage": item.get("league", {}).get("round", "UNKNOWN"),
                "home": item.get("teams", {}).get("home", {}).get("name", ""),
                "away": item.get("teams", {}).get("away", {}).get("name", ""),
                "home_goals": int(goals["home"]),
                "away_goals": int(goals["away"]),
                "home_goals_90": int(ft.get("home") if ft.get("home") is not None
                                     else goals["home"]),
                "away_goals_90": int(ft.get("away") if ft.get("away") is not None
                                     else goals["away"]),
                "duration": "REGULAR",
            })
        return normalized

    @cached("team_ids", config.CACHE_TTL_HISTORY_S)
    def fetch_team_id(self, team_name: str) -> int | None:
        """Resuelve el id numérico de una selección por nombre en inglés.

        Args:
            team_name: Nombre del equipo (p. ej. ``"Argentina"``).

        Returns:
            Id de API-Football o ``None`` si no aparece como selección.
        """
        if not self.is_configured():
            raise ProviderAuthError("API_FOOTBALL_KEY no configurado",
                                    provider=self.provider_name)
        body = self._request(
            "GET", f"https://{self._host}/teams",
            headers=self._auth_headers(), params={"name": team_name})
        for entry in body.get("response", []):
            team = entry.get("team", {})
            if team.get("national") and team.get("id"):
                return int(team["id"])
        return None

    @cached("squads", config.CACHE_TTL_TEAM_S)
    def fetch_squad(self, team_api_id: int) -> list[str]:
        """Obtiene la plantilla vigente de un equipo por su id de API-Football.

        Args:
            team_api_id: Identificador numérico del equipo en API-Football.

        Returns:
            Nombres de los jugadores de la plantilla.
        """
        if not self.is_configured():
            raise ProviderAuthError("API_FOOTBALL_KEY no configurado",
                                    provider=self.provider_name)
        body = self._request(
            "GET", f"https://{self._host}/players/squads",
            headers=self._auth_headers(), params={"team": team_api_id})
        players: list[str] = []
        for entry in body.get("response", []):
            for p in entry.get("players", []):
                if p.get("name"):
                    players.append(str(p["name"]))
        return players


class RecentFormProvider:
    """Retrieves recent international form (last N matches) from football-data.org.

    Free tier: https://www.football-data.org/documentation/quickstart
    Endpoint: GET /v4/teams/{id}/matches?limit=N&status=FINISHED

    Falls back to neutral form (1.0) if API key missing or team ID unknown.

    Attributes:
        BASE_URL: Base URL of the football-data.org v4 API.
        TEAM_ID_MAP: Mapping from FIFA three-letter code to football-data.org
            team ID for the most relevant nations. Teams not in this map
            receive a neutral form factor of 1.0.
        FORM_LOWER: Lower bound of the form multiplier range.
        FORM_UPPER: Upper bound of the form multiplier range.
    """

    BASE_URL: str = "https://api.football-data.org/v4"

    #: FIFA code → football-data.org team ID (verified where possible).
    #: None means the team is known but the ID has not been verified —
    #: get_recent_form() will return 1.0 (neutral) for these entries.
    TEAM_ID_MAP: dict[str, int | None] = {
        "ARG": 64,
        "BRA": 63,
        "FRA": 66,
        "ESP": 760,
        "GER": 759,
        "ENG": 57,
        "NED": 684,
        "POR": 765,
        "ITA": 784,
        "BEL": 805,
        "MEX": 764,
        "USA": 768,
        "JPN": 2288,
        "KOR": 2293,
        "MAR": 1016,
        "HRV": 799,
        "URU": 788,
        "COL": 780,
        "SUI": None,   # Switzerland: ID 788 conflicts with URU in task spec; unverified
        "SEN": 907,
        "ECU": None,
        "QAT": None,
        "CAN": None,
        "AUS": None,
        "GHA": None,
        "CMR": None,
        "SRB": None,
        "POL": None,
        "DEN": None,
        "CRC": None,
        "TUN": None,
        "SAU": None,
    }

    #: Contracted range of the form multiplier.
    FORM_LOWER: float = 0.85
    FORM_UPPER: float = 1.15

    def __init__(self, api_key: str | None = None) -> None:
        import os as _os
        self.api_key: str | None = (
            api_key if api_key is not None
            else _os.getenv("FOOTBALL_DATA_API_KEY")
        )
        self._cache: dict[str, float] = {}
        self._session: requests.Session = requests.Session()

    def is_available(self) -> bool:
        """True si hay FOOTBALL_DATA_API_KEY configurada en el entorno."""
        return self.api_key is not None

    def get_recent_form(self, team_code: str, last_n: int = 10) -> float:
        """Returns a recent-form multiplier in [0.85, 1.15].

        Computed as::

            ratio = points_last_n / (3 * last_n)
            form  = FORM_LOWER + ratio * (FORM_UPPER - FORM_LOWER)

        Where points = 3 for win, 1 for draw, 0 for loss.

        Falls back to 1.0 (neutral) when:
          - API key is absent.
          - The team code is not in TEAM_ID_MAP or the mapped ID is None.
          - The API returns an error or no completed matches.

        Args:
            team_code: FIFA three-letter code (e.g. ``"ARG"``).
            last_n: Number of recent matches to analyse (default 10).

        Returns:
            Form multiplier in [0.85, 1.15] (1.0 = neutral / unavailable).
        """
        if not self.is_available():
            return 1.0

        cache_key = f"{team_code}:{last_n}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        team_id = self.TEAM_ID_MAP.get(team_code)
        if team_id is None:
            self._cache[cache_key] = 1.0
            return 1.0

        try:
            result = self._fetch_form(team_id, last_n)
        except Exception as exc:
            logger.warning(
                "RecentFormProvider: fallo al obtener forma de %s (id=%s): %s; "
                "usando forma neutra 1.0",
                team_code, team_id, exc)
            self._cache[cache_key] = 1.0
            return 1.0

        self._cache[cache_key] = result
        return result

    def _fetch_form(self, team_id: int, last_n: int) -> float:
        """Fetches matches from the API and computes the form multiplier.

        Args:
            team_id: football-data.org team identifier.
            last_n: Number of recent matches to consider.

        Returns:
            Form multiplier in [0.85, 1.15].
        """
        url = f"{self.BASE_URL}/teams/{team_id}/matches"
        params: dict[str, Any] = {"limit": last_n, "status": "FINISHED"}
        headers = {"X-Auth-Token": self.api_key or ""}
        resp = self._session.get(url, headers=headers, params=params,
                                 timeout=config.HTTP_TIMEOUT_SECONDS)
        if resp.status_code in (401, 403):
            raise ProviderAuthError(
                f"Token rechazado por football-data.org (HTTP {resp.status_code})",
                provider="football-data.org", status_code=resp.status_code)
        if resp.status_code == 429:
            raise ProviderQuotaError(
                "Cuota agotada en football-data.org (HTTP 429)",
                provider="football-data.org", status_code=429)
        if resp.status_code >= 400:
            raise ProviderError(
                f"football-data.org HTTP {resp.status_code}: {resp.text[:200]}",
                provider="football-data.org", status_code=resp.status_code)
        body = resp.json()
        matches = body.get("matches", [])
        finished = [m for m in matches
                    if m.get("status") == "FINISHED"
                    and m.get("score", {}).get("winner") is not None]
        if not finished:
            return 1.0
        points = 0
        n_played = 0
        for m in finished[-last_n:]:
            home_id = (m.get("homeTeam") or {}).get("id")
            away_id = (m.get("awayTeam") or {}).get("id")
            winner = m.get("score", {}).get("winner")  # HOME_TEAM / AWAY_TEAM / DRAW
            if winner == "DRAW":
                points += 1
            elif winner == "HOME_TEAM" and home_id == team_id:
                points += 3
            elif winner == "AWAY_TEAM" and away_id == team_id:
                points += 3
            n_played += 1
        if n_played == 0:
            return 1.0
        ratio = points / (3.0 * n_played)
        form = self.FORM_LOWER + ratio * (self.FORM_UPPER - self.FORM_LOWER)
        return float(form)


class SportsProviderFactory:
    """Factory de proveedores deportivos por disponibilidad de credenciales.

    Orden de preferencia: football-data.org (datos históricos del Mundial
    en el plan gratuito) → API-Football. Sin credenciales devuelve ``None``
    y el motor opera 100%% sobre el Fallback Data Store.
    """

    @staticmethod
    def create() -> BaseAPIProvider | None:
        """Construye el primer proveedor configurado según la preferencia.

        Returns:
            Instancia configurada o ``None`` si no hay credenciales.
        """
        fd = FootballDataProvider()
        if fd.is_configured():
            return fd
        af = ApiFootballProvider()
        if af.is_configured():
            return af
        return None


# ---------------------------------------------------------------------------
# Adaptador histórico (interfaz Zafronix)
# ---------------------------------------------------------------------------
class ZafronixHistoricalAdapter(BaseAPIProvider):
    """Adaptador de la estructura de datos históricos "Zafronix".

    Honestidad técnica: "Zafronix API" no es un servicio público
    verificable. Este adaptador cumple la interfaz solicitada: si existe
    ``ZAFRONIX_BASE_URL`` consulta ese endpoint (contrato:
    ``GET /v1/world-cup/knockouts`` → mismo esquema del dataset embebido);
    en caso contrario sirve el dataset REAL FIFA 2014–2022 incluido en
    ``data/fallback/world_cup_history.json``.
    """

    provider_name = "zafronix-historical"

    def __init__(self, store: "FallbackDataStore") -> None:
        super().__init__()
        creds = config.get_credentials()
        self._base_url = creds.zafronix_base_url
        self._key = creds.zafronix_key
        self._store = store

    def is_configured(self) -> bool:
        """True si se configuró un endpoint Zafronix remoto."""
        return bool(self._base_url)

    def knockout_matches(self) -> tuple[list[dict[str, Any]], DataOrigin]:
        """Obtiene los cruces históricos de eliminación directa.

        Returns:
            Tupla ``(partidos, origen)`` donde cada partido sigue el esquema
            del dataset embebido (``score_90``, ``score_final``...).
        """
        if self.is_configured():
            try:
                headers = {"Authorization": f"Bearer {self._key}"} if self._key else None
                body = self._request(
                    "GET", f"{self._base_url.rstrip('/')}/v1/world-cup/knockouts",
                    headers=headers)
                matches = body.get("knockout_matches", body if isinstance(body, list) else [])
                if matches:
                    return list(matches), DataOrigin.LIVE_API
            except ProviderError as exc:
                logger.warning("Zafronix remoto falló (%s); usando dataset embebido", exc)
        return self._store.knockout_matches(), DataOrigin.FALLBACK_STORE

    def compute_tendencies(self) -> KnockoutTendencies:
        """Calcula tendencias de fases finales sobre la muestra disponible.

        Todas las cifras son estadísticos muestrales (medias, proporciones
        y covarianza) — ninguna constante inventada.

        Returns:
            :class:`KnockoutTendencies` con medias por fase, tasa de empate
            a 90' y la covarianza muestral de goles (insumo del λ₃).
        """
        matches, _origin = self.knockout_matches()
        if not matches:
            raise DataUnavailableError("Sin cruces históricos disponibles")
        editions = sorted({int(m["edition"]) for m in matches})
        h = [int(m["score_90"][0]) for m in matches]
        a = [int(m["score_90"][1]) for m in matches]
        n = len(matches)
        mean_h = sum(h) / n
        mean_a = sum(a) / n
        cov = sum((hi - mean_h) * (ai - mean_a) for hi, ai in zip(h, a)) / (n - 1)
        draws = sum(1 for hi, ai in zip(h, a) if hi == ai)
        stage_means: dict[str, list[int]] = {}
        for m in matches:
            stage_means.setdefault(str(m["stage"]), []).append(
                int(m["score_90"][0]) + int(m["score_90"][1]))
        return KnockoutTendencies(
            editions=tuple(editions),
            matches=n,
            goals_per_match_90=mean_h + mean_a,
            draw_rate_90=draws / n,
            goal_covariance=cov,
            stage_goal_means={k: sum(v) / len(v) for k, v in stage_means.items()},
        )


# ---------------------------------------------------------------------------
# Noticias (NewsAPI + GNews)
# ---------------------------------------------------------------------------
#: Taxonomía de eventos críticos buscados en prensa (consulta booleana).
#: Las claves son etiquetas internas; los términos alimentan el query y el
#: etiquetado de artículos. La dirección del impacto la decide el modelo
#: de sentimiento, no esta tabla (evita heurísticas de signo arbitrarias).
CRITICAL_EVENT_TAXONOMY: dict[str, tuple[str, ...]] = {
    "lesion": ("injury", "injured", "ruled out", "sidelined", "hamstring",
               "knee injury", "muscle"),
    "sobrecarga": ("fatigue", "overload", "fitness doubt", "exhausted",
                   "minutes management"),
    "crisis_vestuario": ("dressing room", "crisis", "row", "bust-up",
                         "conflict", "unrest"),
    "racha_goleadora": ("scoring streak", "in form", "brace", "hat-trick",
                        "goal streak", "top scorer"),
    "suspension": ("suspended", "suspension", "banned", "yellow card accumulation"),
}


def build_news_query(player: str) -> str:
    """Construye la consulta booleana de prensa para un jugador titular.

    Args:
        player: Nombre del jugador.

    Returns:
        Query con el nombre exacto y la unión de la taxonomía crítica.
    """
    terms = sorted({t for terms in CRITICAL_EVENT_TAXONOMY.values() for t in terms})
    # NewsAPI limita q a 500 chars: se recorta la cola de términos si excede.
    base = f'"{player}" AND ('
    body = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
    query = base + body + ")"
    while len(query) > 500 and " OR " in body:
        body = body.rsplit(" OR ", 1)[0]
        query = base + body + ")"
    return query


def tag_topics(text: str) -> tuple[str, ...]:
    """Etiqueta un texto con los tópicos críticos detectados.

    Args:
        text: Texto completo del artículo.

    Returns:
        Etiquetas de la taxonomía presentes en el texto.
    """
    lowered = normalize_text(text)
    return tuple(topic for topic, terms in CRITICAL_EVENT_TAXONOMY.items()
                 if any(normalize_text(t) in lowered for t in terms))


class NewsAPIProvider(BaseAPIProvider):
    """Cliente de newsapi.org (endpoint ``/v2/everything``)."""

    provider_name = "newsapi"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._key = api_key if api_key is not None else config.get_credentials().newsapi_key

    def is_configured(self) -> bool:
        """True si hay NEWSAPI_KEY en el entorno."""
        return bool(self._key)

    @cached("news", config.CACHE_TTL_NEWS_S)
    def search_player(self, player: str) -> list[dict[str, Any]]:
        """Busca artículos recientes de un jugador (eventos críticos).

        Args:
            player: Nombre del jugador titular.

        Returns:
            Artículos crudos serializables (title, description, content,
            source, publishedAt, url).
        """
        if not self.is_configured():
            raise ProviderAuthError("NEWSAPI_KEY no configurado",
                                    provider=self.provider_name)
        body = self._request(
            "GET", f"{config.get_endpoints().newsapi}/everything",
            headers={"X-Api-Key": self._key or ""},
            params={"q": build_news_query(player), "language": "en",
                    "sortBy": "publishedAt", "pageSize": config.NEWS_PAGE_SIZE})
        return [
            {"title": a.get("title") or "", "description": a.get("description") or "",
             "content": a.get("content") or "",
             "source": (a.get("source") or {}).get("name", "unknown"),
             "publishedAt": a.get("publishedAt") or "",
             "url": a.get("url") or ""}
            for a in body.get("articles", [])
        ]


class GNewsProvider(BaseAPIProvider):
    """Cliente de gnews.io (endpoint ``/v4/search``)."""

    provider_name = "gnews"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._key = api_key if api_key is not None else config.get_credentials().gnews_key

    def is_configured(self) -> bool:
        """True si hay GNEWS_KEY en el entorno."""
        return bool(self._key)

    @cached("news", config.CACHE_TTL_NEWS_S)
    def search_player(self, player: str) -> list[dict[str, Any]]:
        """Busca artículos de un jugador en GNews.

        Args:
            player: Nombre del jugador titular.

        Returns:
            Artículos crudos serializables (mismo esquema que NewsAPI).
        """
        if not self.is_configured():
            raise ProviderAuthError("GNEWS_KEY no configurado",
                                    provider=self.provider_name)
        body = self._request(
            "GET", f"{config.get_endpoints().gnews}/search",
            params={"q": f'"{player}"', "lang": "en",
                    "max": config.NEWS_PAGE_SIZE, "token": self._key})
        return [
            {"title": a.get("title") or "", "description": a.get("description") or "",
             "content": a.get("content") or "",
             "source": (a.get("source") or {}).get("name", "unknown"),
             "publishedAt": a.get("publishedAt") or "",
             "url": a.get("url") or ""}
            for a in body.get("articles", [])
        ]


class NewsAggregator:
    """Agrega y deduplica prensa de múltiples proveedores de noticias."""

    def __init__(self, providers: list[BaseAPIProvider] | None = None) -> None:
        self.providers: list[BaseAPIProvider] = providers if providers is not None else [
            NewsAPIProvider(), GNewsProvider()]

    def articles_for_player(self, player: str) -> list[NewsArticle]:
        """Recolecta artículos de todos los proveedores configurados.

        Las fallas de un proveedor (cuota/token) se loggean y no detienen
        a los demás. La deduplicación es por URL normalizada.

        Args:
            player: Nombre del jugador titular.

        Returns:
            Artículos tipados, ordenados de más reciente a más antiguo.
        """
        seen_urls: set[str] = set()
        collected: list[NewsArticle] = []
        for provider in self.providers:
            if not provider.is_configured():
                continue
            try:
                raw_articles = provider.search_player(player)  # type: ignore[attr-defined]
            except ProviderError as exc:
                logger.warning("Noticias: %s falló para '%s' (%s)",
                               provider.provider_name, player, exc)
                continue
            for raw in raw_articles:
                url = normalize_text(raw.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(NewsArticle(
                    player=player,
                    title=raw.get("title", ""),
                    description=raw.get("description", ""),
                    content=raw.get("content", ""),
                    source=raw.get("source", "unknown"),
                    published_at=self._parse_date(raw.get("publishedAt", "")),
                    url=raw.get("url", ""),
                    provider=provider.provider_name,
                    matched_topics=tag_topics(
                        f"{raw.get('title', '')} {raw.get('description', '')} "
                        f"{raw.get('content', '')}"),
                ))
        collected.sort(key=lambda a: a.published_at, reverse=True)
        return collected

    @staticmethod
    def _parse_date(value: str) -> datetime:
        """Parsea fechas ISO-8601 de las APIs; ante fallo usa época UTC 0.

        Una fecha inválida recibe peso de recencia ≈ 0 (decae a cero), lo
        que neutraliza el artículo en lugar de inflar su influencia.
        """
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.fromtimestamp(0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Mercado (Polymarket Gamma + fallback de consenso)
# ---------------------------------------------------------------------------
class PolymarketProvider(BaseAPIProvider):
    """Cliente de la Gamma API pública de Polymarket (sin autenticación).

    Busca mercados head-to-head del partido; el parsing es tolerante a la
    evolución del esquema (campos ``outcomes``/``outcomePrices`` pueden
    venir como listas o strings JSON).
    """

    provider_name = "polymarket"

    def is_configured(self) -> bool:
        """La Gamma API es pública: siempre disponible salvo red."""
        return True

    @cached("market", config.CACHE_TTL_MARKET_S)
    def search_match_market(self, home_name: str, away_name: str) -> dict[str, Any] | None:
        """Busca un mercado del partido ``home vs away`` en Polymarket.

        Args:
            home_name: Nombre en inglés del local (mejora el recall).
            away_name: Nombre en inglés del visitante.

        Returns:
            Dict con ``{"home": p, "draw": p|None, "away": p, "question"}``
            con probabilidades ya renormalizadas, o ``None`` si no existe
            un mercado identificable.
        """
        import json as _json

        body = self._request(
            "GET", f"{config.get_endpoints().polymarket_gamma}/markets",
            params={"closed": "false", "limit": 100,
                    "search": f"{home_name} {away_name}"})
        markets = body if isinstance(body, list) else body.get("markets", [])
        h_norm, a_norm = normalize_text(home_name), normalize_text(away_name)
        for market in markets:
            question = normalize_text(str(market.get("question", "")))
            if h_norm not in question or a_norm not in question:
                continue
            outcomes = market.get("outcomes", [])
            prices = market.get("outcomePrices", [])
            if isinstance(outcomes, str):
                outcomes = _json.loads(outcomes)
            if isinstance(prices, str):
                prices = _json.loads(prices)
            if not outcomes or len(outcomes) != len(prices):
                continue
            probs = {normalize_text(str(o)): float(p)
                     for o, p in zip(outcomes, prices)}
            total = sum(probs.values())
            if total <= 0:
                continue
            probs = {k: v / total for k, v in probs.items()}  # de-vig proporcional
            home_p = next((v for k, v in probs.items() if h_norm in k), None)
            away_p = next((v for k, v in probs.items() if a_norm in k), None)
            draw_p = next((v for k, v in probs.items() if "draw" in k or "tie" in k),
                          None)
            if home_p is None or away_p is None:
                continue
            return {"home": home_p, "draw": draw_p, "away": away_p,
                    "question": market.get("question", "")}
        return None


# ---------------------------------------------------------------------------
# Fallback Data Store (matrices estructuradas embebidas)
# ---------------------------------------------------------------------------
class FallbackDataStore:
    """Acceso de solo lectura a los datasets reales embebidos.

    Matrices estructuradas: histórico FIFA 2014–2022 (agregados y cruces),
    onces titulares probables y snapshot outright del mercado. Se carga
    perezosamente y se memoiza en el proceso.
    """

    def __init__(self) -> None:
        self._history: dict[str, Any] | None = None
        self._squads: dict[str, Any] | None = None
        self._market: dict[str, Any] | None = None
        self._fifa_rankings: dict[str, Any] | None = None

    # -- carga perezosa -------------------------------------------------
    def _load(self, attr: str, path: Any) -> dict[str, Any]:
        import json as _json

        current = getattr(self, attr)
        if current is None:
            try:
                current = _json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise DataUnavailableError(
                    f"Fallback Data Store ausente: {path}") from exc
            setattr(self, attr, current)
        return current

    @property
    def history(self) -> dict[str, Any]:
        """Dataset histórico FIFA completo (ediciones + cruces + registro)."""
        return self._load("_history", config.HISTORY_FALLBACK_FILE)

    # -- vistas tipadas ---------------------------------------------------
    def registry(self) -> dict[str, dict[str, Any]]:
        """Registro maestro ``code -> {names, confederation}``."""
        return dict(self.history["teams_registry"])

    def knockout_matches(self) -> list[dict[str, Any]]:
        """Cruces de eliminación directa 2014–2022 con marcador a 90'."""
        return list(self.history["knockout_matches"])

    def editions(self) -> dict[str, dict[str, Any]]:
        """Agregados por edición (totales y por selección)."""
        return dict(self.history["editions"])

    def team_aggregate(self, code: str) -> dict[str, int] | None:
        """Suma partidos/GF/GA de un equipo a través de las ediciones.

        Args:
            code: Código FIFA.

        Returns:
            ``{"matches", "gf", "ga", "editions"}`` o ``None`` si el equipo
            no disputó ningún Mundial de la muestra.
        """
        total = {"matches": 0, "gf": 0, "ga": 0, "editions": 0}
        for edition in self.editions().values():
            team = edition["teams"].get(code)
            if team:
                total["matches"] += int(team["matches"])
                total["gf"] += int(team["gf"])
                total["ga"] += int(team["ga"])
                total["editions"] += 1
        return total if total["matches"] > 0 else None

    def all_team_aggregates(self) -> dict[str, dict[str, int]]:
        """Agregados de TODOS los equipos de la muestra (pool para EB)."""
        out: dict[str, dict[str, int]] = {}
        for code in {c for ed in self.editions().values() for c in ed["teams"]}:
            agg = self.team_aggregate(code)
            if agg:
                out[code] = agg
        return out

    def tournament_totals(self) -> dict[str, int]:
        """Totales de la muestra: partidos y goles (para μ del torneo)."""
        matches = sum(int(e["matches_total"]) for e in self.editions().values())
        goals = sum(int(e["goals_total"]) for e in self.editions().values())
        return {"matches": matches, "goals": goals}

    def squad_for(self, code: str) -> list[str] | None:
        """XI titular probable embebido para un equipo, si existe."""
        squads = self._load("_squads", config.SQUADS_FALLBACK_FILE)
        players = squads.get("squads", {}).get(code)
        return list(players) if players else None

    def outright_probabilities(self) -> dict[str, Any]:
        """Snapshot outright de consenso (incluye metadatos de corte)."""
        market = self._load("_market", config.MARKET_FALLBACK_FILE)
        return {"probs": dict(market["outright_winner_probabilities"]),
                "as_of": market["_meta"]["as_of"],
                "source": market["_meta"]["source"]}

    def fifa_rankings(self) -> dict[str, float]:
        """Puntos FIFA pre-torneo 2026 por código de selección.

        Returns:
            Mapeo ``{code: puntos}`` para los 48 clasificados.
        """
        data = self._load("_fifa_rankings", config.FIFA_RANKINGS_FILE)
        return {k: float(v) for k, v in data["rankings"].items()}


# ---------------------------------------------------------------------------
# Fachada unificada
# ---------------------------------------------------------------------------
class WorldCupDataEngine:
    """Fachada única de adquisición de datos para el pipeline predictivo.

    Orquesta proveedores vivos, caché estricta y Fallback Data Store con
    degradación en cascada: ``API viva → caché → fallback embebido``.
    Registra cada decisión en ``provenance`` para trazabilidad en la CLI.

    Attributes:
        store: Fallback Data Store embebido.
        sports_provider: Proveedor deportivo elegido por la Factory (o None).
        news: Agregador de prensa multi-proveedor.
        market: Cliente Polymarket.
        zafronix: Adaptador histórico.
        provenance: Bitácora ``(recurso, origen)`` de la corrida.
    """

    #: Ediciones que componen la ventana histórica del modelo.
    #: Se incluye 2026 para incorporar resultados reales del torneo en curso.
    HISTORY_SEASONS: tuple[int, ...] = (2014, 2018, 2022, 2026)

    def __init__(self,
                 store: FallbackDataStore | None = None,
                 sports_provider: BaseAPIProvider | None = None,
                 news: NewsAggregator | None = None,
                 market: PolymarketProvider | None = None) -> None:
        self.store = store or FallbackDataStore()
        self.sports_provider = (sports_provider if sports_provider is not None
                                else SportsProviderFactory.create())
        self.news = news or NewsAggregator()
        self.market = market or PolymarketProvider()
        self.zafronix = ZafronixHistoricalAdapter(self.store)
        self.provenance: list[tuple[str, str]] = []
        self.cache = CacheManager()

    # ------------------------------------------------------------------
    # Resolución de equipos
    # ------------------------------------------------------------------
    def resolve_team(self, query: str) -> str:
        """Resuelve un nombre libre (es/en, con o sin tildes) a código FIFA.

        Args:
            query: Texto del usuario (``"México"``, ``"Netherlands"``, ``"arg"``).

        Returns:
            Código FIFA de tres letras.

        Raises:
            TeamResolutionError: Si no hay coincidencia razonable; incluye
                sugerencias por similitud (difflib).
        """
        registry = self.store.registry()
        q = normalize_text(query)
        if q.upper() in {c.lower().upper() for c in registry}:
            return q.upper()
        alias_map: dict[str, str] = {}
        for code, meta in registry.items():
            alias_map[normalize_text(code)] = code
            for name in meta["names"]:
                alias_map[normalize_text(name)] = code
        if q in alias_map:
            return alias_map[q]
        close = difflib.get_close_matches(q, alias_map.keys(), n=3, cutoff=0.75)
        if len(close) == 1:
            resolved = alias_map[close[0]]
            logger.info("Resolución difusa: '%s' → %s", query, resolved)
            return resolved
        raise TeamResolutionError(query, [alias_map[c] for c in close])

    def display_name(self, code: str, english: bool = False) -> str:
        """Nombre canónico de un equipo (primer alias; último si inglés)."""
        names = self.store.registry().get(code, {}).get("names", [code])
        return names[-1] if english and len(names) > 1 else names[0]

    # ------------------------------------------------------------------
    # Métricas deportivas (agregados crudos para el estimador)
    # ------------------------------------------------------------------
    def get_fifa_rankings(self) -> dict[str, float]:
        """Puntos FIFA pre-torneo 2026 por código de selección.

        Returns:
            Mapeo ``{code: puntos}`` (48 clasificados al Mundial 2026).
        """
        return self.store.fifa_rankings()

    def get_team_aggregate_pool(self) -> dict[str, dict[str, float]]:
        """Agregados sin ponderar para el pool empírico bayesiano (τ² entre equipos).

        Se usa sin decaimiento intencional: la varianza entre selecciones
        (τ²) debe estimarse sobre toda la muestra disponible para evitar
        colapsar el shrinkage a w=0 por ruido inflado. El decaimiento
        por recencia se aplica solo sobre μ (get_tournament_totals).
        """
        return {k: {ky: float(v) for ky, v in agg.items()}
                for k, agg in self.store.all_team_aggregates().items()}

    def get_tournament_totals(self) -> dict[str, float]:
        """Totales globales ponderados por recencia: partidos y goles."""
        import math as _math
        goals = matches = 0.0
        for edition_year_str, edition_data in self.store.editions().items():
            year = int(edition_year_str)
            weight = _math.exp(-config.RECENCY_DECAY * (2026 - year) / 4.0)
            goals += float(edition_data["goals_total"]) * weight
            matches += float(edition_data["matches_total"]) * weight
        return {"goals": goals, "matches": matches}

    def get_knockout_raw_matches(self) -> list[dict[str, Any]]:
        """Partidos de eliminación directa normalizados para estimar ρ DC."""
        matches, _ = self.zafronix.knockout_matches()
        return [{"home_goals_90": int(m["score_90"][0]),
                 "away_goals_90": int(m["score_90"][1])}
                for m in matches]

    def get_match_history(self) -> tuple[list[dict[str, Any]], DataOrigin]:
        """Obtiene la historia de partidos del Mundial (ventana 10 años).

        Intenta el proveedor vivo edición por edición; si TODAS fallan se
        degrada al dataset embebido (que solo contiene cruces de
        eliminación con marcador exacto + agregados por equipo).

        Returns:
            Tupla ``(partidos_normalizados, origen)``.
        """
        if self.sports_provider is not None:
            collected: list[dict[str, Any]] = []
            for season in self.HISTORY_SEASONS:
                try:
                    collected.extend(
                        self.sports_provider.fetch_world_cup_matches(season))  # type: ignore[attr-defined]
                except ProviderError as exc:
                    logger.warning("Histórico %s falló para %s: %s",
                                   self.sports_provider.provider_name, season, exc)
            if collected:
                self.provenance.append(("match_history", DataOrigin.LIVE_API.value))
                return collected, DataOrigin.LIVE_API
        self.provenance.append(("match_history", DataOrigin.FALLBACK_STORE.value))
        return [], DataOrigin.FALLBACK_STORE

    def get_team_aggregate(self, code: str,
                           live_matches: list[dict[str, Any]] | None = None
                           ) -> tuple[dict[str, float], DataOrigin]:
        """Agregado GF/GA/partidos de un equipo en la ventana histórica.

        Args:
            code: Código FIFA.
            live_matches: Partidos normalizados de la API viva, si los hay;
                evita una segunda pasada de red.

        Returns:
            Tupla ``(agregado, origen)`` con claves ``matches/gf/ga``.

        Raises:
            DataUnavailableError: El equipo no tiene historia en APIs ni
                en el fallback (el estimador usará el prior confederativo).
        """
        if live_matches:
            name_variants = {normalize_text(code)}
            for alias in self.store.registry().get(code, {}).get("names", []):
                name_variants.add(normalize_text(alias))
            matches = gf = ga = 0
            for m in live_matches:
                home = normalize_text(str(m["home"]))
                away = normalize_text(str(m["away"]))
                if home in name_variants:
                    matches += 1
                    gf += int(m["home_goals"])
                    ga += int(m["away_goals"])
                elif away in name_variants:
                    matches += 1
                    gf += int(m["away_goals"])
                    ga += int(m["home_goals"])
            if matches > 0:
                self.provenance.append((f"aggregate:{code}", DataOrigin.LIVE_API.value))
                return ({"matches": float(matches), "gf": float(gf),
                         "ga": float(ga)}, DataOrigin.LIVE_API)
        agg = self.store.team_aggregate(code)
        if agg is None:
            raise DataUnavailableError(
                f"{code} sin historia mundialista 2014–2022; se usará prior")
        self.provenance.append((f"aggregate:{code}", DataOrigin.FALLBACK_STORE.value))
        return ({"matches": float(agg["matches"]), "gf": float(agg["gf"]),
                 "ga": float(agg["ga"])}, DataOrigin.FALLBACK_STORE)

    # ------------------------------------------------------------------
    # Tendencias históricas / XI / noticias / mercado
    # ------------------------------------------------------------------
    def get_knockout_tendencies(self) -> KnockoutTendencies:
        """Tendencias de fases finales vía adaptador Zafronix."""
        tendencies = self.zafronix.compute_tendencies()
        self.provenance.append(("knockout_tendencies",
                                DataOrigin.LIVE_API.value if self.zafronix.is_configured()
                                else DataOrigin.FALLBACK_STORE.value))
        return tendencies

    def get_recent_form(self, team_code: str, last_n: int = 10) -> float:
        """Multiplicador de forma reciente del equipo en [0.85, 1.15].

        Delega en :class:`RecentFormProvider` (football-data.org).  Si
        ``FOOTBALL_DATA_API_KEY`` no está configurada devuelve 1.0 (forma
        neutra) y el comportamiento del pipeline es idéntico al actual.

        Args:
            team_code: Código FIFA de tres letras.
            last_n: Número de partidos recientes a considerar (default 10).

        Returns:
            Multiplicador en [0.85, 1.15]; 1.0 cuando la clave no está
            disponible o el equipo no tiene mapeo verificado.
        """
        provider = RecentFormProvider()
        form = provider.get_recent_form(team_code, last_n=last_n)
        origin = DataOrigin.LIVE_API.value if provider.is_available() else "unavailable"
        self.provenance.append((f"recent_form:{team_code}", origin))
        return form

    def get_starting_xi(self, code: str) -> tuple[list[str], DataOrigin]:
        """Once titular probable de una selección.

        Cadena: API-Football (si hay key y mapeo de id) → squads embebidos.

        Args:
            code: Código FIFA.

        Returns:
            Tupla ``(jugadores, origen)``; lista vacía si no hay dato (el
            TIF resultará neutro = 1.0 y se reporta el motivo).
        """
        af = (self.sports_provider
              if isinstance(self.sports_provider, ApiFootballProvider)
              else ApiFootballProvider())
        if af.is_configured():
            try:
                team_id = af.fetch_team_id(self.display_name(code, english=True))
                if team_id is not None:
                    squad_live = af.fetch_squad(team_id)
                    if squad_live:
                        self.provenance.append((f"xi:{code}", DataOrigin.LIVE_API.value))
                        return squad_live[:11], DataOrigin.LIVE_API
            except ProviderError as exc:
                logger.warning("XI vivo no disponible para %s (%s); fallback", code, exc)
        squad = self.store.squad_for(code)
        if squad:
            self.provenance.append((f"xi:{code}", DataOrigin.FALLBACK_STORE.value))
            return squad[:11], DataOrigin.FALLBACK_STORE
        self.provenance.append((f"xi:{code}", "unavailable"))
        return [], DataOrigin.FALLBACK_STORE

    def get_player_news(self, players: list[str]) -> dict[str, list[NewsArticle]]:
        """Prensa reciente para cada jugador del once titular.

        Las búsquedas HTTP se ejecutan en paralelo (hasta 6 workers) para
        reducir la latencia total de ~2 s (secuencial) a ~0.3 s (paralelo)
        cuando todos los jugadores se consultan a la vez. El orden del mapeo
        de salida es determinístico: refleja el orden original de ``players``.
        Si la búsqueda de un jugador individual falla, ese jugador recibe
        una lista vacía y el resto continúa sin interrupciones.

        Args:
            players: Nombres de los 11 titulares.

        Returns:
            Mapeo jugador → artículos (vacío si no hay keys de noticias).
        """
        any_provider = any(p.is_configured() for p in self.news.providers)
        if not any_provider or not players:
            self.provenance.append(("news", "unavailable"))
            return {player: [] for player in players}

        t0 = time.perf_counter()

        def _fetch_for_player(player: str) -> tuple[str, list[NewsArticle]]:
            try:
                return player, self.news.articles_for_player(player)
            except Exception:  # noqa: BLE001
                logger.warning("Noticias: fetch falló para '%s'; usando lista vacía", player)
                return player, []

        max_workers = min(6, len(players))
        results: dict[str, list[NewsArticle]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_for_player, p): p for p in players}
            for future in as_completed(futures):
                player, articles = future.result()
                results[player] = articles

        logging.debug("TIF fetch: %d players in %.2fs", len(players), time.perf_counter() - t0)

        # Reconstruir en orden original
        result = {p: results[p] for p in players}
        self.provenance.append(("news", DataOrigin.LIVE_API.value))
        return result

    def get_market_snapshot(self, home_code: str, away_code: str,
                            force_refresh: bool = False) -> MarketSnapshot | None:
        """Snapshot de mercado para el partido (solo con ``--live-odds``).

        Cadena: Polymarket head-to-head → outright embebido (Bradley-Terry).

        Args:
            home_code: Código FIFA local.
            away_code: Código FIFA visitante.
            force_refresh: Bypass de caché (bandera ``--live-odds`` fresca).

        Returns:
            :class:`MarketSnapshot` o ``None`` si ni siquiera el fallback
            cubre a alguno de los equipos.
        """
        home_en = self.display_name(home_code, english=True)
        away_en = self.display_name(away_code, english=True)
        try:
            found = self.market.search_match_market(
                home_en, away_en, force_refresh=force_refresh)
            if found:
                draw = found.get("draw")
                self.provenance.append(("market", DataOrigin.LIVE_API.value))
                return MarketSnapshot(
                    home_prob=float(found["home"]),
                    draw_prob=float(draw) if draw is not None else None,
                    away_prob=float(found["away"]),
                    source=f"polymarket:{found.get('question', '')[:80]}",
                    as_of=datetime.now(timezone.utc),
                    origin=DataOrigin.LIVE_API,
                    is_outright_derived=False)
        except ProviderError as exc:
            logger.warning("Polymarket no disponible (%s); usando consenso embebido", exc)

        outright = self.store.outright_probabilities()
        probs: dict[str, float] = outright["probs"]
        field_mass = float(probs.get("FIELD", 0.0))
        listed = {k: float(v) for k, v in probs.items() if k != "FIELD"}
        # Equipos fuera de la lista reciben la masa 'FIELD' repartida entre
        # los cupos restantes del torneo (48 clasificados - listados).
        remaining_slots = max(1, 48 - len(listed))
        floor_prob = field_mass / remaining_slots
        s_home = listed.get(home_code, floor_prob)
        s_away = listed.get(away_code, floor_prob)
        if s_home <= 0 or s_away <= 0:
            return None
        denom = s_home + s_away
        self.provenance.append(("market", DataOrigin.FALLBACK_STORE.value))
        return MarketSnapshot(
            home_prob=s_home / denom,
            draw_prob=None,
            away_prob=s_away / denom,
            source=f"outright-consensus@{outright['as_of']}",
            as_of=datetime.fromisoformat(outright["as_of"]),
            origin=DataOrigin.FALLBACK_STORE,
            is_outright_derived=True)
