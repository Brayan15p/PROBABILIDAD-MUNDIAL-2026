"""Capa de persistencia del proyecto en Supabase (PostgreSQL gestionado).

Habla directamente con **PostgREST** (la API REST nativa de Supabase) vía
``requests``, sin SDK adicional: cada tabla expuesta en
``supabase/migrations/0001_init.sql`` es accesible en
``{SUPABASE_URL}/rest/v1/<tabla>``.

Responsabilidades:
    - Registrar cada predicción ejecutada (auditoría y backtesting futuro).
    - Sincronizar el registro maestro de selecciones (``teams``).
    - Persistir métricas, snapshots de mercado y reportes TIF con su
      procedencia, para reconstruir cualquier corrida históricamente.

La capa es **opcional y de mejor esfuerzo**: si no hay credenciales
(``SUPABASE_URL`` + key) el motor funciona igual; si Supabase falla en
runtime se emite un warning y el pipeline continúa (la predicción nunca
se bloquea por telemetría).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from src import config
from src.exceptions import ConfigurationError, ProviderError
from src.models import (
    MarketSnapshot,
    SimulationResult,
    TacticalImpactReport,
    TeamMetrics,
)

logger = logging.getLogger(__name__)


class SupabaseRepository:
    """Repositorio PostgREST para las tablas del motor.

    Attributes:
        base_url: URL del proyecto Supabase (``https://<ref>.supabase.co``).
        enabled: True solo si hay credenciales completas en el entorno.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None) -> None:
        creds = config.get_credentials()
        self.base_url: str | None = (base_url or creds.supabase_url or "").rstrip("/") or None
        self._api_key: str | None = api_key or creds.supabase_key
        self._timeout: float = timeout or config.HTTP_TIMEOUT_SECONDS
        self.enabled: bool = bool(self.base_url and self._api_key)

    # ------------------------------------------------------------------
    # Operaciones de escritura (best-effort desde el pipeline)
    # ------------------------------------------------------------------
    def save_prediction(self, result: SimulationResult,
                        tif_home: float, tif_away: float) -> bool:
        """Inserta una predicción completa en ``predictions``.

        Args:
            result: Salida del motor Monte Carlo.
            tif_home: TIF aplicado al equipo local.
            tif_away: TIF aplicado al equipo visitante.

        Returns:
            True si la fila quedó persistida; False si la capa está
            deshabilitada o Supabase rechazó la escritura (solo se loggea).
        """
        row = {
            "home_code": result.home_code,
            "away_code": result.away_code,
            "n_simulations": result.n_simulations,
            "lambda_home": result.lambda_home.value,
            "lambda_away": result.lambda_away.value,
            "lambda3_covariance": result.lambda3_covariance,
            "tif_home": tif_home,
            "tif_away": tif_away,
            "p_home_win": result.home_win_prob,
            "p_draw": result.draw_prob,
            "p_away_win": result.away_win_prob,
            "modal_score": (f"{result.modal_score.home_goals}-"
                            f"{result.modal_score.away_goals}"),
            "modal_probability": result.modal_confidence,
            "seed": result.seed,
            "payload": result.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert("predictions", [row])

    def sync_teams(self, registry: dict[str, dict[str, Any]]) -> bool:
        """Upsert del registro maestro de selecciones en ``teams``.

        Args:
            registry: Mapeo ``code -> {names, confederation}`` proveniente
                del Fallback Data Store o de la API deportiva.

        Returns:
            True si el upsert fue aceptado por PostgREST.
        """
        rows = [
            {
                "code": code,
                "name": meta["names"][0],
                "aliases": meta["names"],
                "confederation": meta.get("confederation", "UNKNOWN"),
            }
            for code, meta in registry.items()
        ]
        return self._insert("teams", rows, upsert_on="code")

    def save_team_metrics(self, metrics: TeamMetrics) -> bool:
        """Persiste un snapshot de métricas estimadas en ``team_metrics``."""
        row = {
            "team_code": metrics.code,
            "matches_sample": metrics.matches_sample,
            "goals_for_per_match": metrics.goals_for_per_match,
            "goals_against_per_match": metrics.goals_against_per_match,
            "attack_strength": metrics.attack_strength,
            "defense_resilience": metrics.defense_resilience,
            "xg_for_per_match": metrics.xg_for_per_match,
            "xg_against_per_match": metrics.xg_against_per_match,
            "origin": metrics.origin.value,
            "shrinkage_weight": metrics.shrinkage_weight,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert("team_metrics", [row])

    def save_market_snapshot(self, home: str, away: str,
                             snapshot: MarketSnapshot) -> bool:
        """Persiste un snapshot de mercado en ``market_snapshots``."""
        row = {
            "home_code": home,
            "away_code": away,
            "p_home": snapshot.home_prob,
            "p_draw": snapshot.draw_prob,
            "p_away": snapshot.away_prob,
            "source": snapshot.source,
            "is_outright_derived": snapshot.is_outright_derived,
            "origin": snapshot.origin.value,
            "as_of": snapshot.as_of.isoformat(),
        }
        return self._insert("market_snapshots", [row])

    def save_tif_report(self, report: TacticalImpactReport) -> bool:
        """Persiste un reporte TIF en ``tif_reports``."""
        row = {
            "team_code": report.team_code,
            "tif": report.tif,
            "team_polarity": report.team_polarity,
            "articles_total": report.articles_total,
            "engine": report.engine,
            "players": [
                {"player": p.player, "articles": p.articles_analyzed,
                 "raw": p.raw_polarity, "shrunk": p.shrunk_polarity,
                 "topics": p.topics}
                for p in report.players
            ],
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        return self._insert("tif_reports", [row])

    # ------------------------------------------------------------------
    # Lectura (para backtesting / dashboards)
    # ------------------------------------------------------------------
    def recent_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Lista las últimas predicciones registradas.

        Args:
            limit: Máximo de filas a devolver.

        Returns:
            Filas de ``predictions`` ordenadas por fecha descendente.

        Raises:
            ConfigurationError: Si la capa Supabase no está configurada.
            ProviderError: Si PostgREST devuelve un error HTTP.
        """
        self._require_enabled()
        resp = self._request(
            "GET", "predictions",
            params={"select": ("created_at,home_code,away_code,p_home_win,"
                               "p_draw,p_away_win,modal_score,modal_probability"),
                    "order": "created_at.desc", "limit": str(limit)})
        return list(resp.json())

    def health_check(self) -> dict[str, Any]:
        """Verifica conectividad y existencia del esquema.

        Returns:
            Dict con ``ok`` y el conteo de predicciones almacenadas.

        Raises:
            ConfigurationError: Si faltan credenciales.
            ProviderError: Si el esquema no existe o la key es inválida.
        """
        self._require_enabled()
        resp = self._request("GET", "predictions",
                             params={"select": "id", "limit": "1"},
                             headers={"Prefer": "count=exact"})
        content_range = resp.headers.get("content-range", "0/0")
        total = content_range.split("/")[-1]
        return {"ok": True, "predictions_stored": total, "url": self.base_url}

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _require_enabled(self) -> None:
        if not self.enabled:
            raise ConfigurationError(
                "Supabase no configurado: defina SUPABASE_URL y "
                "SUPABASE_SERVICE_ROLE_KEY (o SUPABASE_ANON_KEY) en .env")

    def _insert(self, table: str, rows: list[dict[str, Any]],
                upsert_on: str | None = None) -> bool:
        if not self.enabled:
            return False
        headers = {"Prefer": "return=minimal"}
        params: dict[str, str] = {}
        if upsert_on:
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
            params["on_conflict"] = upsert_on
        try:
            self._request("POST", table, json_body=rows,
                          headers=headers, params=params)
            return True
        except (ProviderError, ConfigurationError) as exc:
            logger.warning("Supabase: escritura en '%s' falló (%s); "
                           "el pipeline continúa sin persistencia.", table, exc)
            return False

    def _request(self, method: str, table: str,
                 params: dict[str, str] | None = None,
                 json_body: Any | None = None,
                 headers: dict[str, str] | None = None) -> requests.Response:
        self._require_enabled()
        url = f"{self.base_url}/rest/v1/{table}"
        base_headers = {
            "apikey": self._api_key or "",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if headers:
            base_headers.update(headers)
        try:
            resp = requests.request(method, url, params=params, json=json_body,
                                    headers=base_headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"Supabase inalcanzable: {exc}",
                                provider="supabase") from exc
        if resp.status_code in (401, 403):
            raise ProviderError("Credencial Supabase rechazada",
                                provider="supabase", status_code=resp.status_code)
        if resp.status_code >= 400:
            raise ProviderError(
                f"PostgREST {resp.status_code}: {resp.text[:300]}",
                provider="supabase", status_code=resp.status_code)
        return resp
