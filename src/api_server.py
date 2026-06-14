"""FastAPI REST interface for the WC-2026 prediction engine.

Exposes the same Monte Carlo bivariate Poisson prediction engine used by the
CLI (wc26 predecir) as a REST API, without replacing the CLI.

Endpoints:
    GET  /health                  — Liveness check
    GET  /teams                   — FIFA team registry
    POST /predict                 — Full match prediction
    GET  /tournament/groups       — Tournament group structure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.api_client import NewsAggregator, PolymarketProvider, WorldCupDataEngine
from src.exceptions import WorldCupEngineError
from src.nlp_processor import TacticalSentimentAnalyzer, create_backend
from src.simulator import MarketBlender, PredictionPipeline

app = FastAPI(
    title="Predictor Mundial 2026",
    version="1.0.0",
    description="Motor Monte Carlo bivariado con Dixon-Coles y EB shrinkage",
)


# ---------------------------------------------------------------------------
# Stub offline (mirrors _OfflineMarket in cli.py)
# ---------------------------------------------------------------------------

class _OfflineMarket(PolymarketProvider):
    """Stub que nunca toca la red: fuerza el fallback al consenso embebido."""

    def search_match_market(
        self, home_name: str, away_name: str, **_kwargs: Any
    ) -> dict[str, Any] | None:
        return None


def _build_engine(offline: bool) -> WorldCupDataEngine:
    """Construye la fachada de datos, recortada si se pidió modo offline."""
    engine = WorldCupDataEngine()
    if offline:
        engine.sports_provider = None
        engine.news = NewsAggregator(providers=[])
        engine.market = _OfflineMarket()
    return engine


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    home: str           # Código FIFA o nombre (ej: "ARG", "Argentina")
    away: str           # Código FIFA o nombre (ej: "FRA", "Francia")
    sims: int = 15000   # Mínimo 15000
    seed: int | None = None
    offline: bool = True


class MatchResponse(BaseModel):
    match: dict
    macro_probabilities: dict
    modal_score: dict
    auxiliary_markets: dict
    lambdas: dict
    top_scorelines: list
    n_simulations: int
    context: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness check del servicio."""
    return {"status": "ok", "engine": "bivariate-poisson-dc"}


@app.get("/teams")
def list_teams() -> dict:
    """Lista todos los equipos del Mundial 2026 con sus códigos FIFA.

    Returns a mapping of FIFA code -> team metadata (names, confederation).
    """
    engine = WorldCupDataEngine()
    registry = engine.store.registry()
    # Return code -> {names, confederation} dict
    return {
        code: {
            "names": meta.get("names", []),
            "confederation": meta.get("confederation", ""),
        }
        for code, meta in sorted(registry.items())
    }


@app.post("/predict", response_model=MatchResponse)
def predict(req: MatchRequest) -> MatchResponse:
    """Predice el resultado de un partido usando el motor Monte Carlo bivariado.

    Usa el motor offline por defecto (offline=True), que no realiza llamadas
    HTTP externas y opera sobre el fallback embebido.
    """
    if req.sims < config.MIN_MONTE_CARLO_SIMS:
        raise HTTPException(
            status_code=422,
            detail=f"sims must be >= {config.MIN_MONTE_CARLO_SIMS}",
        )

    engine = _build_engine(req.offline)
    analyzer = TacticalSentimentAnalyzer(backend=create_backend("vader"))
    pipeline = PredictionPipeline(engine, analyzer, blender=MarketBlender())

    try:
        artifacts = pipeline.predict(
            req.home,
            req.away,
            req.sims,
            live_odds=False,  # API does not expose live-odds; use offline blend
            seed=req.seed,
        )
    except WorldCupEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = artifacts.result
    payload = result.to_dict()

    # Add context block (matches the JSON output of the CLI --json flag)
    context: dict[str, Any] = {
        "tournament_mu": artifacts.tournament_mu,
        "estimation_method": artifacts.estimation_method,
        "tif": {
            result.home_code: artifacts.tif_home.tif,
            result.away_code: artifacts.tif_away.tif,
        },
        "market_source": artifacts.market.source if artifacts.market else None,
        "persisted_to_supabase": False,
    }

    return MatchResponse(
        match=payload["match"],
        macro_probabilities=payload["macro_probabilities"],
        modal_score=payload["modal_score"],
        auxiliary_markets=payload["auxiliary_markets"],
        lambdas=payload["lambdas"],
        top_scorelines=payload["top_scorelines"],
        n_simulations=payload["n_simulations"],
        context=context,
    )


@app.get("/tournament/groups")
def get_groups() -> dict:
    """Devuelve los grupos del torneo con todos los equipos.

    Returns the official 2026 FIFA World Cup draw: 12 groups of 4 teams each,
    loaded from data/fallback/groups_2026.json.
    """
    groups_path: Path = config.FALLBACK_DIR / "groups_2026.json"
    try:
        with groups_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"groups_2026.json not found at {groups_path}",
        ) from exc

    groups: dict[str, list[str]] = {
        letter: list(teams)
        for letter, teams in raw.get("groups", {}).items()
    }
    return {"groups": groups, "total_teams": sum(len(v) for v in groups.values())}
