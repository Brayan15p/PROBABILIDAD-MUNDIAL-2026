"""Modelos de dominio inmutables que fluyen entre los componentes del motor.

Cada etapa del pipeline (datos → NLP → simulación → render) se comunica
exclusivamente mediante estas estructuras tipadas; ningún componente conoce
los formatos crudos de las APIs de otro componente (principio de inversión
de dependencias).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DataOrigin(str, Enum):
    """Procedencia de una métrica: API viva, caché local o fallback embebido."""

    LIVE_API = "live_api"
    LOCAL_CACHE = "local_cache"
    FALLBACK_STORE = "fallback_store"


class KnockoutStage(str, Enum):
    """Fases finales de un Mundial, ordenadas por profundidad."""

    ROUND_OF_16 = "R16"
    QUARTER_FINAL = "QF"
    SEMI_FINAL = "SF"
    THIRD_PLACE = "3RD"
    FINAL = "F"


@dataclass(frozen=True)
class TeamMetrics:
    """Métricas agregadas estimadas para una selección nacional.

    Las fuerzas son ratios relativos a la media del torneo (1.0 = promedio):
    ``attack_strength`` multiplica los goles esperados a favor y
    ``defense_resilience`` los goles esperados en contra del rival.

    Attributes:
        code: Código FIFA de tres letras (p. ej. ``ARG``).
        name: Nombre canónico en español.
        matches_sample: Tamaño muestral (partidos) detrás de la estimación.
        goals_for_per_match: Tasa observada de goles a favor.
        goals_against_per_match: Tasa observada de goles en contra.
        attack_strength: Fuerza de ataque relativa (MLE con shrinkage EB).
        defense_resilience: Vulnerabilidad defensiva relativa (1.0 = media;
            valores < 1 indican defensa más sólida que la media).
        xg_for_per_match: xG a favor por partido si la API lo provee.
        xg_against_per_match: xG en contra por partido si la API lo provee.
        squad_rating: Rating medio de la plantilla titular si la API lo provee.
        origin: Procedencia del dato.
        shrinkage_weight: Peso w∈[0,1] que recibió la evidencia propia del
            equipo frente al prior (1 = muestra grande, dato confiable).
    """

    code: str
    name: str
    matches_sample: int
    goals_for_per_match: float
    goals_against_per_match: float
    attack_strength: float
    defense_resilience: float
    xg_for_per_match: float | None = None
    xg_against_per_match: float | None = None
    squad_rating: float | None = None
    origin: DataOrigin = DataOrigin.FALLBACK_STORE
    shrinkage_weight: float = 0.0


@dataclass(frozen=True)
class NewsArticle:
    """Artículo de prensa asociado a un jugador titular.

    Attributes:
        player: Nombre del jugador buscado.
        title: Titular del artículo.
        description: Bajada/resumen provisto por la fuente.
        content: Cuerpo (posiblemente truncado por el plan gratuito).
        source: Medio que publica.
        published_at: Fecha de publicación (UTC).
        url: Enlace canónico.
        provider: API de origen (newsapi | gnews).
        matched_topics: Etiquetas de la taxonomía crítica detectadas
            (lesión, sobrecarga, crisis, racha goleadora, suspensión).
    """

    player: str
    title: str
    description: str
    content: str
    source: str
    published_at: datetime
    url: str
    provider: str
    matched_topics: tuple[str, ...] = ()

    @property
    def full_text(self) -> str:
        """Concatena título, bajada y cuerpo para el análisis semántico."""
        return ". ".join(p for p in (self.title, self.description, self.content) if p)


@dataclass(frozen=True)
class PlayerSentiment:
    """Resultado del análisis de sentimiento agregado de un jugador.

    Attributes:
        player: Nombre del jugador.
        articles_analyzed: Número de artículos con señal válida.
        raw_polarity: Media ponderada (por recencia) de la polaridad VADER.
        shrunk_polarity: Polaridad tras shrinkage bayesiano hacia 0.
        topics: Conteo de tópicos críticos detectados en sus noticias.
    """

    player: str
    articles_analyzed: int
    raw_polarity: float
    shrunk_polarity: float
    topics: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TacticalImpactReport:
    """Tactical Impact Factor (TIF) de una selección.

    Attributes:
        team_code: Código FIFA del equipo.
        tif: Multiplicador final en [0.80, 1.20].
        team_polarity: Polaridad agregada del plantel en [-1, 1].
        players: Desglose por jugador titular.
        articles_total: Total de artículos consumidos.
        engine: Backend NLP usado (vader | transformers | neutral-fallback).
    """

    team_code: str
    tif: float
    team_polarity: float
    players: tuple[PlayerSentiment, ...]
    articles_total: int
    engine: str


@dataclass(frozen=True)
class MarketSnapshot:
    """Probabilidades implícitas del consenso de mercado para un partido.

    Si el proveedor expone un mercado head-to-head se usan las tres patas
    directamente; si solo existe el mercado *outright* (campeón), se derivan
    fuerzas relativas Bradley-Terry y ``is_outright_derived`` queda en True.

    Attributes:
        home_prob: Probabilidad implícita (sin margen) de victoria local.
        draw_prob: Probabilidad implícita de empate (None si outright).
        away_prob: Probabilidad implícita de victoria visitante.
        source: Identificador del mercado consultado.
        as_of: Marca temporal del snapshot.
        origin: Procedencia (API viva / caché / fallback).
        is_outright_derived: True si proviene del mercado de campeón.
    """

    home_prob: float
    draw_prob: float | None
    away_prob: float
    source: str
    as_of: datetime
    origin: DataOrigin
    is_outright_derived: bool


@dataclass(frozen=True)
class LambdaBreakdown:
    """Trazabilidad completa de la construcción de un lambda (goles esperados).

    λ = tournament_mu × attack × opponent_defense × tif × home_advantage
        × recent_form × market_factor

    Attributes:
        tournament_mu: Media de goles por equipo-partido del torneo (MLE).
        attack: Fuerza de ataque del equipo.
        opponent_defense: Vulnerabilidad defensiva del rival.
        tif: Tactical Impact Factor aplicado.
        home_advantage: Factor de ventaja de local (1.0 si no aplica).
        recent_form: Multiplicador de forma reciente en [0.85, 1.15]
            (1.0 cuando FOOTBALL_DATA_API_KEY no está configurada).
        market_factor: Peso multiplicativo inducido por el mercado.
        value: Producto final λ.
    """

    tournament_mu: float
    attack: float
    opponent_defense: float
    tif: float
    home_advantage: float
    recent_form: float
    market_factor: float
    value: float


@dataclass(frozen=True)
class ScorelineProbability:
    """Un marcador exacto con su probabilidad estimada e incertidumbre MC.

    Attributes:
        home_goals: Goles del local.
        away_goals: Goles del visitante.
        probability: Frecuencia relativa Monte Carlo.
        mc_standard_error: Error estándar binomial sqrt(p(1-p)/n).
        analytic_probability: pmf bivariada exacta (validación cruzada).
    """

    home_goals: int
    away_goals: int
    probability: float
    mc_standard_error: float
    analytic_probability: float


@dataclass(frozen=True)
class SimulationResult:
    """Salida completa del motor Monte Carlo bivariado.

    Attributes:
        home_code: Código FIFA del local.
        away_code: Código FIFA del visitante.
        n_simulations: Iteraciones ejecutadas.
        lambda_home: Desglose del λ local.
        lambda_away: Desglose del λ visitante.
        lambda3_covariance: Componente común λ₃ (covarianza de goles).
        home_win_prob: P(victoria local) con su IC del 95%% implícito en se.
        draw_prob: P(empate).
        away_win_prob: P(victoria visitante).
        macro_standard_error: SE Monte Carlo máximo de las tres macro-probs.
        top_scorelines: Marcadores ordenados por probabilidad descendente.
        modal_score: Marcador exacto más probable (moda absoluta conjunta).
        modal_confidence: Probabilidad de la moda.
        modal_confidence_interval: IC 95%% de la probabilidad modal.
        mode_separation_ratio: P(moda)/P(segunda moda) — nitidez de la moda.
        over_2_5_prob: P(goles totales > 2.5).
        btts_prob: P(ambos marcan).
        expected_total_goals: E[goles totales] empírico.
        seed: Semilla del RNG para reproducibilidad.
        overdispersion_k: Parámetro de forma k del modelo Gamma-Poisson compuesto
            (Negative Binomial). k→∞ = Poisson puro; k≈3-6 = sobredispersión
            empírica del fútbol. 0.0 indica que se usó Poisson puro.
    """

    home_code: str
    away_code: str
    n_simulations: int
    lambda_home: LambdaBreakdown
    lambda_away: LambdaBreakdown
    lambda3_covariance: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    macro_standard_error: float
    top_scorelines: tuple[ScorelineProbability, ...]
    modal_score: ScorelineProbability
    modal_confidence: float
    modal_confidence_interval: tuple[float, float]
    mode_separation_ratio: float
    over_2_5_prob: float
    btts_prob: float
    expected_total_goals: float
    dixon_coles_rho: float
    seed: int
    overdispersion_k: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resultado a un dict apto para ``json.dumps``.

        Returns:
            Representación JSON-compatible de la predicción completa.
        """
        return {
            "match": {"home": self.home_code, "away": self.away_code},
            "n_simulations": self.n_simulations,
            "lambdas": {
                "home": vars(self.lambda_home),
                "away": vars(self.lambda_away),
                "covariance_lambda3": self.lambda3_covariance,
            },
            "macro_probabilities": {
                "home_win": round(self.home_win_prob, 6),
                "draw": round(self.draw_prob, 6),
                "away_win": round(self.away_win_prob, 6),
                "mc_standard_error": round(self.macro_standard_error, 6),
            },
            "modal_score": {
                "score": f"{self.modal_score.home_goals}-{self.modal_score.away_goals}",
                "probability": round(self.modal_confidence, 6),
                "ci95": [round(x, 6) for x in self.modal_confidence_interval],
                "analytic_probability": round(self.modal_score.analytic_probability, 6),
                "separation_ratio": round(self.mode_separation_ratio, 4),
            },
            "auxiliary_markets": {
                "over_2_5": round(self.over_2_5_prob, 6),
                "btts": round(self.btts_prob, 6),
                "expected_total_goals": round(self.expected_total_goals, 4),
                "dixon_coles_rho": round(self.dixon_coles_rho, 6),
                "overdispersion_k": round(self.overdispersion_k, 4),
            },
            "top_scorelines": [
                {
                    "score": f"{s.home_goals}-{s.away_goals}",
                    "probability": round(s.probability, 6),
                    "analytic": round(s.analytic_probability, 6),
                }
                for s in self.top_scorelines
            ],
            "seed": self.seed,
        }


@dataclass(frozen=True)
class KnockoutTendencies:
    """Tendencias históricas de fases finales (vía adaptador Zafronix).

    Attributes:
        editions: Ediciones cubiertas por la muestra.
        matches: Número de cruces de eliminación directa analizados.
        goals_per_match_90: Media de goles en 90 minutos en fases finales.
        draw_rate_90: Proporción de cruces igualados tras 90 minutos.
        goal_covariance: Covarianza muestral (goles local, goles visitante);
            insumo directo del λ₃ del modelo bivariado.
        stage_goal_means: Media de goles por fase (R16/QF/SF/3RD/F).
    """

    editions: tuple[int, ...]
    matches: int
    goals_per_match_90: float
    draw_rate_90: float
    goal_covariance: float
    stage_goal_means: dict[str, float]
