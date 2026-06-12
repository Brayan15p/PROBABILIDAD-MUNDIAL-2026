"""Procesador Semántico Táctico: del texto de prensa al Tactical Impact Factor.

Pipeline estadístico (sin heurísticas de signo arbitrarias):

1. **Polaridad por artículo**: backend de sentimiento intercambiable
   (*Strategy*): VADER (default, léxico calibrado para inglés) o un
   pipeline local de ``transformers`` si está instalado.
2. **Agregación por jugador**: media ponderada por recencia con decaimiento
   exponencial (vida media anclada al ciclo de partidos FIFA, ver
   ``config.NEWS_HALF_LIFE_HOURS``).
3. **Shrinkage bayesiano empírico hacia 0**: la polaridad de un jugador con
   poca prensa se contrae hacia el prior neutro proporcionalmente a su
   varianza muestral (estimador James-Stein de momentos); evita que un
   único titular sensacionalista mueva el rendimiento esperado del equipo.
4. **Escalado contractual al TIF**: mapeo lineal simétrico de la polaridad
   de equipo p ∈ [-1, 1] al multiplicador 1 + 0.20·p ∈ [0.80, 1.20]
   (cotas fijadas por especificación de producto, ver ``src.config``).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from src import config
from src.exceptions import ConfigurationError
from src.models import NewsArticle, PlayerSentiment, TacticalImpactReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backends de sentimiento (patrón Strategy)
# ---------------------------------------------------------------------------
class SentimentBackend(ABC):
    """Contrato de un motor de polaridad de texto."""

    #: Identificador del backend para trazabilidad del reporte.
    name: str = "abstract"

    @abstractmethod
    def polarity(self, text: str) -> float:
        """Devuelve la polaridad del texto en el intervalo [-1, 1]."""


class VaderBackend(SentimentBackend):
    """Backend VADER (léxico + reglas, robusto en titulares en inglés)."""

    name = "vader"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def polarity(self, text: str) -> float:
        """Compound score de VADER, ya normalizado en [-1, 1].

        Args:
            text: Texto completo del artículo.

        Returns:
            Polaridad compound.
        """
        return float(self._analyzer.polarity_scores(text)["compound"])


class TransformersBackend(SentimentBackend):
    """Backend opcional con pipeline local de Hugging Face (multilingüe).

    Mapea las etiquetas {negative, neutral, positive} a {-1, 0, +1}
    escaladas por la confianza del modelo — una esperanza sobre el soporte
    discreto del clasificador, no una constante inventada.
    """

    name = "transformers"

    #: Modelo multilingüe estándar de sentimiento en 3 clases.
    DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "Backend 'transformers' solicitado pero no instalado; "
                "instale el extra: pip install '.[nlp-advanced]'") from exc
        self._pipe = pipeline("sentiment-analysis",
                              model=model_name or self.DEFAULT_MODEL,
                              truncation=True)

    def polarity(self, text: str) -> float:
        """Esperanza de polaridad del clasificador en [-1, 1].

        Args:
            text: Texto completo del artículo.

        Returns:
            ``sign(label) * score`` con label ∈ {negative, neutral, positive}.
        """
        result = self._pipe(text[:512])[0]
        label = str(result["label"]).lower()
        score = float(result["score"])
        if "pos" in label:
            return score
        if "neg" in label:
            return -score
        return 0.0


def create_backend(prefer: str = "vader") -> SentimentBackend:
    """Factory de backends de sentimiento con degradación a VADER.

    Args:
        prefer: ``"vader"`` o ``"transformers"``.

    Returns:
        Backend instanciado y listo.
    """
    if prefer == "transformers":
        try:
            return TransformersBackend()
        except ConfigurationError as exc:
            logger.warning("%s — degradando a VADER", exc)
    return VaderBackend()


# ---------------------------------------------------------------------------
# Analizador táctico
# ---------------------------------------------------------------------------
class TacticalSentimentAnalyzer:
    """Convierte la prensa de los 11 titulares en el TIF de la selección.

    Attributes:
        backend: Motor de polaridad (Strategy inyectable, testeable).
        half_life_hours: Vida media del decaimiento de recencia.
    """

    def __init__(self, backend: SentimentBackend | None = None,
                 half_life_hours: float | None = None,
                 now: datetime | None = None) -> None:
        self.backend: SentimentBackend = backend or create_backend()
        self.half_life_hours: float = half_life_hours or config.NEWS_HALF_LIFE_HOURS
        self._now: datetime = now or datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def team_report(self, team_code: str,
                    news_by_player: dict[str, list[NewsArticle]]
                    ) -> TacticalImpactReport:
        """Calcula el Tactical Impact Factor de una selección.

        Args:
            team_code: Código FIFA del equipo.
            news_by_player: Prensa por jugador titular (puede venir vacía).

        Returns:
            Reporte completo con TIF ∈ [0.80, 1.20], desglose por jugador
            y el backend utilizado.
        """
        player_stats: list[tuple[float, float, float, PlayerSentiment]] = []
        articles_total = 0
        for player, articles in news_by_player.items():
            raw, var_of_mean, n_eff, topics = self._player_polarity(articles)
            articles_total += len(articles)
            sentiment = PlayerSentiment(
                player=player,
                articles_analyzed=len(articles),
                raw_polarity=raw,
                shrunk_polarity=raw,  # provisional; se contrae abajo
                topics=topics,
            )
            player_stats.append((raw, var_of_mean, n_eff, sentiment))

        shrunk_players = self._shrink(player_stats)
        if articles_total == 0:
            # Sin evidencia: el posterior colapsa al prior neutro exacto.
            return TacticalImpactReport(
                team_code=team_code, tif=1.0, team_polarity=0.0,
                players=tuple(shrunk_players), articles_total=0,
                engine="neutral-fallback")

        team_polarity = (sum(p.shrunk_polarity for p in shrunk_players)
                         / len(shrunk_players)) if shrunk_players else 0.0
        return TacticalImpactReport(
            team_code=team_code,
            tif=self.scale_to_tif(team_polarity),
            team_polarity=team_polarity,
            players=tuple(shrunk_players),
            articles_total=articles_total,
            engine=self.backend.name,
        )

    @staticmethod
    def scale_to_tif(team_polarity: float) -> float:
        """Mapeo lineal simétrico de polaridad de equipo a multiplicador TIF.

        TIF = 1 + h·p con h = (1.20 − 0.80)/2 = 0.20 (cota contractual).
        El clip final es defensa numérica: con p ∈ [-1, 1] el mapeo ya cae
        dentro del rango.

        Args:
            team_polarity: Polaridad agregada del plantel en [-1, 1].

        Returns:
            Multiplicador en [0.80, 1.20].
        """
        tif = 1.0 + config.TIF_HALF_RANGE * team_polarity
        return min(config.TIF_UPPER_BOUND, max(config.TIF_LOWER_BOUND, tif))

    # ------------------------------------------------------------------
    # Internos estadísticos
    # ------------------------------------------------------------------
    def _player_polarity(self, articles: list[NewsArticle]
                         ) -> tuple[float, float, float, dict[str, int]]:
        """Media ponderada por recencia de la polaridad de un jugador.

        Pesos: w_i = exp(−ln2 · edad_horas / vida_media). Devuelve además
        la varianza de la media ponderada (para el shrinkage) y el tamaño
        muestral efectivo de Kish n_eff = (Σw)²/Σw².

        Args:
            articles: Prensa del jugador.

        Returns:
            ``(media, var_de_la_media, n_eff, conteo_de_tópicos)``.
        """
        if not articles:
            return 0.0, 0.0, 0.0, {}
        weights: list[float] = []
        polarities: list[float] = []
        topics: dict[str, int] = {}
        for art in articles:
            age_h = max(0.0, (self._now - art.published_at).total_seconds() / 3600.0)
            weights.append(math.exp(-math.log(2.0) * age_h / self.half_life_hours))
            polarities.append(self.backend.polarity(art.full_text))
            for t in art.matched_topics:
                topics[t] = topics.get(t, 0) + 1
        w_sum = sum(weights)
        if w_sum <= 0.0:
            return 0.0, 0.0, 0.0, topics
        mean = sum(w * p for w, p in zip(weights, polarities)) / w_sum
        n_eff = (w_sum ** 2) / sum(w * w for w in weights)
        if n_eff > 1.0:
            var_within = (sum(w * (p - mean) ** 2 for w, p in zip(weights, polarities))
                          / w_sum)
            var_of_mean = var_within / n_eff
        else:
            # Un solo artículo efectivo: varianza no estimable desde dentro;
            # se usa la varianza máxima de una variable acotada en [-1, 1]
            # (cota de Popoviciu: (b−a)²/4 = 1), el shrinkage será fuerte.
            var_of_mean = 1.0
        return mean, var_of_mean, n_eff, topics

    @staticmethod
    def _shrink(player_stats: list[tuple[float, float, float, PlayerSentiment]]
                ) -> list[PlayerSentiment]:
        """Shrinkage bayesiano empírico hacia el prior neutro N(0, τ²).

        Como el prior está centrado en 0 (no en la gran media), τ² se
        estima por momentos con el SEGUNDO MOMENTO ALREDEDOR DE 0:
        E[m²] = τ² + ruido ⇒ τ̂² = max(0, media(m_j²) − media(var_j)).
        Así, un plantel con prensa unánimemente positiva conserva su señal
        (consenso ≠ ausencia de señal), mientras que medias pequeñas e
        indistinguibles del ruido de muestreo colapsan al neutro. El factor
        de contracción por jugador es el clásico B_j = τ̂²/(τ̂² + var_j).

        Args:
            player_stats: Tuplas ``(media, var_de_la_media, n_eff, sentimiento)``.

        Returns:
            Sentimientos con ``shrunk_polarity`` actualizado.
        """
        with_news = [(m, v) for m, v, n, _ in player_stats if n > 0]
        if with_news:
            second_moment = sum(m * m for m, _ in with_news) / len(with_news)
            mean_noise = sum(v for _, v in with_news) / len(with_news)
            tau2 = max(0.0, second_moment - mean_noise)
        else:
            tau2 = 0.0

        shrunk: list[PlayerSentiment] = []
        for mean, var_of_mean, n_eff, sentiment in player_stats:
            if n_eff <= 0:
                factor = 0.0
            elif tau2 + var_of_mean <= 0:
                factor = 0.0
            else:
                factor = tau2 / (tau2 + var_of_mean)
            shrunk.append(PlayerSentiment(
                player=sentiment.player,
                articles_analyzed=sentiment.articles_analyzed,
                raw_polarity=mean,
                shrunk_polarity=factor * mean,
                topics=sentiment.topics,
            ))
        return shrunk
