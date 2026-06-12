"""Tests del procesador semántico táctico (TIF)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src import config
from src.models import NewsArticle
from src.nlp_processor import SentimentBackend, TacticalSentimentAnalyzer

NOW = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)


class FixedBackend(SentimentBackend):
    """Backend determinista para tests (sin léxico)."""

    name = "fixed"

    def __init__(self, value: float) -> None:
        self._value = value

    def polarity(self, text: str) -> float:
        return self._value


def article(player: str, hours_ago: float, text: str = "noticia") -> NewsArticle:
    return NewsArticle(
        player=player, title=text, description="", content="",
        source="test", published_at=NOW - timedelta(hours=hours_ago),
        url=f"https://x/{player}/{hours_ago}", provider="test")


def analyzer(value: float) -> TacticalSentimentAnalyzer:
    return TacticalSentimentAnalyzer(backend=FixedBackend(value), now=NOW)


def test_no_news_yields_exact_neutral_tif() -> None:
    report = analyzer(0.9).team_report("ARG", {"Messi": [], "Alvarez": []})
    assert report.tif == 1.0
    assert report.engine == "neutral-fallback"
    assert report.articles_total == 0


def test_tif_bounds_are_contractual() -> None:
    arts = {f"J{i}": [article(f"J{i}", h) for h in (1, 2, 3, 4, 5)]
            for i in range(11)}
    high = analyzer(1.0).team_report("BRA", arts)
    low = analyzer(-1.0).team_report("BRA", arts)
    assert config.TIF_LOWER_BOUND <= low.tif < 1.0 < high.tif <= config.TIF_UPPER_BOUND


def test_tif_symmetry() -> None:
    arts = {f"J{i}": [article(f"J{i}", h) for h in (1, 2, 3)] for i in range(5)}
    up = analyzer(0.6).team_report("FRA", arts)
    down = analyzer(-0.6).team_report("FRA", arts)
    assert abs((up.tif - 1.0) + (down.tif - 1.0)) < 1e-9  # simétrico alrededor de 1


def test_scale_mapping_is_linear_with_half_range() -> None:
    assert TacticalSentimentAnalyzer.scale_to_tif(0.0) == 1.0
    assert TacticalSentimentAnalyzer.scale_to_tif(1.0) == config.TIF_UPPER_BOUND
    assert TacticalSentimentAnalyzer.scale_to_tif(-1.0) == config.TIF_LOWER_BOUND


def test_single_article_is_heavily_shrunk() -> None:
    """Un solo artículo (varianza no estimable) no debe mover mucho el TIF."""
    one = analyzer(0.9).team_report("ENG", {"Kane": [article("Kane", 1)]})
    many = analyzer(0.9).team_report(
        "ENG", {"Kane": [article("Kane", h) for h in range(1, 9)]})
    assert abs(one.tif - 1.0) <= abs(many.tif - 1.0)


def test_recency_decay_weights_old_news_less() -> None:
    class SplitBackend(SentimentBackend):
        name = "split"

        def polarity(self, text: str) -> float:
            return 1.0 if "fresca" in text else -1.0

    backend = SplitBackend()
    fresh_pos = {"X": [article("X", 1, "fresca buena"),
                       article("X", 500, "vieja mala")]}
    report = TacticalSentimentAnalyzer(backend=backend, now=NOW).team_report(
        "GER", fresh_pos)
    assert report.team_polarity > 0  # la noticia fresca domina


def test_vader_backend_smoke() -> None:
    from src.nlp_processor import VaderBackend

    v = VaderBackend()
    assert v.polarity("Star striker suffers terrible injury, ruled out") < 0
    assert v.polarity("Brilliant performance, amazing scoring streak continues") > 0
