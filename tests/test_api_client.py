"""Tests del cliente de datos: integridad del fallback, resolución y cascada."""

from __future__ import annotations

import pytest

from src.api_client import (
    FallbackDataStore,
    SportsProviderFactory,
    WorldCupDataEngine,
    build_news_query,
    tag_topics,
)
from src.exceptions import TeamResolutionError
from src.models import DataOrigin


# ---------------------------------------------------------------------------
# Integridad del Fallback Data Store (datos reales FIFA)
# ---------------------------------------------------------------------------
def test_edition_goal_conservation() -> None:
    """En cada edición: sum(GF) == sum(GA) == goles totales oficiales."""
    store = FallbackDataStore()
    for year, edition in store.editions().items():
        gf = sum(t["gf"] for t in edition["teams"].values())
        ga = sum(t["ga"] for t in edition["teams"].values())
        assert gf == ga == edition["goals_total"], f"Edición {year} inconsistente"
        assert len(edition["teams"]) == 32
        assert sum(t["matches"] for t in edition["teams"].values()) == 2 * 64


def test_group_plus_knockout_goals_reconcile() -> None:
    """goles_grupo + goles_finales(eliminatorias) == total de la edición."""
    store = FallbackDataStore()
    knockouts = store.knockout_matches()
    for year, edition in store.editions().items():
        ko_goals = sum(m["score_final"][0] + m["score_final"][1]
                       for m in knockouts if str(m["edition"]) == year)
        assert edition["group_stage"]["goals"] + ko_goals == edition["goals_total"]


def test_knockout_sample_size() -> None:
    store = FallbackDataStore()
    matches = store.knockout_matches()
    assert len(matches) == 48  # 16 cruces × 3 ediciones
    for m in matches:
        assert m["decided_by"] in {"REG", "AET", "PEN"}
        if m["decided_by"] == "PEN":
            assert m["score_final"][0] == m["score_final"][1]
        h90, a90 = m["score_90"]
        hf, af = m["score_final"]
        assert hf >= h90 and af >= a90  # el global nunca decrece tras 90'


def test_team_aggregate_across_editions() -> None:
    store = FallbackDataStore()
    arg = store.team_aggregate("ARG")
    assert arg == {"matches": 18, "gf": 29, "ga": 21, "editions": 3}
    assert store.team_aggregate("JOR") is None  # sin historia en la muestra


# ---------------------------------------------------------------------------
# Tendencias históricas (adaptador Zafronix)
# ---------------------------------------------------------------------------
def test_knockout_tendencies_are_sample_statistics() -> None:
    engine = WorldCupDataEngine()
    t = engine.get_knockout_tendencies()
    assert t.matches == 48
    assert t.editions == (2014, 2018, 2022)
    assert 0.0 < t.draw_rate_90 < 1.0
    assert t.goals_per_match_90 > 0
    assert set(t.stage_goal_means) == {"R16", "QF", "SF", "3RD", "F"}


# ---------------------------------------------------------------------------
# Resolución de nombres
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("México", "MEX"),
    ("mexico", "MEX"),
    ("Países Bajos", "NED"),
    ("netherlands", "NED"),
    ("ARG", "ARG"),
    ("arg", "ARG"),
    ("Inglaterra", "ENG"),
    ("Corea del Sur", "KOR"),
    ("Estados Unidos", "USA"),
])
def test_resolve_team_aliases(query: str, expected: str) -> None:
    engine = WorldCupDataEngine()
    assert engine.resolve_team(query) == expected


def test_resolution_error_carries_suggestions() -> None:
    engine = WorldCupDataEngine()
    with pytest.raises(TeamResolutionError):
        engine.resolve_team("Reino de Xyzlandia")


# ---------------------------------------------------------------------------
# Cascada de degradación sin credenciales
# ---------------------------------------------------------------------------
def test_factory_returns_none_without_credentials() -> None:
    assert SportsProviderFactory.create() is None


def test_aggregate_falls_back_without_keys() -> None:
    engine = WorldCupDataEngine()
    agg, origin = engine.get_team_aggregate("BRA")
    assert origin is DataOrigin.FALLBACK_STORE
    assert agg["matches"] == 17  # 7 + 5 + 5
    assert agg["gf"] == 27       # 11 + 8 + 8


def test_starting_xi_fallback_and_unavailable() -> None:
    engine = WorldCupDataEngine()
    xi, origin = engine.get_starting_xi("ARG")
    assert len(xi) == 11
    assert origin is DataOrigin.FALLBACK_STORE
    none_xi, _ = engine.get_starting_xi("JOR")
    assert none_xi == []


def test_news_without_keys_is_empty_not_crash() -> None:
    engine = WorldCupDataEngine()
    news = engine.get_player_news(["Lionel Messi"])
    assert news == {"Lionel Messi": []}


def test_market_fallback_is_outright_derived() -> None:
    engine = WorldCupDataEngine()
    snap = engine.get_market_snapshot("ARG", "FRA")
    assert snap is not None
    assert snap.is_outright_derived is True
    assert snap.draw_prob is None
    assert snap.home_prob + snap.away_prob == pytest.approx(1.0)
    assert snap.origin is DataOrigin.FALLBACK_STORE


def test_market_fallback_unlisted_team_gets_field_floor() -> None:
    engine = WorldCupDataEngine()
    snap = engine.get_market_snapshot("ESP", "JOR")
    assert snap is not None
    assert snap.home_prob > snap.away_prob  # favorito claro sobre la cuota piso


# ---------------------------------------------------------------------------
# Utilidades de noticias
# ---------------------------------------------------------------------------
def test_news_query_respects_newsapi_limit() -> None:
    q = build_news_query("Kylian Mbappé")
    assert '"Kylian Mbappé"' in q
    assert len(q) <= 500


def test_topic_tagging() -> None:
    topics = tag_topics("Star striker suffers hamstring injury, "
                        "dressing room crisis grows")
    assert "lesion" in topics
    assert "crisis_vestuario" in topics


# ---------------------------------------------------------------------------
# RecentFormProvider (offline / graceful degradation)
# ---------------------------------------------------------------------------
def test_recent_form_provider_offline_fallback() -> None:
    """Sin API key, is_available() == False y get_recent_form devuelve 1.0."""
    from src.api_client import RecentFormProvider

    provider = RecentFormProvider(api_key=None)
    assert provider.is_available() is False
    # Sin key -> forma neutra para todos los equipos
    assert provider.get_recent_form("ARG") == 1.0
    assert provider.get_recent_form("BRA") == 1.0
    assert provider.get_recent_form("XYZ") == 1.0  # código desconocido


def test_recent_form_provider_unknown_team_id_neutral() -> None:
    """Team codes without a verified ID return 1.0 even with API key set."""
    from src.api_client import RecentFormProvider

    # "ECU" está en el mapa con None -> siempre devuelve 1.0
    provider = RecentFormProvider(api_key="fake_key_for_test")
    assert provider.is_available() is True
    # ECU tiene team_id=None en el mapa, debe devolver forma neutra
    assert provider.get_recent_form("ECU") == 1.0


def test_recent_form_provider_form_range() -> None:
    """_fetch_form computes values strictly within [FORM_LOWER, FORM_UPPER]."""
    from src.api_client import RecentFormProvider
    from unittest.mock import MagicMock, patch

    provider = RecentFormProvider(api_key="test_key")

    # Simula 10 victorias seguidas -> ratio=1.0 -> form=FORM_UPPER
    all_wins_response = {
        "matches": [
            {
                "status": "FINISHED",
                "homeTeam": {"id": 64},
                "awayTeam": {"id": 999},
                "score": {"winner": "HOME_TEAM"},
            }
        ] * 10
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = all_wins_response

    with patch.object(provider._session, "get", return_value=mock_resp):
        form = provider._fetch_form(64, 10)

    assert form == RecentFormProvider.FORM_UPPER

    # Simula 10 derrotas seguidas -> ratio=0.0 -> form=FORM_LOWER
    all_losses_response = {
        "matches": [
            {
                "status": "FINISHED",
                "homeTeam": {"id": 999},
                "awayTeam": {"id": 64},
                "score": {"winner": "HOME_TEAM"},
            }
        ] * 10
    }
    mock_resp.json.return_value = all_losses_response

    with patch.object(provider._session, "get", return_value=mock_resp):
        form = provider._fetch_form(64, 10)

    assert form == RecentFormProvider.FORM_LOWER


def test_world_cup_engine_get_recent_form_offline() -> None:
    """WorldCupDataEngine.get_recent_form returns 1.0 without API key."""
    engine = WorldCupDataEngine()
    # Sin FOOTBALL_DATA_API_KEY en el entorno el resultado debe ser 1.0
    form = engine.get_recent_form("ARG")
    assert form == 1.0
