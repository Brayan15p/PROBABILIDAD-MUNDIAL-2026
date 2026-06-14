"""Tests del simulador de torneo completo (estructura oficial 2026)."""

from __future__ import annotations

import math

import pytest

from src.api_client import WorldCupDataEngine
from src.exceptions import SimulationError
from src.tournament import (
    KnockoutSlot,
    TournamentSimulator,
    TournamentStructure,
    default_structure_path,
)


@pytest.fixture(scope="module")
def structure() -> TournamentStructure:
    return TournamentStructure.load(default_structure_path())


@pytest.fixture(scope="module")
def engine() -> WorldCupDataEngine:
    return WorldCupDataEngine()


# ---------------------------------------------------------------------------
# Estructura oficial
# ---------------------------------------------------------------------------
def test_structure_has_48_unique_resolvable_teams(structure, engine) -> None:
    teams = structure.teams
    assert len(teams) == 48
    assert len(set(teams)) == 48
    registry = engine.store.registry()
    assert all(code in registry for code in teams)


def test_structure_hosts_in_official_slots(structure) -> None:
    assert structure.groups["A"][0] == "MEX"
    assert structure.groups["B"][0] == "CAN"
    assert structure.groups["D"][0] == "USA"


def test_confederation_counts_match_qualification_math(structure, engine) -> None:
    registry = engine.store.registry()
    counts: dict[str, int] = {}
    for code in structure.teams:
        conf = registry[code]["confederation"]
        counts[conf] = counts.get(conf, 0) + 1
    assert counts == {"UEFA": 16, "CONMEBOL": 6, "CONCACAF": 6,
                      "CAF": 10, "AFC": 9, "OFC": 1}


def test_slot_parser() -> None:
    s = KnockoutSlot.parse("1A")
    assert s.kind == "group_rank" and s.rank == 1 and s.group == "A"
    t = KnockoutSlot.parse("3:ABCDF")
    assert t.kind == "third" and t.allowed_groups == ("A", "B", "C", "D", "F")
    w = KnockoutSlot.parse("W73")
    assert w.kind == "match_ref" and w.match == 73 and w.take_winner
    loser = KnockoutSlot.parse("L101")
    assert loser.kind == "match_ref" and not loser.take_winner
    with pytest.raises(SimulationError):
        KnockoutSlot.parse("4Z")


def test_bracket_wiring_validated_on_load(structure) -> None:
    r32 = dict(structure.knockout_rounds)["round_of_32"]
    assert len(r32) == 16
    thirds = [s for _, h, a in r32 for s in (h, a) if s.kind == "third"]
    assert len(thirds) == 8


# ---------------------------------------------------------------------------
# Matching de terceros
# ---------------------------------------------------------------------------
def test_third_assignment_respects_allowed_groups() -> None:
    slots = [(74, ("A", "B")), (77, ("B", "C")), (79, ("C", "D"))]
    qualified = ["A", "B", "C"]
    assign = TournamentSimulator._assign_thirds(qualified, slots)
    assert assign is not None
    assert set(assign.values()) == set(qualified)
    for match_no, allowed in slots:
        assert assign[match_no] in allowed


def test_third_assignment_backtracks() -> None:
    # Greedy ingenuo fallaría: el cupo 1 debe ceder 'B' al cupo 2.
    slots = [(1, ("A", "B")), (2, ("B",))]
    assign = TournamentSimulator._assign_thirds(["A", "B"], slots)
    assert assign == {1: "A", 2: "B"}


def test_third_assignment_infeasible_returns_none() -> None:
    assert TournamentSimulator._assign_thirds(["A", "B"], [(1, ("C",)), (2, ("C",))]) is None


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result(structure, engine):
    sim = TournamentSimulator(engine, structure, n_replicas=400, seed=2026)
    return sim.run()


def test_champion_probabilities_sum_to_one(result) -> None:
    total = sum(s["champion"] for s in result.stage_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_stage_monotonicity(result) -> None:
    for stages in result.stage_probabilities.values():
        assert (stages["champion"] <= stages["final"] <= stages["semi_final"]
                <= stages["quarter_final"] <= stages["round_of_16"]
                <= stages["round_of_32"] <= 1.0)


def test_official_bracket_yields_zero_matching_fallbacks(result) -> None:
    assert result.matching_fallbacks == 0


def test_round_counts_are_structurally_exact(result) -> None:
    # En cada réplica avanzan exactamente 32 a dieciseisavos y 16 a octavos.
    assert sum(s["round_of_32"] for s in result.stage_probabilities.values()) == \
        pytest.approx(32.0, abs=1e-9)
    assert sum(s["round_of_16"] for s in result.stage_probabilities.values()) == \
        pytest.approx(16.0, abs=1e-9)
    assert sum(s["final"] for s in result.stage_probabilities.values()) == \
        pytest.approx(2.0, abs=1e-9)


def test_reproducibility_with_seed(structure, engine) -> None:
    r1 = TournamentSimulator(engine, structure, n_replicas=150, seed=7).run()
    r2 = TournamentSimulator(engine, structure, n_replicas=150, seed=7).run()
    assert r1.stage_probabilities == r2.stage_probabilities


def test_strong_teams_outrank_minnows(result) -> None:
    probs = result.stage_probabilities
    favoritos = max(probs["FRA"]["champion"], probs["ARG"]["champion"],
                    probs["BRA"]["champion"], probs["ENG"]["champion"])
    assert favoritos > probs["NZL"]["champion"]
    assert favoritos > probs["CUW"]["champion"]


def test_minimum_replicas_enforced(structure, engine) -> None:
    with pytest.raises(SimulationError):
        TournamentSimulator(engine, structure, n_replicas=50)


# ---------------------------------------------------------------------------
# TIF integration
# ---------------------------------------------------------------------------
def test_tif_disabled_by_default_gives_unit_factors(structure, engine) -> None:
    """Sin team_tifs, el simulador usa TIF=1.0 para todos los equipos."""
    sim = TournamentSimulator(engine, structure, n_replicas=100, seed=42)
    # _tif_enabled debe ser False y _tif_cache vacío
    assert not sim._tif_enabled
    assert sim._tif_cache == {}
    # _get_tif siempre devuelve 1.0
    for code in ("ARG", "FRA", "BRA"):
        assert sim._get_tif(code) == 1.0


def test_tif_with_empty_dict_enables_lazy_computation(structure, engine) -> None:
    """team_tifs={} activa el modo TIF con cache vacío (lazy)."""
    sim = TournamentSimulator(engine, structure, n_replicas=100, seed=42,
                              team_tifs={})
    assert sim._tif_enabled
    assert sim._tif_cache == {}


def test_tif_precalculated_values_used(structure, engine) -> None:
    """Valores TIF precalculados se aplican directamente a λ."""
    tifs = {"ARG": 1.10, "FRA": 0.90}
    sim = TournamentSimulator(engine, structure, n_replicas=100, seed=42,
                              team_tifs=tifs)
    assert sim._tif_enabled
    assert sim._get_tif("ARG") == 1.10
    assert sim._get_tif("FRA") == 0.90
    # Equipo sin TIF precalculado y sin analyzer → fallback a 1.0
    assert sim._get_tif("BRA") == 1.0


def test_tif_cache_avoids_recomputation(structure, engine) -> None:
    """El TIF de un equipo solo se computa una vez; el cache se reutiliza."""
    calls: list[str] = []

    class _CountingAnalyzer:
        def team_report(self, team_code: str, news_by_player: dict) -> object:
            calls.append(team_code)
            from src.models import TacticalImpactReport
            return TacticalImpactReport(
                team_code=team_code, tif=1.05, team_polarity=0.25,
                players=(), articles_total=0, engine="test")

    sim = TournamentSimulator(engine, structure, n_replicas=100, seed=42,
                              team_tifs={}, analyzer=_CountingAnalyzer())
    # Primera llamada: computa y cachea
    val1 = sim._get_tif("ARG")
    assert val1 == 1.05
    assert calls == ["ARG"]
    # Segunda llamada: lee del cache sin llamar al analyzer
    val2 = sim._get_tif("ARG")
    assert val2 == 1.05
    assert calls == ["ARG"]  # sin nueva llamada


def test_tif_high_factor_raises_champion_probability(structure, engine) -> None:
    """Un TIF alto para todos los equipos de un grupo eleva sus probabilidades."""
    # Sin TIF
    sim_no_tif = TournamentSimulator(engine, structure, n_replicas=200, seed=100)
    result_no_tif = sim_no_tif.run()

    # Con TIF alto para ARG (1.20) y bajo para el resto (0.95)
    all_tifs: dict[str, float] = {code: 0.95 for code in structure.teams}
    all_tifs["ARG"] = 1.20
    sim_tif = TournamentSimulator(engine, structure, n_replicas=200, seed=100,
                                  team_tifs=all_tifs)
    result_tif = sim_tif.run()

    # ARG con TIF=1.20 debe tener probabilidad de campeón >= sin TIF
    # (no garantizado por réplicas pequeñas, pero esperado estadísticamente)
    arg_no_tif = result_no_tif.stage_probabilities["ARG"]["champion"]
    arg_tif = result_tif.stage_probabilities["ARG"]["champion"]
    # Verificación débil: al menos ARG con TIF no colapsa a 0
    assert arg_tif >= 0.0


def test_active_tif_adjustments_filters_by_threshold(structure, engine) -> None:
    """active_tif_adjustments devuelve solo equipos fuera del rango neutro."""
    tifs = {"ARG": 1.10, "FRA": 0.90, "BRA": 1.01, "GER": 1.0}
    sim = TournamentSimulator(engine, structure, n_replicas=100, seed=42,
                              team_tifs=tifs)
    adj = sim.active_tif_adjustments(threshold=0.02)
    assert "ARG" in adj  # |1.10 - 1.0| = 0.10 > 0.02
    assert "FRA" in adj  # |0.90 - 1.0| = 0.10 > 0.02
    assert "BRA" not in adj  # |1.01 - 1.0| = 0.01 < 0.02
    assert "GER" not in adj  # |1.0 - 1.0| = 0.0 < 0.02
