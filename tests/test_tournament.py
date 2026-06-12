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
