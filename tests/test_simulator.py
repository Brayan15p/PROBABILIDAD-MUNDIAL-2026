"""Tests del motor matemático (estimadores, bivariado, MC, blending)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from src.api_client import FallbackDataStore
from src.exceptions import SimulationError
from src.models import DataOrigin, LambdaBreakdown, MarketSnapshot
from src.simulator import (
    BivariatePoissonModel,
    MarketBlender,
    MonteCarloEngine,
    StrengthEstimator,
    backtest_leave_one_edition_out,
)


def _breakdown(value: float) -> LambdaBreakdown:
    return LambdaBreakdown(tournament_mu=1.0, attack=1.0, opponent_defense=1.0,
                           tif=1.0, market_factor=1.0, value=value)


# ---------------------------------------------------------------------------
# Estimadores
# ---------------------------------------------------------------------------
def test_tournament_mu_is_poisson_mle() -> None:
    store = FallbackDataStore()
    totals = store.tournament_totals()
    mu = StrengthEstimator.tournament_mu(totals["goals"], totals["matches"])
    assert totals["goals"] == 171 + 169 + 172
    assert totals["matches"] == 192
    assert mu == pytest.approx(512 / 384)


def test_empirical_bayes_directionality() -> None:
    store = FallbackDataStore()
    pool = store.all_team_aggregates()
    mu = StrengthEstimator.tournament_mu(512, 192)
    ger = StrengthEstimator.empirical_bayes_strengths(
        "GER", "Alemania", pool["GER"], pool, mu, DataOrigin.FALLBACK_STORE)
    pan = StrengthEstimator.empirical_bayes_strengths(
        "PAN", "Panamá", pool["PAN"], pool, mu, DataOrigin.FALLBACK_STORE)
    assert ger.attack_strength > 1.0          # ataque sobre la media
    assert pan.defense_resilience > 1.0       # defensa más vulnerable
    assert 0.0 < ger.shrinkage_weight <= 1.0


def test_unknown_team_collapses_to_prior() -> None:
    store = FallbackDataStore()
    pool = store.all_team_aggregates()
    metrics = StrengthEstimator.empirical_bayes_strengths(
        "JOR", "Jordania", None, pool, 1.33, DataOrigin.FALLBACK_STORE)
    assert metrics.attack_strength == 1.0
    assert metrics.defense_resilience == 1.0
    assert metrics.shrinkage_weight == 0.0


def test_shrinkage_decreases_with_smaller_sample() -> None:
    store = FallbackDataStore()
    pool = store.all_team_aggregates()
    mu = StrengthEstimator.tournament_mu(512, 192)
    big = StrengthEstimator.empirical_bayes_strengths(
        "FRA", "Francia", pool["FRA"], pool, mu, DataOrigin.FALLBACK_STORE)
    small = StrengthEstimator.empirical_bayes_strengths(
        "WAL", "Gales", pool["WAL"], pool, mu, DataOrigin.FALLBACK_STORE)
    assert big.shrinkage_weight > small.shrinkage_weight


# ---------------------------------------------------------------------------
# Modelo bivariado
# ---------------------------------------------------------------------------
def test_bivariate_moments_match_parameters() -> None:
    model = BivariatePoissonModel(1.5, 1.2, 0.3)
    rng = np.random.Generator(np.random.PCG64(7))
    h, a = model.sample(400_000, rng)
    assert h.mean() == pytest.approx(1.5, abs=0.02)
    assert a.mean() == pytest.approx(1.2, abs=0.02)
    cov = float(np.cov(h, a, ddof=1)[0, 1])
    assert cov == pytest.approx(0.3, abs=0.02)


def test_pmf_matrix_sums_to_one() -> None:
    model = BivariatePoissonModel(1.4, 1.1, 0.15)
    assert float(model.pmf_matrix().sum()) == pytest.approx(1.0, abs=1e-6)


def test_lambda3_is_projected_into_feasible_region() -> None:
    model = BivariatePoissonModel(0.9, 1.4, 5.0)  # cov imposible
    assert model.lambda3 < 0.9
    assert model.lambda1 > 0 and model.lambda2 > 0


def test_invalid_lambdas_raise() -> None:
    with pytest.raises(SimulationError):
        BivariatePoissonModel(0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def test_minimum_sims_is_enforced() -> None:
    with pytest.raises(SimulationError):
        MonteCarloEngine(14_999)


def test_monte_carlo_matches_analytic_1x2() -> None:
    model = BivariatePoissonModel(1.6, 1.1, 0.1)
    engine = MonteCarloEngine(60_000, seed=42)
    result = engine.run(model, _breakdown(1.6), _breakdown(1.1), "ARG", "FRA")
    ph, pd_, pa = model.analytic_1x2()
    tol = 4 * result.macro_standard_error
    assert result.home_win_prob == pytest.approx(ph, abs=tol)
    assert result.draw_prob == pytest.approx(pd_, abs=tol)
    assert result.away_win_prob == pytest.approx(pa, abs=tol)
    assert result.home_win_prob + result.draw_prob + result.away_win_prob == \
        pytest.approx(1.0, abs=1e-9)


def test_modal_score_is_argmax_and_validated() -> None:
    model = BivariatePoissonModel(1.3, 0.9, 0.0)
    result = MonteCarloEngine(50_000, seed=11).run(
        model, _breakdown(1.3), _breakdown(0.9), "BRA", "MEX")
    top = result.top_scorelines
    assert result.modal_score == top[0]
    assert all(top[i].probability >= top[i + 1].probability
               for i in range(len(top) - 1))
    # validación cruzada: la moda MC debe estar cerca de su pmf analítica
    assert result.modal_confidence == pytest.approx(
        result.modal_score.analytic_probability,
        abs=4 * result.modal_score.mc_standard_error)
    lo, hi = result.modal_confidence_interval
    assert lo <= result.modal_confidence <= hi


def test_reproducibility_with_seed() -> None:
    model = BivariatePoissonModel(1.2, 1.2, 0.2)
    r1 = MonteCarloEngine(20_000, seed=99).run(
        model, _breakdown(1.2), _breakdown(1.2), "A", "B")
    r2 = MonteCarloEngine(20_000, seed=99).run(
        model, _breakdown(1.2), _breakdown(1.2), "A", "B")
    assert r1.home_win_prob == r2.home_win_prob
    assert r1.modal_score == r2.modal_score


# ---------------------------------------------------------------------------
# Blending de mercado
# ---------------------------------------------------------------------------
def _outright_snapshot(ph: float, pa: float) -> MarketSnapshot:
    return MarketSnapshot(home_prob=ph, draw_prob=None, away_prob=pa,
                          source="test", as_of=datetime.now(timezone.utc),
                          origin=DataOrigin.FALLBACK_STORE,
                          is_outright_derived=True)


def test_outright_blend_full_weight_reproduces_market_ratio() -> None:
    model = BivariatePoissonModel(1.0, 1.0, 0.0)
    f_h, f_a = MarketBlender(weight=1.0).blend(model, _outright_snapshot(0.8, 0.2))
    new_h, new_a = 1.0 * f_h, 1.0 * f_a
    assert new_h / new_a == pytest.approx(4.0, rel=1e-9)
    assert new_h + new_a == pytest.approx(2.0, rel=1e-9)  # total preservado


def test_outright_blend_zero_weight_is_identity() -> None:
    model = BivariatePoissonModel(1.7, 0.9, 0.1)
    f_h, f_a = MarketBlender(weight=0.0).blend(model, _outright_snapshot(0.9, 0.1))
    assert f_h == pytest.approx(1.0, rel=1e-9)
    assert f_a == pytest.approx(1.0, rel=1e-9)


def test_h2h_blend_zero_weight_converges_to_identity() -> None:
    model = BivariatePoissonModel(1.5, 1.0, 0.1)
    snap = MarketSnapshot(home_prob=0.7, draw_prob=0.2, away_prob=0.1,
                          source="test", as_of=datetime.now(timezone.utc),
                          origin=DataOrigin.LIVE_API, is_outright_derived=False)
    f_h, f_a = MarketBlender(weight=0.0).blend(model, snap)
    assert f_h == pytest.approx(1.0, abs=5e-3)
    assert f_a == pytest.approx(1.0, abs=5e-3)


def test_h2h_blend_moves_towards_market() -> None:
    model = BivariatePoissonModel(1.0, 1.0, 0.0)
    snap = MarketSnapshot(home_prob=0.6, draw_prob=0.25, away_prob=0.15,
                          source="test", as_of=datetime.now(timezone.utc),
                          origin=DataOrigin.LIVE_API, is_outright_derived=False)
    f_h, f_a = MarketBlender(weight=0.5).blend(model, snap)
    blended = BivariatePoissonModel(1.0 * f_h, 1.0 * f_a, 0.0)
    ph, _, pa = blended.analytic_1x2()
    ph0, _, pa0 = model.analytic_1x2()
    assert ph > ph0  # se movió hacia el favorito del mercado
    assert pa < pa0


def test_calibrate_weight_recovers_perfect_market() -> None:
    records = [((1 / 3, 1 / 3, 1 / 3), (0.9, 0.05, 0.05), 0)] * 25
    assert MarketBlender.calibrate_weight(records) == pytest.approx(1.0)


def test_market_weight_validation() -> None:
    with pytest.raises(SimulationError):
        MarketBlender(weight=1.5)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def test_backtest_structure_and_sanity() -> None:
    report = backtest_leave_one_edition_out(FallbackDataStore())
    assert set(report["per_edition"]) == {"2014", "2018", "2022"}
    agg = report["aggregate"]
    assert agg["matches"] == 48
    assert 0.0 < agg["log_loss"] < 1.6  # finito y en rango razonable
    assert 0.0 < agg["brier"] < 1.0
    assert report["baselines"]["uniform_log_loss"] == pytest.approx(math.log(3))
