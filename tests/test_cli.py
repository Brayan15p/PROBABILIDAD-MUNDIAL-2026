"""Tests de la CLI (modo offline: cero red)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from src.cli import cli


def test_equipos_lists_registry() -> None:
    result = CliRunner().invoke(cli, ["equipos"])
    assert result.exit_code == 0
    assert "ARG" in result.output
    assert "CONMEBOL" in result.output


def test_predecir_rejects_low_sims() -> None:
    result = CliRunner().invoke(cli, [
        "predecir", "--local", "Argentina", "--visitante", "Francia",
        "--sims", "1000", "--offline"])
    assert result.exit_code != 0
    assert "15" in result.output  # menciona el mínimo contractual


def test_predecir_offline_renders_report() -> None:
    result = CliRunner().invoke(cli, [
        "predecir", "--local", "Argentina", "--visitante", "Francia",
        "--sims", "15000", "--offline", "--seed", "7", "--no-db"])
    assert result.exit_code == 0, result.output
    assert "Probabilidades macro" in result.output
    assert "Marcador exacto" in result.output


def test_predecir_json_contract() -> None:
    result = CliRunner().invoke(cli, [
        "predecir", "--local", "México", "--visitante", "Brasil",
        "--sims", "15000", "--offline", "--seed", "3", "--no-db", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["match"] == {"home": "MEX", "away": "BRA"}
    macro = payload["macro_probabilities"]
    total = macro["home_win"] + macro["draw"] + macro["away_win"]
    assert abs(total - 1.0) < 2e-6  # tolerancia: 3 valores × redondeo a 6 decimales
    assert payload["n_simulations"] == 15000
    assert "modal_score" in payload and "score" in payload["modal_score"]
    assert payload["context"]["estimation_method"] in {
        "empirical-bayes", "poisson-glm+eb",
        "empirical-bayes+fifa-prior", "poisson-glm+eb+fifa-prior"}


def test_predecir_offline_with_live_odds_uses_fallback_market() -> None:
    result = CliRunner().invoke(cli, [
        "predecir", "--local", "España", "--visitante", "Alemania",
        "--sims", "15000", "--offline", "--live-odds", "--seed", "5",
        "--no-db", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["context"]["market_source"] is not None
    assert "outright-consensus" in payload["context"]["market_source"]


def test_unknown_team_is_friendly_error() -> None:
    result = CliRunner().invoke(cli, [
        "predecir", "--local", "Xyzlandia", "--visitante", "Francia",
        "--sims", "15000", "--offline", "--no-db"])
    assert result.exit_code != 0
    assert "no reconocido" in result.output


def test_calibrar_runs_backtest() -> None:
    result = CliRunner().invoke(cli, ["calibrar"])
    assert result.exit_code == 0, result.output
    assert "2014" in result.output and "Agregado" in result.output


def test_cache_stats_command() -> None:
    result = CliRunner().invoke(cli, ["cache", "stats"])
    assert result.exit_code == 0
