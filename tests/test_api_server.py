"""Tests básicos del servidor FastAPI REST.

Verifica que los endpoints responden correctamente y que los invariantes
del motor (suma de probabilidades = 1, mínimo de sims) se cumplen.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api_server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "engine" in body


def test_predict_offline():
    r = client.post(
        "/predict",
        json={"home": "ARG", "away": "FRA", "sims": 15000, "seed": 42, "offline": True},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "macro_probabilities" in data
    mp = data["macro_probabilities"]
    total = mp["home_win"] + mp["draw"] + mp["away_win"]
    assert abs(total - 1.0) < 1e-4, f"Probabilidades no suman 1: {total}"
    assert "modal_score" in data
    assert "lambdas" in data
    assert "top_scorelines" in data
    assert "context" in data
    assert data["n_simulations"] == 15000


def test_predict_low_sims_rejected():
    r = client.post(
        "/predict",
        json={"home": "ARG", "away": "FRA", "sims": 1000},
    )
    assert r.status_code == 422


def test_list_teams():
    r = client.get("/teams")
    assert r.status_code == 200
    teams = r.json()
    assert "ARG" in teams
    # Each entry should have names and confederation
    assert "names" in teams["ARG"]
    assert "confederation" in teams["ARG"]


def test_tournament_groups():
    r = client.get("/tournament/groups")
    assert r.status_code == 200
    body = r.json()
    assert "groups" in body
    groups = body["groups"]
    # Must have 12 groups with 4 teams each
    assert len(groups) == 12
    for letter, teams in groups.items():
        assert len(teams) == 4, f"Grupo {letter} tiene {len(teams)} equipos, esperaba 4"
    assert body["total_teams"] == 48
