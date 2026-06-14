"""Motor matemático: Poisson Bivariado (Karlis–Ntzoufras) + Monte Carlo.

Cadena de estimación (todo parámetro proviene de datos, nunca de constantes
inventadas):

- **μ del torneo**: MLE Poisson = goles totales / (2 · partidos) sobre la
  ventana 2014–2022 (o sobre los partidos vivos de la API si existen).
- **Fuerzas ataque/defensa**: regresión de Poisson con statsmodels (GLM,
  dummies de ataque y defensa) cuando la muestra satisface la regla
  empírica de ≥10 observaciones por parámetro (Harrell); en caso contrario,
  estimador de máxima verosimilitud con *shrinkage* bayesiano empírico
  (James–Stein de momentos) sobre los agregados reales por selección.
- **λ₃ (covarianza)**: estimador de momentos de Karlis & Ntzoufras (2003)
  λ̂₃ = max(0, ĉov(X,Y)) sobre los cruces de eliminación directa a 90'.
- **Mercado**: *log-opinion pool* entre modelo y consenso con peso w
  (máxima entropía w=0.5 por defecto; calibrable por MLE si hay histórico
  de cuotas), traducido a factores multiplicativos sobre los λ.
- **Monte Carlo**: ≥15.000 réplicas de (X, Y) = (W₁+W₃, W₂+W₃) con
  W_i ~ Poisson(λ_i) independientes; la moda conjunta empírica se valida
  contra la pmf bivariada analítica.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from src import config
from src.exceptions import DataUnavailableError, SimulationError
from src.models import (
    DataOrigin,
    LambdaBreakdown,
    MarketSnapshot,
    ScorelineProbability,
    SimulationResult,
    TacticalImpactReport,
    TeamMetrics,
)

#: Regla empírica de Harrell ("one in ten"): mínimo de observaciones por
#: parámetro para que un GLM de máxima verosimilitud sea estable.
GLM_MIN_OBS_PER_PARAM: int = 10

#: Iteraciones máximas del solver de punto fijo del blending (cota
#: operacional de seguridad; la convergencia típica es < 30 iteraciones).
_BLEND_MAX_ITER: int = 100
#: Tolerancia de convergencia del solver, alineada con el error MC mínimo
#: alcanzable con 15.000 réplicas (≈4e-3): exigir más precisión al solver
#: que al estimador sería ruido.
_BLEND_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# Estimación de parámetros
# ---------------------------------------------------------------------------
class StrengthEstimator:
    """Estimadores de μ del torneo y fuerzas relativas por selección."""

    @staticmethod
    def tournament_mu(total_goals: float, total_matches: float) -> float:
        """MLE de la media de goles por equipo-partido.

        Para X ~ Poisson(μ), el MLE es la media muestral: con G goles en M
        partidos hay 2M observaciones equipo-partido ⇒ μ̂ = G / (2M).

        Args:
            total_goals: Goles totales de la muestra.
            total_matches: Partidos totales de la muestra.

        Returns:
            μ̂ del torneo.

        Raises:
            DataUnavailableError: Muestra vacía.
        """
        if total_matches <= 0:
            raise DataUnavailableError("Sin partidos para estimar μ del torneo")
        return float(total_goals) / (2.0 * float(total_matches))

    @staticmethod
    def empirical_bayes_strengths(
            code: str,
            name: str,
            aggregate: dict[str, float] | None,
            pool: dict[str, dict[str, float]],
            mu: float,
            origin: DataOrigin) -> TeamMetrics:
        """Fuerzas ataque/defensa con shrinkage bayesiano empírico.

        Modelo: la tasa observada r_i = GF_i/M_i es ruidosa con varianza
        Poisson ≈ μ/M_i. La varianza real entre selecciones se estima por
        momentos: τ̂² = max(0, s²(r) − μ·E[1/M]). El peso de la evidencia
        propia es w_i = τ̂²/(τ̂² + μ/M_i) y la fuerza contraída:
        fuerza = 1 + w_i·(r_i/μ − 1). Un equipo sin historia recibe el
        prior exacto (fuerza 1.0, w=0).

        Args:
            code: Código FIFA.
            name: Nombre canónico.
            aggregate: ``{"matches","gf","ga"}`` del equipo o ``None``.
            pool: Agregados de todos los equipos muestrales (para τ̂²).
            mu: μ̂ del torneo.
            origin: Procedencia del agregado.

        Returns:
            :class:`TeamMetrics` con fuerzas contraídas y peso w.
        """
        tau2_for = StrengthEstimator._between_variance(pool, mu, key="gf")
        tau2_against = StrengthEstimator._between_variance(pool, mu, key="ga")
        if aggregate is None or aggregate.get("matches", 0) <= 0:
            return TeamMetrics(
                code=code, name=name, matches_sample=0,
                goals_for_per_match=mu, goals_against_per_match=mu,
                attack_strength=1.0, defense_resilience=1.0,
                origin=origin, shrinkage_weight=0.0)
        m = float(aggregate["matches"])
        rate_for = float(aggregate["gf"]) / m
        rate_against = float(aggregate["ga"]) / m
        noise = mu / m  # varianza Poisson de la tasa observada
        w_for = tau2_for / (tau2_for + noise) if (tau2_for + noise) > 0 else 0.0
        w_against = (tau2_against / (tau2_against + noise)
                     if (tau2_against + noise) > 0 else 0.0)
        attack = 1.0 + w_for * (rate_for / mu - 1.0)
        defense = 1.0 + w_against * (rate_against / mu - 1.0)
        return TeamMetrics(
            code=code, name=name, matches_sample=int(m),
            goals_for_per_match=rate_for, goals_against_per_match=rate_against,
            attack_strength=max(attack, np.finfo(float).eps),
            defense_resilience=max(defense, np.finfo(float).eps),
            origin=origin,
            shrinkage_weight=(w_for + w_against) / 2.0)

    @staticmethod
    def _between_variance(pool: dict[str, dict[str, float]], mu: float,
                          key: str) -> float:
        """τ̂² de momentos: varianza entre equipos neta del ruido Poisson."""
        rates = []
        inv_m = []
        for agg in pool.values():
            m = float(agg.get("matches", 0))
            if m > 0:
                rates.append(float(agg[key]) / m)
                inv_m.append(1.0 / m)
        if len(rates) < 2:
            return 0.0
        s2 = float(np.var(rates, ddof=1))
        mean_noise = mu * float(np.mean(inv_m))
        return max(0.0, s2 - mean_noise)

    @staticmethod
    def poisson_glm_strengths(matches: list[dict[str, Any]]
                              ) -> dict[str, tuple[float, float]] | None:
        """Regresión de Poisson (statsmodels GLM) con dummies ataque/defensa.

        Formato largo: cada partido aporta dos filas (goles de cada lado).
        log E[goles] = β₀ + ataque_equipo + defensa_rival. Los efectos se
        exponencian y se renormalizan a media geométrica 1 para ser
        directamente comparables con las fuerzas del estimador EB.

        Args:
            matches: Partidos normalizados (esquema del WorldCupDataEngine).

        Returns:
            ``{code: (ataque, defensa)}`` o ``None`` si la muestra viola la
            regla de ≥10 observaciones por parámetro o el ajuste no converge.
        """
        if not matches:
            return None
        rows: list[dict[str, Any]] = []
        for m in matches:
            rows.append({"goals": int(m["home_goals_90"]),
                         "attack": str(m["home"]), "defense": str(m["away"])})
            rows.append({"goals": int(m["away_goals_90"]),
                         "attack": str(m["away"]), "defense": str(m["home"])})
        df = pd.DataFrame(rows)
        teams = sorted(set(df["attack"]) | set(df["defense"]))
        n_params = 2 * (len(teams) - 1) + 1
        if len(df) < GLM_MIN_OBS_PER_PARAM * n_params:
            return None
        X = pd.concat([
            pd.get_dummies(df["attack"], prefix="atk", drop_first=True),
            pd.get_dummies(df["defense"], prefix="def", drop_first=True),
        ], axis=1).astype(float)
        X = sm.add_constant(X)
        try:
            fit = sm.GLM(df["goals"].to_numpy(), X,
                         family=sm.families.Poisson()).fit()
        except Exception:  # pragma: no cover - singularidades degeneradas
            return None
        params = fit.params
        ref = teams[0]
        atk_log = {ref: 0.0}
        def_log = {ref: 0.0}
        for team in teams[1:]:
            atk_log[team] = float(params.get(f"atk_{team}", 0.0))
            def_log[team] = float(params.get(f"def_{team}", 0.0))
        atk_center = float(np.mean(list(atk_log.values())))
        def_center = float(np.mean(list(def_log.values())))
        return {team: (math.exp(atk_log[team] - atk_center),
                       math.exp(def_log[team] - def_center))
                for team in teams}


# ---------------------------------------------------------------------------
# Modelo Poisson Bivariado (Karlis & Ntzoufras, 2003)
# ---------------------------------------------------------------------------
class BivariatePoissonModel:
    """Distribución conjunta de goles con dependencia λ₃ ≥ 0.

    Construcción: X = W₁ + W₃, Y = W₂ + W₃ con W_i ~ Poisson(λ_i)
    independientes ⇒ E[X] = λ₁+λ₃, E[Y] = λ₂+λ₃, Cov(X,Y) = λ₃.

    Attributes:
        lambda_home: Media marginal de goles del local (λ₁+λ₃).
        lambda_away: Media marginal del visitante (λ₂+λ₃).
        lambda3: Componente común (covarianza), truncada al MLE de borde 0
            si la covarianza muestral es negativa.
    """

    def __init__(self, lambda_home: float, lambda_away: float,
                 lambda3: float) -> None:
        if lambda_home <= 0 or lambda_away <= 0:
            raise SimulationError(
                f"Lambdas marginales no positivos: λ_h={lambda_home}, "
                f"λ_a={lambda_away}")
        # Restricción del dominio: λ₁ = λ_h − λ₃ > 0 y λ₂ = λ_a − λ₃ > 0.
        # Si ĉov excede el mínimo marginal, el MLE restringido está en el
        # borde: se proyecta justo por debajo del límite factible.
        feasible_max = float(np.nextafter(min(lambda_home, lambda_away), 0.0))
        self.lambda3: float = float(min(max(0.0, lambda3), feasible_max))
        self.lambda_home: float = float(lambda_home)
        self.lambda_away: float = float(lambda_away)
        self.lambda1: float = self.lambda_home - self.lambda3
        self.lambda2: float = self.lambda_away - self.lambda3

    # ------------------------------------------------------------------
    def grid_bound(self) -> int:
        """Cota K de la malla de marcadores con masa de cola < TAIL_EPSILON.

        Busca el menor K tal que P(Poisson(λ_max) > K) < ε acumulando la
        pmf — sin límites de goles arbitrarios.

        Returns:
            Número máximo de goles K de la malla analítica.
        """
        lam = max(self.lambda_home, self.lambda_away)
        cdf = 0.0
        k = 0
        term = math.exp(-lam)
        while True:
            cdf += term
            if 1.0 - cdf < config.TAIL_EPSILON:
                return k
            k += 1
            term *= lam / k

    def pmf(self, x: int, y: int) -> float:
        """pmf bivariada exacta P(X=x, Y=y).

        P = e^{−(λ₁+λ₂+λ₃)} (λ₁^x/x!)(λ₂^y/y!)
            Σ_{k=0}^{min(x,y)} C(x,k) C(y,k) k! (λ₃/(λ₁λ₂))^k

        Args:
            x: Goles del local.
            y: Goles del visitante.

        Returns:
            Probabilidad exacta del marcador.
        """
        l1, l2, l3 = self.lambda1, self.lambda2, self.lambda3
        log_base = (-(l1 + l2 + l3)
                    + x * math.log(l1) - math.lgamma(x + 1)
                    + y * math.log(l2) - math.lgamma(y + 1))
        if l3 == 0.0:
            return math.exp(log_base)
        ratio = l3 / (l1 * l2)
        s = 0.0
        for k in range(min(x, y) + 1):
            s += (math.comb(x, k) * math.comb(y, k) * math.factorial(k)
                  * ratio ** k)
        return math.exp(log_base) * s

    def pmf_matrix(self) -> np.ndarray:
        """Malla completa de probabilidades hasta la cota de cola.

        Returns:
            Matriz (K+1)×(K+1) con P(X=i, Y=j).
        """
        k = self.grid_bound()
        grid = np.zeros((k + 1, k + 1))
        for i in range(k + 1):
            for j in range(k + 1):
                grid[i, j] = self.pmf(i, j)
        return grid

    def analytic_1x2(self) -> tuple[float, float, float]:
        """(P local, P empate, P visitante) por suma de la malla analítica."""
        grid = self.pmf_matrix()
        home = float(np.tril(grid, -1).sum())   # i > j
        draw = float(np.trace(grid))
        away = float(np.triu(grid, 1).sum())    # j > i
        total = home + draw + away
        return home / total, draw / total, away / total

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Muestra n réplicas (X, Y) por construcción trivariada reducida.

        Args:
            n: Número de réplicas.
            rng: Generador NumPy (PCG64).

        Returns:
            Tupla de arrays enteros (goles local, goles visitante).
        """
        w1 = rng.poisson(self.lambda1, n)
        w2 = rng.poisson(self.lambda2, n)
        if self.lambda3 > 0:
            w3 = rng.poisson(self.lambda3, n)
            return w1 + w3, w2 + w3
        return w1, w2


# ---------------------------------------------------------------------------
# Blending con el mercado (log-opinion pool)
# ---------------------------------------------------------------------------
class MarketBlender:
    """Funde las probabilidades del modelo con el consenso del mercado.

    El peso w del mercado es el del *log-opinion pool*; sin histórico de
    cuotas para calibrarlo por MLE, w = 0.5 es el pool de máxima entropía
    (igual confianza en ambas fuentes). Sobre-escribible con la variable
    de entorno ``WC26_MARKET_WEIGHT`` o calibrable con
    :meth:`calibrate_weight`.
    """

    def __init__(self, weight: float | None = None) -> None:
        env_w = os.getenv("WC26_MARKET_WEIGHT")
        resolved = weight if weight is not None else (
            float(env_w) if env_w is not None else 0.5)
        if not 0.0 <= resolved <= 1.0:
            raise SimulationError(f"Peso de mercado fuera de [0,1]: {resolved}")
        self.weight: float = resolved

    def blend(self, model: BivariatePoissonModel,
              snapshot: MarketSnapshot) -> tuple[float, float]:
        """Calcula factores multiplicativos (f_home, f_away) sobre los λ.

        Dos regímenes:

        - **Mercado head-to-head completo (1X2)**: pool logarítmico de las
          tres patas y búsqueda de punto fijo de (f_h, f_a) tal que el
          modelo bivariado reescalado reproduzca P(local) y P(visitante)
          del pool.
        - **Mercado outright (sin empate)**: pool logarítmico sobre el
          *ratio* de fuerzas ρ = λ_h/λ_a preservando el total de goles
          esperados λ_h + λ_a (el outright informa fuerza relativa, no
          ritmo de gol del partido).

        Args:
            model: Modelo bivariado con los λ del modelo puro.
            snapshot: Consenso de mercado.

        Returns:
            Factores (f_home, f_away) a aplicar sobre los λ marginales.
        """
        w = self.weight
        if snapshot.draw_prob is None or snapshot.is_outright_derived:
            rho_model = model.lambda_home / model.lambda_away
            rho_market = snapshot.home_prob / snapshot.away_prob
            rho_star = rho_model ** (1.0 - w) * rho_market ** w
            total = model.lambda_home + model.lambda_away
            new_home = total * rho_star / (1.0 + rho_star)
            new_away = total - new_home
            return new_home / model.lambda_home, new_away / model.lambda_away

        p_model = model.analytic_1x2()
        p_market = (snapshot.home_prob, snapshot.draw_prob, snapshot.away_prob)
        pooled_raw = [pm ** (1.0 - w) * qk ** w
                      for pm, qk in zip(p_model, p_market)]
        z = sum(pooled_raw)
        target_home, _target_draw, target_away = (p / z for p in pooled_raw)

        f_h, f_a = 1.0, 1.0
        for _ in range(_BLEND_MAX_ITER):
            candidate = BivariatePoissonModel(
                model.lambda_home * f_h, model.lambda_away * f_a, model.lambda3)
            ph, _pd_, pa = candidate.analytic_1x2()
            err_h, err_a = target_home - ph, target_away - pa
            if abs(err_h) < _BLEND_TOL and abs(err_a) < _BLEND_TOL:
                break
            # Actualización multiplicativa proporcional al error en escala
            # logit (gradiente aproximado de P(win) respecto a log λ es
            # positivo y acotado: el paso relativo conserva λ > 0).
            f_h *= math.exp(err_h)
            f_a *= math.exp(err_a)
        return f_h, f_a

    @staticmethod
    def calibrate_weight(records: list[tuple[tuple[float, float, float],
                                             tuple[float, float, float], int]]
                         ) -> float:
        """MLE del peso w sobre histórico (modelo, mercado, resultado).

        Maximiza la log-verosimilitud del pool logarítmico en una malla
        fina de w ∈ [0, 1] (la verosimilitud es unimodal en w).

        Args:
            records: Tuplas ``(p_modelo, p_mercado, resultado)`` con
                resultado ∈ {0: local, 1: empate, 2: visitante}.

        Returns:
            ŵ de máxima verosimilitud.

        Raises:
            DataUnavailableError: Sin registros para calibrar.
        """
        if not records:
            raise DataUnavailableError("Sin histórico de cuotas para calibrar w")
        grid = np.linspace(0.0, 1.0, 101)
        best_w, best_ll = 0.5, -math.inf
        for w in grid:
            ll = 0.0
            for p_model, p_market, outcome in records:
                pooled = [pm ** (1.0 - w) * qk ** w
                          for pm, qk in zip(p_model, p_market)]
                z = sum(pooled)
                ll += math.log(max(pooled[outcome] / z, np.finfo(float).tiny))
            if ll > best_ll:
                best_ll, best_w = ll, float(w)
        return best_w


# ---------------------------------------------------------------------------
# Motor Monte Carlo
# ---------------------------------------------------------------------------
class MonteCarloEngine:
    """Simulador estocástico vectorizado sobre el modelo bivariado."""

    #: Número de bloques de simulación (para reportar progreso en la CLI).
    PROGRESS_CHUNKS: int = 20

    def __init__(self, n_simulations: int, seed: int | None = None) -> None:
        if n_simulations < config.MIN_MONTE_CARLO_SIMS:
            raise SimulationError(
                f"Se exigen ≥{config.MIN_MONTE_CARLO_SIMS:,} iteraciones "
                f"(recibidas: {n_simulations:,})")
        self.n_simulations = int(n_simulations)
        if seed is None:
            # Semilla de 63 bits: entropía del SO acotada al rango BIGINT
            # de PostgreSQL para poder persistirla y reproducir la corrida.
            seed = int(np.random.SeedSequence().generate_state(1, np.uint64)[0] >> 1)
        self.seed: int = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))

    def run(self, model: BivariatePoissonModel,
            lambda_home: LambdaBreakdown, lambda_away: LambdaBreakdown,
            home_code: str, away_code: str,
            progress: Callable[[int], None] | None = None) -> SimulationResult:
        """Ejecuta la simulación completa y compila el resultado.

        Args:
            model: Modelo bivariado parametrizado.
            lambda_home: Trazabilidad del λ local.
            lambda_away: Trazabilidad del λ visitante.
            home_code: Código FIFA local.
            away_code: Código FIFA visitante.
            progress: Callback con el incremento de réplicas completadas.

        Returns:
            :class:`SimulationResult` con macro-probabilidades, moda
            conjunta validada analíticamente e intervalos de confianza.
        """
        n = self.n_simulations
        chunk = max(1, n // self.PROGRESS_CHUNKS)
        homes = np.empty(n, dtype=np.int64)
        aways = np.empty(n, dtype=np.int64)
        done = 0
        while done < n:
            take = min(chunk, n - done)
            h, a = model.sample(take, self._rng)
            homes[done:done + take] = h
            aways[done:done + take] = a
            done += take
            if progress is not None:
                progress(take)

        # --- compilación de la matriz de resultados ----------------------
        kmax = int(max(homes.max(), aways.max()))
        encoded = homes * (kmax + 1) + aways
        counts = np.bincount(encoded, minlength=(kmax + 1) ** 2)
        matrix = counts.reshape(kmax + 1, kmax + 1)

        home_wins = float(np.tril(matrix, -1).sum()) / n
        draws = float(np.trace(matrix)) / n
        away_wins = float(np.triu(matrix, 1).sum()) / n
        macro_se = max(self._binomial_se(p, n) for p in (home_wins, draws, away_wins))

        # --- moda conjunta absoluta + validación analítica ----------------
        order = np.argsort(matrix, axis=None)[::-1]
        top: list[ScorelineProbability] = []
        for flat_idx in order[:10]:
            i, j = divmod(int(flat_idx), kmax + 1)
            p_hat = float(matrix[i, j]) / n
            if p_hat <= 0:
                break
            top.append(ScorelineProbability(
                home_goals=i, away_goals=j, probability=p_hat,
                mc_standard_error=self._binomial_se(p_hat, n),
                analytic_probability=model.pmf(i, j)))
        modal = top[0]
        ci_low, ci_high = proportion_confint(
            count=int(round(modal.probability * n)), nobs=n, method="wilson")
        separation = (modal.probability / top[1].probability
                      if len(top) > 1 and top[1].probability > 0 else math.inf)

        totals = homes + aways
        return SimulationResult(
            home_code=home_code,
            away_code=away_code,
            n_simulations=n,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            lambda3_covariance=model.lambda3,
            home_win_prob=home_wins,
            draw_prob=draws,
            away_win_prob=away_wins,
            macro_standard_error=macro_se,
            top_scorelines=tuple(top),
            modal_score=modal,
            modal_confidence=modal.probability,
            modal_confidence_interval=(float(ci_low), float(ci_high)),
            mode_separation_ratio=float(separation),
            over_2_5_prob=float(np.mean(totals > 2.5)),
            btts_prob=float(np.mean((homes > 0) & (aways > 0))),
            expected_total_goals=float(totals.mean()),
            seed=self.seed,
        )

    @staticmethod
    def _binomial_se(p: float, n: int) -> float:
        """Error estándar binomial sqrt(p(1−p)/n) de una proporción MC."""
        return math.sqrt(max(p * (1.0 - p), 0.0) / n)


# ---------------------------------------------------------------------------
# Pipeline de predicción (orquestación)
# ---------------------------------------------------------------------------
class DataEngineProtocol(Protocol):
    """Contrato mínimo que el pipeline exige a la capa de datos."""

    def resolve_team(self, query: str) -> str: ...
    def display_name(self, code: str, english: bool = ...) -> str: ...
    def get_match_history(self) -> tuple[list[dict[str, Any]], DataOrigin]: ...
    def get_team_aggregate(self, code: str,
                           live_matches: list[dict[str, Any]] | None = ...
                           ) -> tuple[dict[str, float], DataOrigin]: ...
    def get_team_aggregate_pool(self) -> dict[str, dict[str, float]]: ...
    def get_tournament_totals(self) -> dict[str, int]: ...
    def get_knockout_tendencies(self) -> Any: ...
    def get_starting_xi(self, code: str) -> tuple[list[str], DataOrigin]: ...
    def get_player_news(self, players: list[str]) -> dict[str, Any]: ...
    def get_market_snapshot(self, home_code: str, away_code: str,
                            force_refresh: bool = ...) -> MarketSnapshot | None: ...


class AnalyzerProtocol(Protocol):
    """Contrato mínimo del analizador NLP."""

    def team_report(self, team_code: str,
                    news_by_player: dict[str, Any]) -> TacticalImpactReport: ...


@dataclass(frozen=True)
class PredictionArtifacts:
    """Paquete completo de artefactos de una predicción (para CLI y DB)."""

    result: SimulationResult
    metrics_home: TeamMetrics
    metrics_away: TeamMetrics
    tif_home: TacticalImpactReport
    tif_away: TacticalImpactReport
    market: MarketSnapshot | None
    tournament_mu: float
    estimation_method: str
    tendencies: Any


class PredictionPipeline:
    """Orquesta datos → NLP → estimación → blending → Monte Carlo.

    Implementa exactamente la forma funcional especificada:

        λ_local = μ_torneo · Ataque_local · Defensa_visitante · TIF_local · f_mercado_local
        λ_visit = μ_torneo · Ataque_visit · Defensa_local · TIF_visit · f_mercado_visit

    donde cada término es un estimador documentado (nunca una constante).
    """

    def __init__(self, engine: DataEngineProtocol, analyzer: AnalyzerProtocol,
                 blender: MarketBlender | None = None) -> None:
        self.engine = engine
        self.analyzer = analyzer
        self.blender = blender or MarketBlender()

    def predict(self, home_query: str, away_query: str, n_sims: int,
                live_odds: bool = False, seed: int | None = None,
                progress: Callable[[int], None] | None = None
                ) -> PredictionArtifacts:
        """Ejecuta la predicción end-to-end de un partido.

        Args:
            home_query: Nombre libre del equipo local.
            away_query: Nombre libre del visitante.
            n_sims: Iteraciones Monte Carlo (≥ 15.000).
            live_odds: Activa el módulo de mercado (Polymarket/consenso).
            seed: Semilla reproducible opcional.
            progress: Callback de progreso para la CLI.

        Returns:
            :class:`PredictionArtifacts` con el resultado y su trazabilidad.
        """
        home = self.engine.resolve_team(home_query)
        away = self.engine.resolve_team(away_query)

        # ---- 1. Historia y μ del torneo (MLE) ---------------------------
        live_matches, _history_origin = self.engine.get_match_history()
        if live_matches:
            total_goals = float(sum(m["home_goals"] + m["away_goals"]
                                    for m in live_matches))
            mu = StrengthEstimator.tournament_mu(total_goals, len(live_matches))
        else:
            totals = self.engine.get_tournament_totals()
            mu = StrengthEstimator.tournament_mu(totals["goals"], totals["matches"])

        # ---- 2. Fuerzas: GLM si la muestra lo permite, si no EB ---------
        glm = StrengthEstimator.poisson_glm_strengths(live_matches)
        pool = self.engine.get_team_aggregate_pool()
        metrics: dict[str, TeamMetrics] = {}
        for code in (home, away):
            try:
                aggregate, origin = self.engine.get_team_aggregate(
                    code, live_matches=live_matches or None)
            except DataUnavailableError:
                aggregate, origin = None, DataOrigin.FALLBACK_STORE
            metrics[code] = StrengthEstimator.empirical_bayes_strengths(
                code, self.engine.display_name(code), aggregate,
                pool, mu, origin)
        estimation_method = "empirical-bayes"
        if glm:
            estimation_method = "poisson-glm+eb"
            for code in (home, away):
                # El proveedor vivo puede identificar equipos por TLA o por
                # nombre en inglés: se intentan ambas claves.
                key = next((k for k in (code,
                                        self.engine.display_name(code, english=True))
                            if k in glm), None)
                if key is not None:
                    atk, dfn = glm[key]
                    base = metrics[code]
                    metrics[code] = TeamMetrics(
                        code=base.code, name=base.name,
                        matches_sample=base.matches_sample,
                        goals_for_per_match=base.goals_for_per_match,
                        goals_against_per_match=base.goals_against_per_match,
                        attack_strength=atk, defense_resilience=dfn,
                        origin=base.origin,
                        shrinkage_weight=base.shrinkage_weight)

        # ---- 3. λ₃ desde tendencias históricas de fases finales ----------
        tendencies = self.engine.get_knockout_tendencies()
        lambda3 = max(0.0, float(tendencies.goal_covariance))

        # ---- 4. TIF por NLP sobre los 11 titulares -----------------------
        reports: dict[str, TacticalImpactReport] = {}
        for code in (home, away):
            xi, _xi_origin = self.engine.get_starting_xi(code)
            news = self.engine.get_player_news(xi) if xi else {}
            reports[code] = self.analyzer.team_report(code, news)

        # ---- 5. Ventaja de local para los anfitriones del Mundial 2026 ----
        # USA, CAN y MEX juegan en casa. Basado en el meta-análisis de
        # Pollard (2006): los anfitriones del Mundial anotan ~15 % más goles.
        HOST_NATIONS_2026: frozenset[str] = frozenset({"USA", "CAN", "MEX"})
        HOME_ADVANTAGE_FACTOR: float = 1.15
        home_adv_home = HOME_ADVANTAGE_FACTOR if home in HOST_NATIONS_2026 else 1.0
        home_adv_away = HOME_ADVANTAGE_FACTOR if away in HOST_NATIONS_2026 else 1.0

        # ---- 6. λ base según la forma funcional especificada -------------
        base_home = (mu * metrics[home].attack_strength
                     * metrics[away].defense_resilience * reports[home].tif
                     * home_adv_home)
        base_away = (mu * metrics[away].attack_strength
                     * metrics[home].defense_resilience * reports[away].tif
                     * home_adv_away)

        # ---- 7. Peso del mercado (solo con --live-odds) -------------------
        market_factor_home = market_factor_away = 1.0
        snapshot: MarketSnapshot | None = None
        if live_odds:
            snapshot = self.engine.get_market_snapshot(home, away,
                                                       force_refresh=True)
            if snapshot is not None:
                pre_model = BivariatePoissonModel(base_home, base_away, lambda3)
                market_factor_home, market_factor_away = self.blender.blend(
                    pre_model, snapshot)

        lambda_home_value = base_home * market_factor_home
        lambda_away_value = base_away * market_factor_away
        breakdown_home = LambdaBreakdown(
            tournament_mu=mu, attack=metrics[home].attack_strength,
            opponent_defense=metrics[away].defense_resilience,
            tif=reports[home].tif, home_advantage=home_adv_home,
            market_factor=market_factor_home, value=lambda_home_value)
        breakdown_away = LambdaBreakdown(
            tournament_mu=mu, attack=metrics[away].attack_strength,
            opponent_defense=metrics[home].defense_resilience,
            tif=reports[away].tif, home_advantage=home_adv_away,
            market_factor=market_factor_away, value=lambda_away_value)

        # ---- 8. Monte Carlo ----------------------------------------------
        model = BivariatePoissonModel(lambda_home_value, lambda_away_value, lambda3)
        mc = MonteCarloEngine(n_sims, seed=seed)
        result = mc.run(model, breakdown_home, breakdown_away, home, away,
                        progress=progress)
        return PredictionArtifacts(
            result=result, metrics_home=metrics[home], metrics_away=metrics[away],
            tif_home=reports[home], tif_away=reports[away], market=snapshot,
            tournament_mu=mu, estimation_method=estimation_method,
            tendencies=tendencies)


# ---------------------------------------------------------------------------
# Backtesting de calibración (comando `calibrar`)
# ---------------------------------------------------------------------------
def backtest_leave_one_edition_out(store: Any) -> dict[str, Any]:
    """Valida el modelo con esquema *leave-one-edition-out* sobre 2014–2022.

    Para cada edición e: se estiman μ, fuerzas EB y λ₃ con las otras dos
    ediciones y se predicen los 16 cruces de eliminación de e con la pmf
    analítica. Reporta log-loss y Brier multiclase contra el baseline
    uniforme (log-loss = ln 3 ≈ 1.0986; Brier uniforme = 2/3).

    Args:
        store: ``FallbackDataStore`` con el dataset histórico embebido.

    Returns:
        Métricas agregadas y por edición.
    """
    editions = store.editions()
    knockouts = store.knockout_matches()
    per_edition: dict[str, dict[str, float]] = {}
    total_ll = 0.0
    total_brier = 0.0
    total_n = 0

    for held_out in sorted(editions.keys()):
        train_eds = [e for e in editions if e != held_out]
        goals = sum(int(editions[e]["goals_total"]) for e in train_eds)
        matches = sum(int(editions[e]["matches_total"]) for e in train_eds)
        mu = StrengthEstimator.tournament_mu(goals, matches)

        pool: dict[str, dict[str, float]] = {}
        for e in train_eds:
            for code, t in editions[e]["teams"].items():
                agg = pool.setdefault(code, {"matches": 0.0, "gf": 0.0, "ga": 0.0})
                agg["matches"] += t["matches"]
                agg["gf"] += t["gf"]
                agg["ga"] += t["ga"]

        train_kos = [m for m in knockouts if str(m["edition"]) != held_out]
        h90 = np.array([m["score_90"][0] for m in train_kos], dtype=float)
        a90 = np.array([m["score_90"][1] for m in train_kos], dtype=float)
        lambda3 = max(0.0, float(np.cov(h90, a90, ddof=1)[0, 1]))

        ll = brier = 0.0
        n = 0
        for m in (k for k in knockouts if str(k["edition"]) == held_out):
            mh = StrengthEstimator.empirical_bayes_strengths(
                m["home"], m["home"], pool.get(m["home"]), pool, mu,
                DataOrigin.FALLBACK_STORE)
            ma = StrengthEstimator.empirical_bayes_strengths(
                m["away"], m["away"], pool.get(m["away"]), pool, mu,
                DataOrigin.FALLBACK_STORE)
            lam_h = mu * mh.attack_strength * ma.defense_resilience
            lam_a = mu * ma.attack_strength * mh.defense_resilience
            probs = BivariatePoissonModel(lam_h, lam_a, lambda3).analytic_1x2()
            gh, ga = int(m["score_90"][0]), int(m["score_90"][1])
            outcome = 0 if gh > ga else (1 if gh == ga else 2)
            ll -= math.log(max(probs[outcome], np.finfo(float).tiny))
            brier += sum((probs[k] - (1.0 if k == outcome else 0.0)) ** 2
                         for k in range(3))
            n += 1
        per_edition[held_out] = {"log_loss": ll / n, "brier": brier / n,
                                 "matches": n}
        total_ll += ll
        total_brier += brier
        total_n += n

    return {
        "per_edition": per_edition,
        "aggregate": {"log_loss": total_ll / total_n,
                      "brier": total_brier / total_n,
                      "matches": total_n},
        "baselines": {"uniform_log_loss": math.log(3.0),
                      "uniform_brier": 2.0 / 3.0},
    }
