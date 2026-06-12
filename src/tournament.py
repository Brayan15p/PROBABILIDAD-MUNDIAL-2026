"""Simulación estocástica del torneo completo (Mundial 2026, 104 partidos).

Modela el formato oficial de 48 selecciones: 12 grupos de 4 (72 partidos),
clasificación de los dos primeros + los 8 mejores terceros, y eliminación
directa desde dieciseisavos (partidos 73–104) según el cuadro publicado por
la FIFA, cargado como DATO desde ``data/fallback/groups_2026.json`` (nunca
cableado en código; el archivo es editable y se valida al cargar).

Detalles estadísticos:
- Cada partido se muestrea del mismo Poisson bivariado del motor
  (λ = μ·ataque·defensa_rival con fuerzas EB y λ₃ histórico).
- Empate en eliminación directa → prórroga modelada como proceso Poisson
  proporcional al tiempo (λ·30/90) → si persiste, tanda de penales con
  probabilidad ½ por lado (máxima entropía: no hay dato público de
  habilidad de tanda por selección; documentado como límite).
- Mejores terceros → ranking (pts, DG, GF, desempate aleatorio) y
  asignación a las llaves por *matching* bipartito con backtracking sobre
  los grupos permitidos por la FIFA en cada llave (mecanismo exacto del
  reglamento, que enumera las combinaciones factibles).
- Desempates de grupo: puntos → diferencia de gol → goles a favor →
  aleatorio (se omite el head-to-head del reglamento; con marcadores
  Poisson continuos el empate triple exacto es raro y el sesgo, despreciable).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src import config
from src.exceptions import DataUnavailableError, SimulationError
from src.models import DataOrigin
from src.simulator import BivariatePoissonModel, StrengthEstimator

#: Réplicas por defecto del torneo: con R=5000, el SE de una probabilidad
#: de campeón ~15% es sqrt(0.15·0.85/5000) ≈ 0.5 puntos porcentuales,
#: suficiente para ordenar favoritos sin ambigüedad.
DEFAULT_TOURNAMENT_REPLICAS: int = 5_000

#: Piso operativo de réplicas (por debajo el ranking de colas es ruido).
MIN_TOURNAMENT_REPLICAS: int = 100

_SLOT_RE = re.compile(r"^([12])([A-L])$|^3:([A-L]+)$|^([WL])(\d+)$")

#: Etapas reportadas, de la más profunda a la más temprana.
STAGES: tuple[str, ...] = ("champion", "final", "semi_final", "quarter_final",
                           "round_of_16", "round_of_32")


@dataclass(frozen=True)
class KnockoutSlot:
    """Referencia simbólica de un cupo del cuadro (parseada y validada).

    Attributes:
        kind: ``group_rank`` (1X/2X), ``third`` (3:ABC..) o ``match_ref`` (W/L n).
        group: Letra de grupo si aplica.
        rank: 1 o 2 si aplica.
        allowed_groups: Grupos admitidos para un cupo de tercero.
        match: Número de partido referenciado por W/L.
        take_winner: True para W, False para L.
    """

    kind: str
    group: str | None = None
    rank: int | None = None
    allowed_groups: tuple[str, ...] = ()
    match: int | None = None
    take_winner: bool = True

    @classmethod
    def parse(cls, raw: str) -> "KnockoutSlot":
        """Parsea la sintaxis de cupos del JSON de estructura.

        Args:
            raw: ``"1A"``, ``"2K"``, ``"3:ABCDF"``, ``"W73"`` o ``"L101"``.

        Returns:
            Slot tipado.

        Raises:
            SimulationError: Sintaxis no reconocida.
        """
        m = _SLOT_RE.match(raw.strip())
        if not m:
            raise SimulationError(f"Cupo de cuadro inválido: '{raw}'")
        if m.group(1):
            return cls(kind="group_rank", rank=int(m.group(1)), group=m.group(2))
        if m.group(3):
            return cls(kind="third", allowed_groups=tuple(m.group(3)))
        return cls(kind="match_ref", match=int(m.group(5)),
                   take_winner=m.group(4) == "W")


@dataclass(frozen=True)
class TournamentStructure:
    """Estructura completa del torneo cargada y validada desde JSON.

    Attributes:
        groups: Mapeo letra → 4 códigos FIFA.
        knockout_rounds: Rondas en orden cronológico; cada partido es
            ``(numero, slot_home, slot_away)``.
        source_meta: Metadatos de procedencia del archivo.
    """

    groups: dict[str, tuple[str, ...]]
    knockout_rounds: tuple[tuple[str, tuple[tuple[int, KnockoutSlot, KnockoutSlot], ...]], ...]
    source_meta: dict[str, Any]

    @property
    def teams(self) -> tuple[str, ...]:
        """Los 48 códigos FIFA del torneo."""
        return tuple(t for g in self.groups.values() for t in g)

    @classmethod
    def load(cls, path: Path) -> "TournamentStructure":
        """Carga y valida la estructura del torneo.

        Validaciones: 12 grupos × 4 equipos únicos; cada 1X/2X usado
        exactamente una vez en dieciseisavos; exactamente 8 cupos de
        terceros; toda referencia W/L apunta a un partido de una ronda
        anterior.

        Args:
            path: Ruta del JSON de estructura.

        Returns:
            Estructura inmutable validada.

        Raises:
            DataUnavailableError: Archivo ausente.
            SimulationError: Estructura inconsistente.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DataUnavailableError(f"Estructura de torneo ausente: {path}") from exc

        groups = {k: tuple(v) for k, v in raw["groups"].items()}
        if len(groups) != 12 or any(len(v) != 4 for v in groups.values()):
            raise SimulationError("Se requieren exactamente 12 grupos de 4 equipos")
        teams = [t for g in groups.values() for t in g]
        if len(set(teams)) != 48:
            raise SimulationError("Los 48 equipos del torneo deben ser únicos")

        def parse_round(items: list[dict[str, Any]]) -> tuple[tuple[int, KnockoutSlot, KnockoutSlot], ...]:
            return tuple((int(m["match"]), KnockoutSlot.parse(m["home"]),
                          KnockoutSlot.parse(m["away"])) for m in items)

        rounds: list[tuple[str, tuple[tuple[int, KnockoutSlot, KnockoutSlot], ...]]] = [
            ("round_of_32", parse_round(raw["round_of_32"])),
            ("round_of_16", parse_round(raw["round_of_16"])),
            ("quarter_final", parse_round(raw["quarter_finals"])),
            ("semi_final", parse_round(raw["semi_finals"])),
            ("third_place", parse_round([raw["third_place"]])),
            ("final", parse_round([raw["final"]])),
        ]

        r32 = rounds[0][1]
        used_ranks = [f"{s.rank}{s.group}" for _, h, a in r32
                      for s in (h, a) if s.kind == "group_rank"]
        if sorted(used_ranks) != sorted(f"{r}{g}" for g in groups for r in (1, 2)):
            raise SimulationError(
                "Dieciseisavos: cada ganador y segundo de grupo debe usarse "
                "exactamente una vez")
        thirds = [s for _, h, a in r32 for s in (h, a) if s.kind == "third"]
        if len(thirds) != 8:
            raise SimulationError("Dieciseisavos: deben existir 8 cupos de terceros")
        for slot in thirds:
            unknown = set(slot.allowed_groups) - set(groups)
            if unknown:
                raise SimulationError(f"Cupo de tercero con grupos inexistentes: {unknown}")

        known_matches = {n for n, _, _ in r32}
        for _name, matches in rounds[1:]:
            for n, home, away in matches:
                for s in (home, away):
                    if s.kind != "match_ref" or s.match not in known_matches:
                        raise SimulationError(
                            f"Partido {n}: referencia inválida a '{s}'")
            known_matches |= {n for n, _, _ in matches}

        return cls(groups=groups, knockout_rounds=tuple(rounds),
                   source_meta=dict(raw.get("_meta", {})))


@dataclass(frozen=True)
class TournamentResult:
    """Probabilidades por selección de alcanzar cada etapa del torneo.

    Attributes:
        n_replicas: Réplicas Monte Carlo del torneo completo.
        seed: Semilla reproducible.
        tournament_mu: μ usado en todos los λ.
        lambda3: Covarianza de goles aplicada.
        stage_probabilities: ``code → {etapa: probabilidad}``.
        third_match_wins: ``code → P(ganar el partido por el 3er puesto)``.
        max_standard_error: Cota del SE binomial de cualquier probabilidad.
        matching_fallbacks: Réplicas donde el matching de terceros requirió
            relajar restricciones (0 esperado; >0 indica cuadro mal editado).
        source_meta: Procedencia de la estructura usada.
    """

    n_replicas: int
    seed: int
    tournament_mu: float
    lambda3: float
    stage_probabilities: dict[str, dict[str, float]]
    third_match_wins: dict[str, float]
    max_standard_error: float
    matching_fallbacks: int
    source_meta: dict[str, Any] = field(default_factory=dict)

    def ranking(self) -> list[tuple[str, dict[str, float]]]:
        """Equipos ordenados por P(campeón) descendente."""
        return sorted(self.stage_probabilities.items(),
                      key=lambda kv: kv[1]["champion"], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialización JSON-compatible del resultado completo."""
        return {
            "n_replicas": self.n_replicas,
            "seed": self.seed,
            "tournament_mu": self.tournament_mu,
            "lambda3": self.lambda3,
            "max_standard_error": round(self.max_standard_error, 6),
            "matching_fallbacks": self.matching_fallbacks,
            "stage_probabilities": {
                code: {k: round(v, 6) for k, v in stages.items()}
                for code, stages in self.stage_probabilities.items()
            },
            "third_place_match_win": {k: round(v, 6)
                                      for k, v in self.third_match_wins.items()},
            "structure_meta": self.source_meta,
        }


class TournamentSimulator:
    """Motor Monte Carlo del torneo completo sobre el modelo bivariado."""

    def __init__(self, engine: Any, structure: TournamentStructure,
                 n_replicas: int = DEFAULT_TOURNAMENT_REPLICAS,
                 seed: int | None = None) -> None:
        if n_replicas < MIN_TOURNAMENT_REPLICAS:
            raise SimulationError(
                f"Se exigen ≥{MIN_TOURNAMENT_REPLICAS} réplicas de torneo "
                f"(recibidas: {n_replicas})")
        self.engine = engine
        self.structure = structure
        self.n_replicas = int(n_replicas)
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1, np.uint64)[0] >> 1)
        self.seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))

        # ---- parámetros del modelo (mismos estimadores del pipeline) ----
        totals = engine.store.tournament_totals()
        self.mu: float = StrengthEstimator.tournament_mu(
            totals["goals"], totals["matches"])
        self.lambda3: float = max(
            0.0, float(engine.get_knockout_tendencies().goal_covariance))
        pool = engine.store.all_team_aggregates()
        self._metrics: dict[str, tuple[float, float]] = {}
        for code in structure.teams:
            agg = engine.store.team_aggregate(code)
            m = StrengthEstimator.empirical_bayes_strengths(
                code, code, agg, pool, self.mu, DataOrigin.FALLBACK_STORE)
            self._metrics[code] = (m.attack_strength, m.defense_resilience)

    # ------------------------------------------------------------------
    def _lambdas(self, team_a: str, team_b: str) -> tuple[float, float]:
        """λ marginales del cruce a vs b (neutral: sin TIF ni mercado)."""
        atk_a, def_a = self._metrics[team_a]
        atk_b, def_b = self._metrics[team_b]
        return self.mu * atk_a * def_b, self.mu * atk_b * def_a

    def _sample_match(self, lam_h: float, lam_a: float,
                      n: int) -> tuple[np.ndarray, np.ndarray]:
        """n marcadores bivariados del cruce (vectorizado)."""
        model = BivariatePoissonModel(lam_h, lam_a, self.lambda3)
        return model.sample(n, self._rng)

    def _knockout_winner(self, team_h: str, team_a: str) -> str:
        """Resuelve un cruce de eliminación: 90' → prórroga → penales."""
        lam_h, lam_a = self._lambdas(team_h, team_a)
        # 90 minutos (bivariado escalar)
        model = BivariatePoissonModel(lam_h, lam_a, self.lambda3)
        h, a = model.sample(1, self._rng)
        gh, ga = int(h[0]), int(a[0])
        if gh != ga:
            return team_h if gh > ga else team_a
        # Prórroga: proceso Poisson proporcional al tiempo (30/90 = 1/3).
        et_model = BivariatePoissonModel(lam_h / 3.0, lam_a / 3.0, self.lambda3 / 3.0)
        h, a = et_model.sample(1, self._rng)
        eh, ea = int(h[0]), int(a[0])
        if eh != ea:
            return team_h if eh > ea else team_a
        # Penales: máxima entropía (sin dato de habilidad de tanda).
        return team_h if self._rng.random() < 0.5 else team_a

    # ------------------------------------------------------------------
    @staticmethod
    def _assign_thirds(qualified: list[str],
                       slots: list[tuple[int, tuple[str, ...]]]
                       ) -> dict[int, str] | None:
        """Matching bipartito (backtracking + MRV) terceros → llaves.

        Args:
            qualified: Letras de grupo de los 8 terceros clasificados.
            slots: ``(numero_de_partido, grupos_permitidos)`` por llave.

        Returns:
            Asignación partido → letra de grupo, o ``None`` si no existe
            (solo posible con un cuadro editado inconsistente).
        """
        remaining = list(qualified)
        assignment: dict[int, str] = {}

        def backtrack(pending: list[tuple[int, tuple[str, ...]]]) -> bool:
            if not pending:
                return True
            # MRV: la llave con menos candidatos disponibles primero.
            pending = sorted(
                pending,
                key=lambda s: sum(1 for g in remaining if g in s[1]))
            match_no, allowed = pending[0]
            rest = pending[1:]
            for g in [g for g in allowed if g in remaining]:
                remaining.remove(g)
                assignment[match_no] = g
                if backtrack(rest):
                    return True
                remaining.append(g)
                del assignment[match_no]
            return False

        return assignment if backtrack(slots) else None

    # ------------------------------------------------------------------
    def run(self, progress: Callable[[int], None] | None = None) -> TournamentResult:
        """Simula el torneo completo ``n_replicas`` veces.

        Args:
            progress: Callback con el incremento de réplicas completadas.

        Returns:
            :class:`TournamentResult` con probabilidades por etapa.
        """
        R = self.n_replicas
        groups = self.structure.groups
        team_idx = {code: i for i, code in enumerate(self.structure.teams)}

        # ---- fase de grupos, vectorizada por fixture ---------------------
        pts = np.zeros((R, 48))
        gd = np.zeros((R, 48))
        gf = np.zeros((R, 48))
        for letter, members in groups.items():
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = members[i], members[j]
                    lam_h, lam_a = self._lambdas(a, b)
                    gh, ga = self._sample_match(lam_h, lam_a, R)
                    ia, ib = team_idx[a], team_idx[b]
                    pts[:, ia] += 3 * (gh > ga) + (gh == ga)
                    pts[:, ib] += 3 * (ga > gh) + (gh == ga)
                    gd[:, ia] += gh - ga
                    gd[:, ib] += ga - gh
                    gf[:, ia] += gh
                    gf[:, ib] += ga

        # Clave de orden escalar estricta: pts ≫ DG ≫ GF ≫ uniforme(0,10).
        # Los offsets garantizan dominancia lexicográfica (DG ∈ [-100,100],
        # GF < 100 en 6 partidos): ver derivación en el docstring del módulo.
        tiebreak = self._rng.uniform(0.0, 10.0, size=(R, 48))
        order_key = pts * 1e8 + (gd + 100.0) * 1e4 + gf * 10.0 + tiebreak

        group_rank: dict[str, np.ndarray] = {}
        for letter, members in groups.items():
            cols = np.array([team_idx[t] for t in members])
            ranks = np.argsort(order_key[:, cols], axis=1)[:, ::-1]
            group_rank[letter] = cols[ranks]  # (R,4): índices ordenados 1º→4º

        # ---- acumuladores -------------------------------------------------
        counters: dict[str, dict[str, int]] = {
            code: {s: 0 for s in STAGES} for code in self.structure.teams}
        third_wins: dict[str, int] = {code: 0 for code in self.structure.teams}
        idx_team = {i: c for c, i in team_idx.items()}
        matching_fallbacks = 0
        letters = sorted(groups)

        r32_def = self.structure.knockout_rounds[0][1]
        third_slots = [(n, s.allowed_groups) for n, h, a in r32_def
                       for s in (h, a) if s.kind == "third"]

        chunk = max(1, R // 50)
        for r in range(R):
            firsts = {L: idx_team[int(group_rank[L][r, 0])] for L in letters}
            seconds = {L: idx_team[int(group_rank[L][r, 1])] for L in letters}
            thirds = {L: idx_team[int(group_rank[L][r, 2])] for L in letters}

            # Mejores 8 terceros por la misma clave de orden.
            third_scores = sorted(
                letters,
                key=lambda L: order_key[r, team_idx[thirds[L]]], reverse=True)
            qualified_thirds = third_scores[:8]
            assign = self._assign_thirds(list(qualified_thirds), third_slots)
            if assign is None:
                # Cuadro editado sin matching factible: asignación libre.
                matching_fallbacks += 1
                assign = {n: g for (n, _), g in zip(third_slots, qualified_thirds)}

            winners: dict[int, str] = {}
            losers: dict[int, str] = {}

            def resolve(slot: KnockoutSlot, match_no: int) -> str:
                if slot.kind == "group_rank":
                    return (firsts if slot.rank == 1 else seconds)[slot.group]  # type: ignore[index]
                if slot.kind == "third":
                    return thirds[assign[match_no]]
                return (winners if slot.take_winner else losers)[slot.match]  # type: ignore[index]

            for stage_name, matches in self.structure.knockout_rounds:
                for n, home_slot, away_slot in matches:
                    th = resolve(home_slot, n)
                    ta = resolve(away_slot, n)
                    if stage_name != "third_place":
                        stage_key = ("round_of_32" if stage_name == "round_of_32"
                                     else stage_name)
                        for t in (th, ta):
                            if stage_key in counters[t]:
                                counters[t][stage_key] += 1
                    w = self._knockout_winner(th, ta)
                    winners[n] = w
                    losers[n] = th if w == ta else ta
                    if stage_name == "third_place":
                        third_wins[w] += 1
                    if stage_name == "final":
                        counters[w]["champion"] += 1

            if progress is not None and (r + 1) % chunk == 0:
                progress(chunk)

        probs = {
            code: {stage: counters[code][stage] / R for stage in STAGES}
            for code in self.structure.teams
        }
        max_se = max(
            math.sqrt(max(p * (1 - p), 0.0) / R)
            for stages in probs.values() for p in stages.values())
        return TournamentResult(
            n_replicas=R, seed=self.seed, tournament_mu=self.mu,
            lambda3=self.lambda3, stage_probabilities=probs,
            third_match_wins={c: w / R for c, w in third_wins.items()},
            max_standard_error=max_se,
            matching_fallbacks=matching_fallbacks,
            source_meta=self.structure.source_meta)


def default_structure_path() -> Path:
    """Ruta del cuadro oficial embebido del Mundial 2026."""
    return config.FALLBACK_DIR / "groups_2026.json"
