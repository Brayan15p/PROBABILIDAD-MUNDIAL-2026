"""Interfaz de línea de comandos del motor predictivo Mundial 2026.

Comandos:
    predecir   Predicción completa de un partido (Monte Carlo bivariado).
    equipos    Lista las selecciones reconocidas y sus alias.
    calibrar   Backtest leave-one-edition-out sobre 2014–2022.
    cache      Observabilidad y limpieza del caché local estricto.
    db         Estado y sincronización de la base de datos Supabase.

Ejemplo:
    $ wc26 predecir --local "México" --visitante "Argentina" \
          --sims 20000 --live-odds
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src import __version__, config
from src.api_client import NewsAggregator, PolymarketProvider, WorldCupDataEngine
from src.cache import CacheManager
from src.db import SupabaseRepository
from src.exceptions import WorldCupEngineError
from src.models import MarketSnapshot, SimulationResult, TacticalImpactReport
from src.nlp_processor import TacticalSentimentAnalyzer, create_backend
from src.simulator import (
    MarketBlender,
    PredictionArtifacts,
    PredictionPipeline,
    backtest_leave_one_edition_out,
)

console = Console()
err_console = Console(stderr=True)


def _setup_logging(verbose: bool) -> None:
    """Configura logging enriquecido hacia stderr."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_path=False,
                              rich_tracebacks=True)],
        force=True,
    )


def _validate_sims(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    """Valida el mínimo contractual de iteraciones Monte Carlo."""
    if value < config.MIN_MONTE_CARLO_SIMS:
        raise click.BadParameter(
            f"el mínimo contractual es {config.MIN_MONTE_CARLO_SIMS:,} "
            f"iteraciones (recibido: {value:,})")
    return value


class _OfflineMarket(PolymarketProvider):
    """Stub de mercado para ``--offline``: nunca toca la red."""

    def search_match_market(self, home_name: str, away_name: str,
                            **_kwargs: Any) -> dict[str, Any] | None:
        """Devuelve None: fuerza el fallback al consenso embebido."""
        return None


def _build_engine(offline: bool) -> WorldCupDataEngine:
    """Construye la fachada de datos, recortada si se pidió modo offline."""
    engine = WorldCupDataEngine()
    if offline:
        engine.sports_provider = None
        engine.news = NewsAggregator(providers=[])
        engine.market = _OfflineMarket()
    return engine


# ---------------------------------------------------------------------------
# Grupo raíz
# ---------------------------------------------------------------------------
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="wc26")
def cli() -> None:
    """Motor predictivo estocástico para la Copa del Mundo FIFA 2026."""


# ---------------------------------------------------------------------------
# predecir
# ---------------------------------------------------------------------------
@cli.command("predecir")
@click.option("--local", required=True, help="Selección local (es/en, ej. 'México').")
@click.option("--visitante", required=True, help="Selección visitante.")
@click.option("--sims", default=config.MIN_MONTE_CARLO_SIMS, show_default=True,
              callback=_validate_sims, help="Iteraciones Monte Carlo (≥15.000).")
@click.option("--live-odds", is_flag=True, default=False,
              help="Consulta mercados de predicción (Polymarket) y los funde "
                   "con el modelo (log-opinion pool).")
@click.option("--seed", type=int, default=None,
              help="Semilla RNG para reproducibilidad exacta.")
@click.option("--offline", is_flag=True, default=False,
              help="No toca la red: opera 100% sobre caché + fallback embebido.")
@click.option("--nlp-backend", type=click.Choice(["vader", "transformers"]),
              default="vader", show_default=True,
              help="Backend del análisis de sentimiento.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emite el resultado completo como JSON (apto para pipes).")
@click.option("--no-db", is_flag=True, default=False,
              help="No persiste la corrida en Supabase aunque haya credenciales.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Bitácora detallada (procedencia de cada dato).")
def predecir(local: str, visitante: str, sims: int, live_odds: bool,
             seed: int | None, offline: bool, nlp_backend: str,
             as_json: bool, no_db: bool, verbose: bool) -> None:
    """Predice un partido: marcador modal, 1X2 y mercados auxiliares."""
    _setup_logging(verbose)
    engine = _build_engine(offline)
    analyzer = TacticalSentimentAnalyzer(backend=create_backend(nlp_backend))
    pipeline = PredictionPipeline(engine, analyzer, blender=MarketBlender())

    try:
        if as_json:
            artifacts = pipeline.predict(local, visitante, sims,
                                         live_odds=live_odds, seed=seed)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Simulando {sims:,} réplicas bivariadas…", total=sims)
                artifacts = pipeline.predict(
                    local, visitante, sims, live_odds=live_odds, seed=seed,
                    progress=lambda inc: progress.advance(task, inc))
    except WorldCupEngineError as exc:
        raise click.ClickException(str(exc)) from exc

    persisted = False
    if not no_db:
        repo = SupabaseRepository()
        if repo.enabled:
            persisted = repo.save_prediction(
                artifacts.result, artifacts.tif_home.tif, artifacts.tif_away.tif)
            repo.save_team_metrics(artifacts.metrics_home)
            repo.save_team_metrics(artifacts.metrics_away)
            repo.save_tif_report(artifacts.tif_home)
            repo.save_tif_report(artifacts.tif_away)
            if artifacts.market is not None:
                repo.save_market_snapshot(artifacts.result.home_code,
                                          artifacts.result.away_code,
                                          artifacts.market)

    if as_json:
        payload = artifacts.result.to_dict()
        payload["context"] = {
            "tournament_mu": artifacts.tournament_mu,
            "estimation_method": artifacts.estimation_method,
            "tif": {
                artifacts.result.home_code: artifacts.tif_home.tif,
                artifacts.result.away_code: artifacts.tif_away.tif,
            },
            "market_source": artifacts.market.source if artifacts.market else None,
            "persisted_to_supabase": persisted,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    _render_prediction(engine, artifacts, persisted, verbose)


def _prob_bar(p: float, width: int = 24) -> str:
    """Barra unicode proporcional a una probabilidad."""
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def _render_prediction(engine: WorldCupDataEngine,
                       artifacts: PredictionArtifacts,
                       persisted: bool, verbose: bool) -> None:
    """Renderiza el reporte completo de la predicción en la terminal."""
    res: SimulationResult = artifacts.result
    home_name = engine.display_name(res.home_code)
    away_name = engine.display_name(res.away_code)

    console.print(Panel.fit(
        f"[bold cyan]{home_name}[/bold cyan] vs "
        f"[bold magenta]{away_name}[/bold magenta]\n"
        f"Copa del Mundo 2026 · Poisson bivariado + Monte Carlo "
        f"({res.n_simulations:,} réplicas · semilla {res.seed})",
        title="⚽ Motor Predictivo", border_style="green"))

    # --- marcador modal -------------------------------------------------
    ci_low, ci_high = res.modal_confidence_interval
    console.print(Panel.fit(
        f"[bold yellow]{res.modal_score.home_goals} – "
        f"{res.modal_score.away_goals}[/bold yellow]\n"
        f"P(moda) = {res.modal_confidence:.2%}  "
        f"IC95% [{ci_low:.2%}, {ci_high:.2%}]\n"
        f"pmf analítica = {res.modal_score.analytic_probability:.2%} · "
        f"nitidez vs 2.ª moda = ×{res.mode_separation_ratio:.2f}",
        title="Marcador exacto más probable (moda conjunta)",
        border_style="yellow"))

    # --- macro probabilidades --------------------------------------------
    macro = Table(title="Probabilidades macro (1X2)", show_lines=False)
    macro.add_column("Resultado", style="bold")
    macro.add_column("Probabilidad", justify="right")
    macro.add_column("", no_wrap=True)
    macro.add_row(f"Victoria {home_name}", f"{res.home_win_prob:.2%}",
                  f"[cyan]{_prob_bar(res.home_win_prob)}[/cyan]")
    macro.add_row("Empate", f"{res.draw_prob:.2%}",
                  f"[white]{_prob_bar(res.draw_prob)}[/white]")
    macro.add_row(f"Victoria {away_name}", f"{res.away_win_prob:.2%}",
                  f"[magenta]{_prob_bar(res.away_win_prob)}[/magenta]")
    macro.caption = (f"SE Monte Carlo ≤ {res.macro_standard_error:.4f} · "
                     f"Over 2.5: {res.over_2_5_prob:.2%} · "
                     f"Ambos marcan: {res.btts_prob:.2%} · "
                     f"E[goles] = {res.expected_total_goals:.2f}")
    console.print(macro)

    # --- desglose de lambdas ----------------------------------------------
    lam = Table(title="Construcción de λ (goles esperados)")
    lam.add_column("Término", style="bold")
    lam.add_column(home_name, justify="right", style="cyan")
    lam.add_column(away_name, justify="right", style="magenta")
    bh, ba = res.lambda_home, res.lambda_away
    lam.add_row("μ torneo (MLE 2014–2022)", f"{bh.tournament_mu:.4f}",
                f"{ba.tournament_mu:.4f}")
    lam.add_row("Fuerza de ataque", f"{bh.attack:.4f}", f"{ba.attack:.4f}")
    lam.add_row("Defensa rival (vulnerab.)", f"{bh.opponent_defense:.4f}",
                f"{ba.opponent_defense:.4f}")
    lam.add_row("TIF (NLP prensa titular)", f"{bh.tif:.4f}", f"{ba.tif:.4f}")
    lam.add_row("Factor mercado", f"{bh.market_factor:.4f}",
                f"{ba.market_factor:.4f}")
    lam.add_row("[bold]λ final[/bold]", f"[bold]{bh.value:.4f}[/bold]",
                f"[bold]{ba.value:.4f}[/bold]")
    lam.caption = (f"λ₃ (covarianza KO 2014–2022) = {res.lambda3_covariance:.4f} · "
                   f"método fuerzas: {artifacts.estimation_method}")
    console.print(lam)

    # --- top marcadores ----------------------------------------------------
    tops = Table(title="Top 5 marcadores exactos")
    tops.add_column("Marcador", style="bold")
    tops.add_column("MC", justify="right")
    tops.add_column("Analítica", justify="right")
    tops.add_column("±SE", justify="right")
    for s in res.top_scorelines[:5]:
        tops.add_row(f"{s.home_goals}–{s.away_goals}", f"{s.probability:.2%}",
                     f"{s.analytic_probability:.2%}",
                     f"{s.mc_standard_error:.4f}")
    console.print(tops)

    # --- TIF ----------------------------------------------------------------
    tif = Table(title="Tactical Impact Factor (NLP sobre el XI titular)")
    tif.add_column("Equipo", style="bold")
    tif.add_column("TIF", justify="right")
    tif.add_column("Polaridad", justify="right")
    tif.add_column("Artículos", justify="right")
    tif.add_column("Motor")
    for rep in (artifacts.tif_home, artifacts.tif_away):
        tif.add_row(engine.display_name(rep.team_code), f"{rep.tif:.4f}",
                    f"{rep.team_polarity:+.4f}", str(rep.articles_total),
                    rep.engine)
    console.print(tif)
    if verbose:
        _render_tif_details(artifacts.tif_home)
        _render_tif_details(artifacts.tif_away)

    # --- mercado / procedencia ----------------------------------------------
    if artifacts.market is not None:
        m: MarketSnapshot = artifacts.market
        draw_txt = f"{m.draw_prob:.2%}" if m.draw_prob is not None else "n/d (outright)"
        console.print(Panel.fit(
            f"Local {m.home_prob:.2%} · Empate {draw_txt} · "
            f"Visitante {m.away_prob:.2%}\n"
            f"Fuente: {m.source} · origen: {m.origin.value} · "
            f"corte: {m.as_of:%Y-%m-%d %H:%M} UTC",
            title="Consenso de mercado integrado", border_style="blue"))

    cache_stats = CacheManager().stats()
    footer = (f"Caché: {cache_stats['hits']} hits / {cache_stats['misses']} misses · "
              f"Supabase: {'✓ persistido' if persisted else '— no persistido'}")
    if verbose:
        prov = Table(title="Procedencia de los datos")
        prov.add_column("Recurso")
        prov.add_column("Origen")
        for resource, origin in engine.provenance:
            prov.add_row(resource, origin)
        console.print(prov)
        tend = artifacts.tendencies
        console.print(Panel.fit(
            f"Cruces analizados: {tend.matches} ({', '.join(map(str, tend.editions))})\n"
            f"Goles/partido 90': {tend.goals_per_match_90:.3f} · "
            f"Empates 90': {tend.draw_rate_90:.2%}\n"
            f"Covarianza goles: {tend.goal_covariance:+.4f} → λ₃\n"
            f"Medias por fase: "
            + " · ".join(f"{k}: {v:.2f}" for k, v in tend.stage_goal_means.items()),
            title="Tendencias de fases finales (adaptador Zafronix)",
            border_style="dim"))
    console.print(f"[dim]{footer}[/dim]")


def _render_tif_details(report: TacticalImpactReport) -> None:
    """Tabla detallada del TIF por jugador (modo verbose)."""
    if not report.players:
        return
    t = Table(title=f"Detalle NLP · {report.team_code}")
    t.add_column("Jugador")
    t.add_column("Artículos", justify="right")
    t.add_column("Polaridad cruda", justify="right")
    t.add_column("Polaridad contraída", justify="right")
    t.add_column("Tópicos críticos")
    for p in report.players:
        t.add_row(p.player, str(p.articles_analyzed), f"{p.raw_polarity:+.3f}",
                  f"{p.shrunk_polarity:+.3f}",
                  ", ".join(f"{k}×{v}" for k, v in p.topics.items()) or "—")
    console.print(t)


# ---------------------------------------------------------------------------
# equipos
# ---------------------------------------------------------------------------
@cli.command("equipos")
@click.option("--confederacion", default=None,
              help="Filtra por confederación (UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC).")
def equipos(confederacion: str | None) -> None:
    """Lista las selecciones reconocidas por el motor y sus alias."""
    _setup_logging(False)
    engine = WorldCupDataEngine()
    table = Table(title="Selecciones reconocidas")
    table.add_column("Código", style="bold")
    table.add_column("Nombre")
    table.add_column("Alias")
    table.add_column("Confederación")
    for code, meta in sorted(engine.store.registry().items()):
        conf = meta.get("confederation", "?")
        if confederacion and conf.upper() != confederacion.upper():
            continue
        names = meta["names"]
        table.add_row(code, names[0], ", ".join(names[1:]) or "—", conf)
    console.print(table)


# ---------------------------------------------------------------------------
# calibrar
# ---------------------------------------------------------------------------
@cli.command("calibrar")
def calibrar() -> None:
    """Backtest leave-one-edition-out del modelo sobre 2014–2022."""
    _setup_logging(False)
    engine = WorldCupDataEngine()
    with console.status("Ejecutando backtest leave-one-edition-out…"):
        report = backtest_leave_one_edition_out(engine.store)
    table = Table(title="Calibración del modelo (fases finales 2014–2022)")
    table.add_column("Edición excluida", style="bold")
    table.add_column("Partidos", justify="right")
    table.add_column("Log-loss", justify="right")
    table.add_column("Brier", justify="right")
    for edition, m in report["per_edition"].items():
        table.add_row(edition, str(m["matches"]), f"{m['log_loss']:.4f}",
                      f"{m['brier']:.4f}")
    agg = report["aggregate"]
    base = report["baselines"]
    table.add_row("[bold]Agregado[/bold]", str(agg["matches"]),
                  f"[bold]{agg['log_loss']:.4f}[/bold]",
                  f"[bold]{agg['brier']:.4f}[/bold]")
    table.caption = (f"Baseline uniforme: log-loss {base['uniform_log_loss']:.4f} · "
                     f"Brier {base['uniform_brier']:.4f} — el modelo debe quedar "
                     "por debajo para aportar señal real.")
    console.print(table)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
@cli.group("cache")
def cache_group() -> None:
    """Observabilidad y mantenimiento del caché local estricto."""


@cache_group.command("stats")
def cache_stats() -> None:
    """Muestra hits/misses y entradas por namespace."""
    stats = CacheManager().stats()
    table = Table(title=f"Caché local · {stats['base_dir']}")
    table.add_column("Namespace")
    table.add_column("Entradas", justify="right")
    for ns, count in stats["entries_by_namespace"].items():
        table.add_row(ns, str(count))
    table.caption = f"Proceso actual: {stats['hits']} hits / {stats['misses']} misses"
    console.print(table)


@cache_group.command("clear")
@click.option("--namespace", default=None, help="Limita la limpieza a un namespace.")
@click.confirmation_option(prompt="¿Eliminar las entradas de caché?")
def cache_clear(namespace: str | None) -> None:
    """Elimina entradas de caché (todas o por namespace)."""
    deleted = CacheManager().clear(namespace)
    console.print(f"[green]✓[/green] {deleted} entradas eliminadas.")


# ---------------------------------------------------------------------------
# db (Supabase)
# ---------------------------------------------------------------------------
@cli.group("db")
def db_group() -> None:
    """Estado y sincronización de la base de datos Supabase."""


@db_group.command("status")
def db_status() -> None:
    """Verifica conectividad con Supabase y el esquema aplicado."""
    _setup_logging(False)
    repo = SupabaseRepository()
    try:
        health = repo.health_check()
    except WorldCupEngineError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(Panel.fit(
        f"URL: {health['url']}\nPredicciones almacenadas: "
        f"{health['predictions_stored']}",
        title="Supabase ✓", border_style="green"))


@db_group.command("sync-teams")
def db_sync_teams() -> None:
    """Sincroniza el registro maestro de selecciones hacia Supabase."""
    _setup_logging(True)
    engine = WorldCupDataEngine()
    repo = SupabaseRepository()
    if not repo.enabled:
        raise click.ClickException(
            "Supabase no configurado (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")
    ok = repo.sync_teams(engine.store.registry())
    if not ok:
        raise click.ClickException("El upsert de equipos fue rechazado por PostgREST")
    console.print("[green]✓[/green] Registro de selecciones sincronizado.")


@db_group.command("recientes")
@click.option("--limit", default=10, show_default=True, help="Filas a mostrar.")
def db_recientes(limit: int) -> None:
    """Lista las últimas predicciones persistidas en Supabase."""
    repo = SupabaseRepository()
    try:
        rows = repo.recent_predictions(limit=limit)
    except WorldCupEngineError as exc:
        raise click.ClickException(str(exc)) from exc
    table = Table(title="Últimas predicciones (Supabase)")
    table.add_column("Fecha")
    table.add_column("Partido")
    table.add_column("1X2", justify="right")
    table.add_column("Moda", justify="right")
    for r in rows:
        table.add_row(
            str(r.get("created_at", ""))[:16],
            f"{r.get('home_code')} vs {r.get('away_code')}",
            f"{r.get('p_home_win', 0):.0%}/{r.get('p_draw', 0):.0%}/"
            f"{r.get('p_away_win', 0):.0%}",
            f"{r.get('modal_score')} ({r.get('modal_probability', 0):.1%})")
    console.print(table)


# ---------------------------------------------------------------------------
# torneo
# ---------------------------------------------------------------------------
def _validate_replicas(_ctx: click.Context, _param: click.Parameter,
                       value: int) -> int:
    """Valida el piso de réplicas del torneo."""
    from src.tournament import MIN_TOURNAMENT_REPLICAS

    if value < MIN_TOURNAMENT_REPLICAS:
        raise click.BadParameter(
            f"mínimo {MIN_TOURNAMENT_REPLICAS} réplicas de torneo "
            f"(recibidas: {value})")
    return value


@cli.command("torneo")
@click.option("--grupos", "grupos_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="JSON con grupos y cuadro (default: sorteo oficial embebido).")
@click.option("--replicas", default=None, type=int, callback=None,
              help="Réplicas Monte Carlo del torneo completo [default: 5000].")
@click.option("--seed", type=int, default=None, help="Semilla RNG reproducible.")
@click.option("--offline", is_flag=True, default=False,
              help="No toca la red: opera sobre el fallback embebido.")
@click.option("--top", default=15, show_default=True,
              help="Número de selecciones a mostrar en el ranking.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emite el resultado completo como JSON.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def torneo(grupos_path: str | None, replicas: int | None, seed: int | None,
           offline: bool, top: int, as_json: bool, verbose: bool) -> None:
    """Simula el Mundial 2026 completo (104 partidos × N réplicas)."""
    from pathlib import Path

    from src.tournament import (
        DEFAULT_TOURNAMENT_REPLICAS,
        TournamentSimulator,
        TournamentStructure,
        default_structure_path,
    )

    _setup_logging(verbose)
    n_replicas = _validate_replicas(None, None,  # type: ignore[arg-type]
                                    replicas or DEFAULT_TOURNAMENT_REPLICAS)
    engine = _build_engine(offline)
    try:
        structure = TournamentStructure.load(
            Path(grupos_path) if grupos_path else default_structure_path())
        sim = TournamentSimulator(engine, structure,
                                  n_replicas=n_replicas, seed=seed)
        if as_json:
            result = sim.run()
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Simulando torneo completo × {n_replicas:,} réplicas…",
                    total=n_replicas)
                result = sim.run(progress=lambda inc: progress.advance(task, inc))
    except WorldCupEngineError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    console.print(Panel.fit(
        f"Copa del Mundo 2026 · 12 grupos + eliminación directa de 32\n"
        f"{result.n_replicas:,} réplicas · semilla {result.seed} · "
        f"μ = {result.tournament_mu:.4f} · λ₃ = {result.lambda3:.4f} · "
        f"SE ≤ {result.max_standard_error:.4f}",
        title="🏆 Simulación de torneo completo", border_style="green"))
    table = Table(title=f"Top {top} — probabilidades por etapa alcanzada")
    table.add_column("#", justify="right")
    table.add_column("Selección", style="bold")
    table.add_column("Campeón", justify="right")
    table.add_column("", no_wrap=True)
    table.add_column("Final", justify="right")
    table.add_column("Semis", justify="right")
    table.add_column("Cuartos", justify="right")
    table.add_column("Octavos", justify="right")
    for pos, (code, stages) in enumerate(result.ranking()[:top], start=1):
        table.add_row(
            str(pos), engine.display_name(code),
            f"{stages['champion']:.2%}",
            f"[yellow]{_prob_bar(stages['champion'], width=16)}[/yellow]",
            f"{stages['final']:.2%}",
            f"{stages['semi_final']:.2%}",
            f"{stages['quarter_final']:.2%}",
            f"{stages['round_of_16']:.2%}")
    table.caption = ("Cuadro: sorteo oficial 5-dic-2025 + llaves FIFA 73–104 "
                     "(data/fallback/groups_2026.json, editable). "
                     "Partidos neutrales: sin TIF ni mercado.")
    console.print(table)
    if result.matching_fallbacks:
        err_console.print(
            f"[yellow]⚠ {result.matching_fallbacks} réplicas requirieron "
            "asignación libre de terceros (revise el cuadro editado)[/yellow]")


def main() -> None:
    """Punto de entrada del ejecutable ``wc26``."""
    try:
        cli(standalone_mode=True)
    except KeyboardInterrupt:  # pragma: no cover
        err_console.print("[red]Interrumpido por el usuario[/red]")
        sys.exit(130)


if __name__ == "__main__":
    main()
