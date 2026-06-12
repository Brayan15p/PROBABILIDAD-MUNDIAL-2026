-- =====================================================================
-- Motor Predictivo Mundial 2026 — Esquema inicial Supabase (PostgreSQL)
-- Aplicar con: supabase db push  |  o via MCP apply_migration
-- =====================================================================

-- Registro maestro de selecciones -------------------------------------
create table if not exists public.teams (
    code            text primary key,            -- trigram FIFA (ARG, FRA...)
    name            text not null,
    aliases         jsonb not null default '[]'::jsonb,
    confederation   text not null default 'UNKNOWN',
    created_at      timestamptz not null default now()
);

-- Snapshots de métricas estimadas por selección ------------------------
create table if not exists public.team_metrics (
    id                      bigint generated always as identity primary key,
    team_code               text not null references public.teams(code),
    matches_sample          integer not null,
    goals_for_per_match     double precision not null,
    goals_against_per_match double precision not null,
    attack_strength         double precision not null,
    defense_resilience      double precision not null,
    xg_for_per_match        double precision,
    xg_against_per_match    double precision,
    origin                  text not null,        -- live_api | local_cache | fallback_store
    shrinkage_weight        double precision not null default 0,
    as_of                   timestamptz not null default now()
);
create index if not exists idx_team_metrics_team_time
    on public.team_metrics (team_code, as_of desc);

-- Snapshots de mercado (Polymarket / consenso casas) -------------------
create table if not exists public.market_snapshots (
    id                  bigint generated always as identity primary key,
    home_code           text not null,
    away_code           text not null,
    p_home              double precision not null,
    p_draw              double precision,
    p_away              double precision not null,
    source              text not null,
    is_outright_derived boolean not null default false,
    origin              text not null,
    as_of               timestamptz not null default now()
);
create index if not exists idx_market_snapshots_match
    on public.market_snapshots (home_code, away_code, as_of desc);

-- Reportes del Tactical Impact Factor (NLP) ----------------------------
create table if not exists public.tif_reports (
    id              bigint generated always as identity primary key,
    team_code       text not null,
    tif             double precision not null
                    check (tif >= 0.80 and tif <= 1.20),
    team_polarity   double precision not null
                    check (team_polarity >= -1 and team_polarity <= 1),
    articles_total  integer not null default 0,
    engine          text not null,
    players         jsonb not null default '[]'::jsonb,
    as_of           timestamptz not null default now()
);
create index if not exists idx_tif_reports_team_time
    on public.tif_reports (team_code, as_of desc);

-- Auditoría completa de predicciones ejecutadas ------------------------
create table if not exists public.predictions (
    id                  bigint generated always as identity primary key,
    home_code           text not null,
    away_code           text not null,
    n_simulations       integer not null check (n_simulations >= 15000),
    lambda_home         double precision not null check (lambda_home > 0),
    lambda_away         double precision not null check (lambda_away > 0),
    lambda3_covariance  double precision not null default 0,
    tif_home            double precision not null,
    tif_away            double precision not null,
    p_home_win          double precision not null,
    p_draw              double precision not null,
    p_away_win          double precision not null,
    modal_score         text not null,
    modal_probability   double precision not null,
    seed                bigint not null,
    payload             jsonb not null,
    created_at          timestamptz not null default now(),
    constraint chk_probs_sum check (abs(p_home_win + p_draw + p_away_win - 1.0) < 0.01)
);
create index if not exists idx_predictions_match_time
    on public.predictions (home_code, away_code, created_at desc);

-- Resultados reales (para backtesting de calibración) ------------------
create table if not exists public.match_results (
    id          bigint generated always as identity primary key,
    home_code   text not null,
    away_code   text not null,
    home_goals  integer not null check (home_goals >= 0),
    away_goals  integer not null check (away_goals >= 0),
    stage       text,
    played_at   timestamptz not null,
    source      text not null default 'manual'
);

-- Seguridad: RLS activado; solo service_role escribe --------------------
alter table public.teams            enable row level security;
alter table public.team_metrics     enable row level security;
alter table public.market_snapshots enable row level security;
alter table public.tif_reports      enable row level security;
alter table public.predictions      enable row level security;
alter table public.match_results    enable row level security;

-- Lectura pública (dashboards), escritura solo backend con service key.
do $$
declare
    t text;
begin
    foreach t in array array['teams','team_metrics','market_snapshots',
                             'tif_reports','predictions','match_results']
    loop
        execute format(
            'drop policy if exists "%s_read_all" on public.%I', t, t);
        execute format(
            'create policy "%s_read_all" on public.%I for select using (true)',
            t, t);
    end loop;
end $$;
