# Instalación manual del esquema en Supabase

Guía para aplicar la base de datos del motor **tú mismo**, en el proyecto
Supabase que elijas (idealmente uno dedicado a este repositorio). Ninguna
herramienta automática toca tu proyecto: copias y pegas dos archivos SQL.

## Paso 1 — Crear el esquema (tablas, índices, RLS)

1. Abre tu proyecto en https://supabase.com/dashboard
2. Ve a **SQL Editor → New query**
3. Pega el contenido COMPLETO de [`migrations/0001_init.sql`](migrations/0001_init.sql)
4. Pulsa **Run**

Esto crea 6 tablas (todas comentadas como `PROBABILIDAD-MUNDIAL-2026:` para
identificarlas):

| Tabla | Propósito |
|---|---|
| `teams` | Registro maestro de selecciones (código FIFA, alias, confederación) |
| `team_metrics` | Snapshots de fuerzas ataque/defensa estimadas |
| `market_snapshots` | Consenso de mercado por partido |
| `tif_reports` | Tactical Impact Factor (NLP) por corrida |
| `predictions` | Auditoría completa de cada predicción Monte Carlo |
| `match_results` | Resultados reales para backtesting |

Seguridad incluida: RLS activado en todas, política de **solo lectura**
pública; las escrituras requieren la *service role key* (solo backend).

## Paso 2 — Sembrar las 63 selecciones

En el mismo SQL Editor, ejecuta el contenido de
[`seed/0002_seed_teams.sql`](seed/0002_seed_teams.sql). Es idempotente
(`on conflict ... do update`): puedes re-ejecutarlo sin duplicar filas.

## Paso 3 — Conectar la CLI

En el `.env` del repositorio:

```
SUPABASE_URL=https://<TU_PROJECT_REF>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<Dashboard → Settings → API Keys → service_role>
```

> ⚠️ La `service_role` key salta el RLS: nunca la publiques ni la subas al
> repositorio (el `.gitignore` ya excluye `.env`).

## Paso 4 — Verificar

```bash
wc26 db status        # conectividad + conteo de predicciones
wc26 db sync-teams    # (opcional) re-sincroniza el registro vía API
wc26 predecir --local "México" --visitante "Argentina" --sims 20000
wc26 db recientes     # la corrida anterior debe aparecer aquí
```

O directamente en SQL Editor:

```sql
select count(*) as equipos from public.teams;          -- esperado: 63
select * from public.predictions order by id desc limit 5;
```

## Desinstalación limpia

```sql
drop table if exists
    public.team_metrics, public.market_snapshots, public.tif_reports,
    public.predictions, public.match_results, public.teams;
```
