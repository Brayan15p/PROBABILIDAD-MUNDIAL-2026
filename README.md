# ⚽ PROBABILIDAD-MUNDIAL-2026

Motor predictivo estocástico de nivel producción para la Copa del Mundo FIFA 2026:
**Poisson bivariado (Karlis–Ntzoufras) + Monte Carlo (≥15.000 réplicas)**, alimentado por
APIs reales con caché local estricto, análisis NLP de prensa de los 11 titulares,
validación contra mercados de predicción y persistencia en Supabase.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              src/cli.py (click + rich)                   │
│        predecir · equipos · calibrar · cache · db                        │
└────────────┬─────────────────────────────────────────────┬───────────────┘
             │                                             │
┌────────────▼────────────────┐               ┌────────────▼───────────────┐
│ src/api_client.py           │               │ src/simulator.py           │
│ WorldCupDataEngine (Facade) │── agregados ─▶│ StrengthEstimator (MLE/EB) │
│  ├ SportsProviderFactory    │   y cruces    │ BivariatePoissonModel      │
│  │  ├ football-data.org     │               │ MarketBlender (log-pool)   │
│  │  └ API-Football          │               │ MonteCarloEngine (PCG64)   │
│  ├ ZafronixHistoricalAdapter│               │ PredictionPipeline         │
│  ├ NewsAggregator           │── noticias ─▶ ┌────────────────────────────┐
│  │  ├ NewsAPI.org           │               │ src/nlp_processor.py       │
│  │  └ GNews                 │               │ TacticalSentimentAnalyzer  │
│  └ PolymarketProvider       │── mercado ──▶ │ (VADER / transformers)     │
│        │                    │               │ TIF ∈ [0.80, 1.20]         │
│  @cached → data/cache/*.json│               └────────────────────────────┘
│  FallbackDataStore (FIFA    │               ┌────────────────────────────┐
│  2014–2022 verificado)      │               │ src/db.py → Supabase       │
└─────────────────────────────┘               │ (PostgREST, best-effort)   │
                                              └────────────────────────────┘
```

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # o: pip install -e ".[dev]"
cp .env.example .env                   # añada sus API keys (opcional)
```

## Uso

```bash
# Predicción completa (funciona sin ninguna API key, modo fallback)
python -m src.cli predecir --local "México" --visitante "Argentina" --sims 20000

# Con mercados de predicción (Polymarket → consenso embebido si falla)
python -m src.cli predecir --local "España" --visitante "Francia" \
    --sims 50000 --live-odds --verbose

# Salida JSON para pipes / dashboards
python -m src.cli predecir --local ARG --visitante FRA --sims 15000 --json

# Modo estrictamente sin red (caché + fallback embebido)
python -m src.cli predecir --local Brasil --visitante Alemania --sims 15000 --offline

# Backtest de calibración (leave-one-edition-out 2014–2022)
python -m src.cli calibrar

# Observabilidad
python -m src.cli equipos --confederacion CONMEBOL
python -m src.cli cache stats
python -m src.cli db status && python -m src.cli db sync-teams
```

Instalado como paquete (`pip install -e .`), el ejecutable es `wc26`.

## Metodología (cero números mágicos)

Cada término de la forma funcional

```
λ_local = μ_torneo · Ataque_local · Defensa_visitante · TIF_local · f_mercado_local
λ_visit = μ_torneo · Ataque_visit · Defensa_local    · TIF_visit · f_mercado_visit
```

es un **estimador documentado sobre datos reales**:

| Término | Estimador | Fuente |
|---|---|---|
| `μ_torneo` | MLE Poisson: G/(2M) = 512/(2·192) ≈ 1.3333 | 3 Mundiales completos 2014–2022 (o partidos vivos de la API) |
| `Ataque/Defensa` | Regresión de Poisson (statsmodels GLM) si la muestra cumple ≥10 obs/parámetro (regla de Harrell); si no, MLE con **shrinkage bayesiano empírico** w_i = τ̂²/(τ̂²+μ/M_i), τ̂² por método de momentos | Agregados GF/GA por selección, verificados contra actas FIFA |
| `λ₃` (covarianza de goles) | Estimador de momentos de Karlis & Ntzoufras: max(0, ĉov) | 48 cruces de eliminación directa a 90' (2014–2022) |
| `TIF` | Polaridad VADER ponderada por recencia (vida media 72 h = ciclo de partidos FIFA), shrinkage hacia el prior neutro N(0, τ̂²), mapeo lineal contractual 1+0.20·p | Prensa real de los 11 titulares (NewsAPI + GNews) |
| `f_mercado` | Log-opinion pool modelo↔mercado (w=0.5 máxima entropía; calibrable por MLE con `MarketBlender.calibrate_weight`) resuelto como factores multiplicativos sobre λ | Polymarket Gamma API o consenso outright de-vig embebido |

- La **moda conjunta** (marcador exacto) se reporta con IC 95% de Wilson, validación
  cruzada contra la pmf bivariada analítica y ratio de nitidez vs la segunda moda.
- La malla analítica se trunca donde la masa de cola < 1e-9 (CDF de Poisson), no
  en un máximo de goles fijo.
- `calibrar` ejecuta el backtest *leave-one-edition-out*: el modelo queda por
  debajo del baseline uniforme (log-loss 1.094 < ln 3 ≈ 1.099; Brier 0.662 < 0.667)
  — honesto recordatorio de que los cruces entre élites son casi equiprobables.

## Cuotas free-tier y caché estricto

| Proveedor | Límite gratuito | Mitigación |
|---|---|---|
| football-data.org | 10 req/min | Throttle 6 s + TTL histórico ∞ |
| API-Football | 100 req/día | TTL plantillas 24 h |
| NewsAPI / GNews | 100 req/día c/u | TTL noticias 12 h (peor caso 88 req/día) |
| Polymarket Gamma | pública | TTL 15 min; `--live-odds` fuerza bypass |

Toda entrada vive en `data/cache/<namespace>/<sha256>.json` con envelope
versionado (timestamp + TTL); las entradas corruptas se purgan y regeneran.

## Base de datos (Supabase)

`supabase/migrations/0001_init.sql` crea: `teams`, `team_metrics`,
`market_snapshots`, `tif_reports`, `predictions` (con checks de coherencia:
probabilidades que suman 1, TIF en rango, ≥15.000 sims) y `match_results`
para backtesting, con RLS activado (lectura pública, escritura service-role).

```bash
supabase db push            # o aplicar la migración vía MCP/SQL editor
wc26 db sync-teams          # sincroniza el registro maestro
wc26 predecir ... (sin --no-db)   # cada corrida queda auditada
```

La capa es de mejor esfuerzo: sin credenciales o ante fallo, el pipeline
continúa y solo emite un warning.

## Honestidad técnica (límites conocidos)

1. **"Zafronix API" no es un servicio público verificable.** El adaptador
   implementa la interfaz solicitada: usa `ZAFRONIX_BASE_URL` si se define
   y, en su defecto, el dataset REAL FIFA 2014–2022 embebido (verificado:
   ΣGF = ΣGA = goles oficiales en cada edición; los tests lo garantizan).
2. **El snapshot de mercado embebido tiene fecha de corte** (`as_of`
   2026-01-15); `--live-odds` lo reemplaza por Polymarket en vivo.
3. **VADER es un léxico para inglés**: las consultas de prensa se hacen en
   inglés; para análisis multilingüe instale el extra `nlp-advanced`
   (backend transformers local).
4. Los onces titulares embebidos son *probables* a la fecha de corte; con
   `API_FOOTBALL_KEY` se obtienen en vivo.
5. Las tasas por partido del fallback incluyen goles de tiempo extra
   (sesgo homogéneo entre semifinalistas; los cruces a 90' se almacenan
   por separado para la covarianza).
6. xG y ratings de plantilla quedan estructurados en `TeamMetrics` y se
   pueblan cuando el plan de la API los expone; en su ausencia el MLE usa
   goles reales (proxy insesgado del xG a nivel selección-torneo).

## Tests

```bash
python -m pytest tests/ -v    # 65 tests: integridad de datos, estadística,
                              # caché, NLP, CLI end-to-end (sin red)
```
