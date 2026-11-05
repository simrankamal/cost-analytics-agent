# cost-analytics-agent

A conversational agent that answers questions about a cloud bill: what was spent, what changed
and why, and what next month will cost. It runs over [FOCUS 1.2](https://focus.finops.org/)
billing data in DuckDB.

The dataset is synthetic but priced with real AWS list and 1-year reserved rates. Nine incidents
(spikes, level shifts, a cleanup, an expired reservation) are injected with known ground truth,
so the analytics can be tested against the right answer.

## Setup

```sh
uv sync
uv run python -m cost_analytics_agent.generate   # data/focus/focus_cost.parquet + incidents.json
uv run python -m cost_analytics_agent.db         # data/cost.duckdb
export ANTHROPIC_API_KEY=...
uv run cost-agent
```

```
> why did spend jump in early February 2026?
> is anything unusual in August?
> how well are our reservations utilized, and did any expire?
> what will prod-core cost over the next 30 days?
```

`uv run cost-agent "question"` answers a single question and exits.

## How it works

| Module | Role |
| --- | --- |
| `generate.py` | Builds the FOCUS dataset from `config/scenario.yaml` and `data/seed/price_catalog.csv` |
| `focus_schema.py` | Column set and a subset of the FOCUS conformance rules |
| `db.py` | Loads the parquet into DuckDB, defines the analyst-facing views, runs guarded queries |
| `sql_guard.py` | Parses model-written SQL and rejects anything but a single SELECT over the views |
| `analytics/rca.py` | STL and isolation-forest anomaly detection; delta decomposition across dimensions |
| `analytics/forecast.py` | Seasonal-naive and ETS forecasts, chosen by rolling-origin backtest |
| `tools.py` | The four operations the model can call: `run_sql`, `detect_anomalies`, `explain_change`, `forecast_cost` |
| `agent.py` | System prompt, tool definitions, and the conversation loop |
| `cli.py` | Terminal chat |

The model never computes numbers itself. It writes SQL or calls an analytics tool, and reports
what comes back.

Model-written SQL is checked twice: `sql_guard` allows one SELECT over four views with a row
limit, and the DuckDB connection is read-only with external access disabled and a query timeout.

## Tests

```sh
uv run pytest
```

Tests regenerate the dataset in memory, so they need only the committed price catalog. They cover
FOCUS conformance, the SQL guard, root-cause recovery of all nine incidents, and the forecasting
backtest.

## Rebuilding the price catalog

`data/seed/price_catalog.csv` is committed. To rebuild it, download the AWS Price List CSVs into
`data/raw/pricing/` and run `uv run python scripts/build_price_catalog.py`.
