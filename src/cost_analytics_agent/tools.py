import json

import duckdb
import pandas as pd

from cost_analytics_agent.analytics import forecast as fc
from cost_analytics_agent.analytics import rca
from cost_analytics_agent.db import QueryRunner
from cost_analytics_agent.sql_guard import UnsafeSQLError

FILTERS = ("account", "service", "region")
MAX_RESULT_ROWS = 200
MIN_HISTORY_DAYS = 60


def _records(df: pd.DataFrame, digits: int = 2) -> list[dict]:
    df = df.copy()
    for col in df.select_dtypes(include="datetime").columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].round(digits)
    return json.loads(df.to_json(orient="records"))


def _dump(payload: dict) -> str:
    return json.dumps(payload, default=str)


class Toolbox:
    def __init__(self, runner: QueryRunner):
        self.runner = runner

    def _where(self, filters: dict[str, str | None]) -> tuple[str, list[str]]:
        active = {k: v for k, v in filters.items() if v}
        unknown = set(active) - set(FILTERS)
        if unknown:
            raise ValueError(f"unknown filters: {sorted(unknown)}")
        clause = " and ".join(f"{k} = ?" for k in active)
        return (f"where {clause}" if clause else ""), list(active.values())

    def daily_series(self, **filters: str | None) -> pd.Series:
        where, args = self._where(filters)
        df = self.runner.con.execute(
            f"select date, sum(effective_cost) as cost from daily_cost {where} "
            "group by date order by date",
            args,
        ).df()
        if df.empty:
            raise ValueError(f"no cost rows match {filters}")
        series = pd.Series(df["cost"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        return series.asfreq("D", fill_value=0.0)

    def date_range(self) -> tuple[str, str]:
        lo, hi = self.runner.con.execute("select min(date), max(date) from daily_total").fetchone()
        return str(lo), str(hi)

    def run_sql(self, sql: str) -> str:
        try:
            executed, df = self.runner.run(sql)
        except (UnsafeSQLError, duckdb.Error) as e:
            return _dump({"error": str(e)})
        return _dump(
            {
                "sql": executed,
                "row_count": len(df),
                "truncated": len(df) > MAX_RESULT_ROWS,
                "rows": _records(df.head(MAX_RESULT_ROWS), digits=4),
            }
        )

    def detect_anomalies(
        self,
        start: str | None = None,
        end: str | None = None,
        account: str | None = None,
        service: str | None = None,
        region: str | None = None,
    ) -> str:
        try:
            series = self.daily_series(account=account, service=service, region=region)
        except ValueError as e:
            return _dump({"error": str(e)})
        result = rca.detect_stl(series)
        result["forest_flag"] = rca.detect_isolation_forest(series)
        flagged = result[result["is_anomaly"]].loc[start:end]
        flagged = flagged.reindex(flagged["z"].abs().sort_values(ascending=False).index)
        out = flagged.head(30).sort_index().reset_index(names="date")
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return _dump(
            {
                "method": "STL residual robust z-score, weekly and day-of-month seasonality",
                "threshold": rca.Z_THRESHOLD,
                "anomalous_days": len(flagged),
                "shown": len(out),
                "days": _records(out.drop(columns="is_anomaly")),
            }
        )

    def explain_change(
        self,
        current_start: str,
        current_end: str,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
    ) -> str:
        if bool(baseline_start) != bool(baseline_end):
            return _dump({"error": "give both baseline_start and baseline_end, or neither"})
        baseline = (baseline_start, baseline_end) if baseline_start else None
        daily_cost = self.runner.con.execute(
            "select date, account, region, service, usage_type, pricing_category, effective_cost "
            "from daily_cost"
        ).df()
        e = rca.explain_change(daily_cost, (current_start, current_end), baseline)
        return _dump(
            {
                "current": [e.current_start.date(), e.current_end.date()],
                "baseline": [e.baseline_start.date(), e.baseline_end.date()],
                "baseline_daily": round(e.baseline_daily, 2),
                "current_daily": round(e.current_daily, 2),
                "delta_daily": round(e.delta_daily, 2),
                "top_drivers": _records(e.drivers, digits=3),
                "by_dimension": {k: _records(v, digits=3) for k, v in e.by_dimension.items()},
            }
        )

    def forecast_cost(
        self,
        horizon_days: int = 30,
        account: str | None = None,
        service: str | None = None,
        region: str | None = None,
    ) -> str:
        if not 1 <= horizon_days <= 180:
            return _dump({"error": "horizon_days must be between 1 and 180"})
        try:
            series = self.daily_series(account=account, service=service, region=region)
        except ValueError as e:
            return _dump({"error": str(e)})
        if len(series) < MIN_HISTORY_DAYS:
            return _dump({"error": f"need at least {MIN_HISTORY_DAYS} days of history"})
        folds = min(4, (len(series) - MIN_HISTORY_DAYS // 2) // horizon_days)
        if folds:
            scores = fc.backtest(series, horizon=horizon_days, folds=folds)
            model = fc.best_model(scores)
            backtest = _records(scores.groupby("model").mean(numeric_only=True).reset_index())
        else:
            model, backtest = "seasonal_naive", []
        f = fc.forecast(series, horizon_days, model)
        weekly = pd.DataFrame({"mean": f.mean, "lower80": f.lower80, "upper80": f.upper80})
        weekly = weekly.resample("W").sum().reset_index(names="week_ending")
        weekly["week_ending"] = weekly["week_ending"].dt.strftime("%Y-%m-%d")
        return _dump(
            {
                "model": model,
                "history_end": series.index[-1].date(),
                "horizon_days": horizon_days,
                "total": round(f.total, 2),
                "total_lower80": round(f.total_lower80, 2),
                "total_upper80": round(f.total_upper80, 2),
                "trailing_same_length_total": round(float(series.iloc[-horizon_days:].sum()), 2),
                "backtest_mean_error_pct": backtest,
                "weekly": _records(weekly),
            }
        )
