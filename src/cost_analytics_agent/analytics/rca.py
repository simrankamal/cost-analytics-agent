from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

Z_THRESHOLD = 3.5
MIN_MONTHS_FOR_PROFILE = 6
SEASONAL_SMOOTHER = 15
DIMENSIONS = ["account", "region", "service", "usage_type", "pricing_category"]
FINEST = ["account", "region", "service", "usage_type"]


def _month_position(index: pd.DatetimeIndex) -> np.ndarray:
    from_end = index.days_in_month - index.day
    return np.where(index.day <= 15, index.day, -(from_end + 1))


def _decompose(series: pd.Series, period: int):
    return STL(series, period=period, seasonal=SEASONAL_SMOOTHER, robust=True).fit()


def _monthly_component(series: pd.Series, period: int) -> pd.Series:
    fit = _decompose(series, period)
    ratio = (series - fit.trend - fit.seasonal) / fit.trend
    return ratio.groupby(_month_position(series.index)).transform("median") * fit.trend


def detect_stl(series: pd.Series, period: int = 7, threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    series = series.astype(float)
    monthly = 0.0
    if len(series) >= MIN_MONTHS_FOR_PROFILE * 31:
        monthly = _monthly_component(series, period)
    fit = _decompose(series - monthly, period)
    expected = fit.trend + fit.seasonal + monthly
    resid = series - expected
    mad = np.median(np.abs(resid - np.median(resid)))
    z = 0.6745 * (resid - np.median(resid)) / mad
    return pd.DataFrame(
        {
            "actual": series,
            "expected": expected,
            "residual": resid,
            "z": z,
            "is_anomaly": np.abs(z) > threshold,
        }
    )


def detect_isolation_forest(series: pd.Series, contamination: float = 0.02) -> pd.Series:
    frame = pd.DataFrame(
        {
            "value": series,
            "change_1d": series.pct_change().fillna(0),
            "vs_week_ago": (series / series.shift(7) - 1).fillna(0),
            "dow": series.index.dayofweek,
        }
    )
    model = IsolationForest(contamination=contamination, random_state=0).fit(frame)
    return pd.Series(model.predict(frame) == -1, index=series.index)


@dataclass
class Explanation:
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    baseline_daily: float
    current_daily: float
    delta_daily: float
    drivers: pd.DataFrame
    by_dimension: dict[str, pd.DataFrame]


def _mean_daily(df: pd.DataFrame, dims: list[str], start, end) -> pd.Series:
    window = df[(df["date"] >= start) & (df["date"] <= end)]
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return window.groupby(dims)["effective_cost"].sum() / days


def _contributions(df, dims, cur, base, total_delta) -> pd.DataFrame:
    current = _mean_daily(df, dims, *cur).rename("current_daily")
    baseline = _mean_daily(df, dims, *base).rename("baseline_daily")
    out = pd.concat([baseline, current], axis=1).fillna(0.0)
    out["delta_daily"] = out["current_daily"] - out["baseline_daily"]
    out["share_of_change"] = out["delta_daily"] / total_delta if total_delta else np.nan
    out = out.reindex(out["delta_daily"].abs().sort_values(ascending=False).index)
    return out.reset_index()


def explain_change(
    daily_cost: pd.DataFrame,
    current: tuple[str, str],
    baseline: tuple[str, str] | None = None,
    baseline_days: int = 28,
    top: int = 5,
) -> Explanation:
    df = daily_cost.copy()
    df["date"] = pd.to_datetime(df["date"])
    cur = (pd.Timestamp(current[0]), pd.Timestamp(current[1]))
    if baseline is None:
        base = (cur[0] - pd.Timedelta(days=baseline_days), cur[0] - pd.Timedelta(days=1))
    else:
        base = (pd.Timestamp(baseline[0]), pd.Timestamp(baseline[1]))

    cur_total = float(_mean_daily(df.assign(all="all"), ["all"], *cur).sum())
    base_total = float(_mean_daily(df.assign(all="all"), ["all"], *base).sum())
    total_delta = cur_total - base_total

    drivers = _contributions(df, FINEST, cur, base, total_delta).head(top)
    by_dimension = {
        dim: _contributions(df, [dim], cur, base, total_delta).head(top) for dim in DIMENSIONS
    }
    return Explanation(
        current_start=cur[0],
        current_end=cur[1],
        baseline_start=base[0],
        baseline_end=base[1],
        baseline_daily=base_total,
        current_daily=cur_total,
        delta_daily=total_delta,
        drivers=drivers,
        by_dimension=by_dimension,
    )
