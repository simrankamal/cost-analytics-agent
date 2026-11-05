import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

SEASON = 7


@dataclass
class Forecast:
    model: str
    mean: pd.Series
    lower80: pd.Series
    upper80: pd.Series
    total: float
    total_lower80: float
    total_upper80: float


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _fit_ets(series: pd.Series):
    model = ETSModel(
        series.astype(float),
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASON,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False)


def seasonal_naive_mean(series: pd.Series, horizon: int) -> np.ndarray:
    last = series.to_numpy()[-SEASON:]
    return np.tile(last, int(np.ceil(horizon / SEASON)))[:horizon]


def ets_mean(series: pd.Series, horizon: int) -> np.ndarray:
    return np.asarray(_fit_ets(series).forecast(horizon))


MODELS: dict[str, Callable[[pd.Series, int], np.ndarray]] = {
    "seasonal_naive": seasonal_naive_mean,
    "ets": ets_mean,
}


def backtest(series: pd.Series, horizon: int = 30, folds: int = 4) -> pd.DataFrame:
    rows = []
    for fold in range(folds, 0, -1):
        cutoff = len(series) - fold * horizon
        train, test = series.iloc[:cutoff], series.iloc[cutoff : cutoff + horizon]
        actual = test.to_numpy()
        for name, fn in MODELS.items():
            pred = fn(train, len(test))
            rows.append(
                {
                    "model": name,
                    "cutoff": train.index[-1].date(),
                    "daily_mape": float(np.mean(np.abs(pred - actual) / actual) * 100),
                    "period_total_error_pct": float(
                        abs(pred.sum() - actual.sum()) / actual.sum() * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def best_model(scores: pd.DataFrame, metric: str = "period_total_error_pct") -> str:
    return str(scores.groupby("model")[metric].mean().idxmin())


def forecast(series: pd.Series, horizon: int, model: str = "ets", sims: int = 2000) -> Forecast:
    index = _future_index(series, horizon)
    if model == "seasonal_naive":
        mean = seasonal_naive_mean(series, horizon)
        residuals = (series - series.shift(SEASON)).dropna().to_numpy()
        rng = np.random.default_rng(0)
        paths = mean + rng.choice(residuals, size=(sims, horizon))
    else:
        fit = _fit_ets(series)
        mean = np.asarray(fit.forecast(horizon))
        paths = np.asarray(
            fit.simulate(horizon, repetitions=sims, anchor="end", rng=np.random.default_rng(0))
        ).T
    lo, hi = np.percentile(paths, [10, 90], axis=0)
    totals = paths.sum(axis=1)
    return Forecast(
        model=model,
        mean=pd.Series(mean, index=index),
        lower80=pd.Series(lo, index=index),
        upper80=pd.Series(hi, index=index),
        total=float(mean.sum()),
        total_lower80=float(np.percentile(totals, 10)),
        total_upper80=float(np.percentile(totals, 90)),
    )
