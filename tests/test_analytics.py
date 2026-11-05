import pandas as pd
import pytest

from cost_analytics_agent.analytics import forecast as fc
from cost_analytics_agent.analytics import rca

CLEAN_BASELINE = {"INC-04": ("2026-01-01", "2026-01-28")}


def first_week(inc: dict) -> tuple[str, str]:
    start = pd.Timestamp(inc["start"])
    end = min(pd.Timestamp(inc["end"]), start + pd.Timedelta(days=6))
    return str(start.date()), str(end.date())


@pytest.mark.parametrize("inc_id", [f"INC-0{i}" for i in range(1, 10)])
def test_explain_change_finds_incident(daily_cost, incidents, inc_id):
    inc = incidents[inc_id]
    result = rca.explain_change(daily_cost, first_week(inc), CLEAN_BASELINE.get(inc_id))
    top = result.drivers.iloc[0]
    assert (top["account"], top["region"], top["usage_type"]) == (
        inc["sub_account_name"],
        inc["region_id"],
        inc["usage_type"],
    )
    assert (top["delta_daily"] > 0) == (inc["total_impact"] > 0)


def test_contributions_sum_to_total(daily_cost):
    result = rca.explain_change(daily_cost, ("2026-02-03", "2026-02-11"))
    all_services = rca._contributions(
        daily_cost.assign(date=pd.to_datetime(daily_cost["date"])),
        ["service"],
        (result.current_start, result.current_end),
        (result.baseline_start, result.baseline_end),
        result.delta_daily,
    )
    assert all_services["share_of_change"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("inc_id", ["INC-02", "INC-03", "INC-06"])
def test_stl_flags_large_spikes(daily_total, incidents, inc_id):
    flagged = rca.detect_stl(daily_total)
    peak = flagged.loc[incidents[inc_id]["peak_date"]]
    assert peak["is_anomaly"] and peak["z"] > 0


@pytest.mark.parametrize("inc_id", ["INC-01", "INC-05"])
def test_stl_flags_moderate_spikes(daily_total, incidents, inc_id):
    assert rca.detect_stl(daily_total).loc[incidents[inc_id]["peak_date"], "is_anomaly"]


def test_stl_learns_month_end_pattern(daily_total):
    result = rca.detect_stl(daily_total)
    index = result.index
    month_end = (index.days_in_month - index.day) < 3
    assert (result.loc[month_end, "expected"] > result.loc[month_end, "actual"] * 0.9).all()
    assert result.loc[month_end, "is_anomaly"].mean() < 0.15
    assert result["is_anomaly"].sum() < 40


def test_stl_short_series_skips_monthly_profile(daily_total):
    result = rca.detect_stl(daily_total.iloc[:90])
    assert len(result) == 90 and result["z"].notna().all()


def test_backtest_and_forecast(daily_total):
    scores = fc.backtest(daily_total, horizon=30, folds=4)
    assert set(scores["model"]) == set(fc.MODELS)
    assert scores.groupby("model")["period_total_error_pct"].mean().min() < 10
    result = fc.forecast(daily_total, 30, fc.best_model(scores))
    assert len(result.mean) == 30
    assert result.mean.index[0] == daily_total.index[-1] + pd.Timedelta(days=1)
    assert result.total_lower80 < result.total < result.total_upper80
    trailing = daily_total.iloc[-30:].sum()
    assert 0.8 * trailing < result.total < 1.2 * trailing
