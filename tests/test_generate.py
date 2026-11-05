import pytest

from cost_analytics_agent.focus_schema import COLUMNS, validate


def test_focus_conformance(focus):
    assert list(focus.columns) == COLUMNS
    assert validate(focus) == []


def test_validate_reports_violations(focus):
    bad = focus.head(100).copy()
    bad.loc[bad.index[0], "ChargeCategory"] = "Refund"
    bad.loc[bad.index[1], "BilledCost"] = None
    errors = validate(bad)
    assert any("ChargeCategory" in e for e in errors)
    assert any("BilledCost" in e for e in errors)


def test_billed_equals_effective_overall(focus):
    usage_or_purchase = focus[focus["ChargeCategory"].isin(["Usage", "Purchase"])]
    assert usage_or_purchase["BilledCost"].sum() == pytest.approx(
        usage_or_purchase["EffectiveCost"].sum(), rel=1e-6
    )


def test_incident_truth_is_complete(incidents):
    assert len(incidents) == 9
    assert incidents["INC-08"]["total_impact"] < 0
    assert all(inc["total_impact"] != 0 for inc in incidents.values())
