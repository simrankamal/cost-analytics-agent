import json

from cost_analytics_agent.agent import build_system_prompt, build_tools


def test_run_sql(toolbox):
    out = json.loads(toolbox.run_sql("select count(*) as n from daily_total"))
    assert out["rows"][0]["n"] == 548


def test_run_sql_returns_errors(toolbox):
    assert "unknown table" in json.loads(toolbox.run_sql("select * from focus_cost"))["error"]
    assert "error" in json.loads(toolbox.run_sql("select nope from daily_total"))


def test_run_sql_truncates(toolbox):
    out = json.loads(toolbox.run_sql("select * from daily_cost"))
    assert out["truncated"] and len(out["rows"]) == 200


def test_detect_anomalies_window(toolbox):
    out = json.loads(toolbox.detect_anomalies(start="2026-08-10", end="2026-08-18"))
    assert "2026-08-14" in [d["date"] for d in out["days"]]
    assert all("2026-08-10" <= d["date"] <= "2026-08-18" for d in out["days"])


def test_detect_anomalies_unknown_filter_value(toolbox):
    assert "error" in json.loads(toolbox.detect_anomalies(account="nope"))


def test_explain_change(toolbox):
    out = json.loads(toolbox.explain_change("2026-08-14", "2026-08-14"))
    assert out["top_drivers"][0]["usage_type"] == "Lambda-GB-Second"
    assert out["baseline"] == ["2026-07-17", "2026-08-13"]
    assert set(out["by_dimension"]) >= {"account", "service", "region"}


def test_explain_change_needs_full_baseline(toolbox):
    assert "error" in json.loads(toolbox.explain_change("2026-08-14", "2026-08-14", "2026-07-01"))


def test_forecast_cost(toolbox):
    out = json.loads(toolbox.forecast_cost(30, account="prod-core"))
    assert out["total_lower80"] < out["total"] < out["total_upper80"]
    assert out["history_end"] == "2026-09-30"
    assert "error" in json.loads(toolbox.forecast_cost(0))


def test_tool_definitions(toolbox):
    tools = {t.name: t.to_dict() for t in build_tools(toolbox)}
    assert set(tools) == {"run_sql", "detect_anomalies", "explain_change", "forecast_cost"}
    assert tools["run_sql"]["input_schema"]["required"] == ["sql"]
    assert "current_start" in tools["explain_change"]["input_schema"]["properties"]


def test_system_prompt_is_stable(toolbox):
    prompt = build_system_prompt(toolbox)
    assert "daily_cost(" in prompt and "2026-09-30" in prompt
    assert prompt == build_system_prompt(toolbox)
