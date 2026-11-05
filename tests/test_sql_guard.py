import pytest

from cost_analytics_agent.sql_guard import UnsafeSQLError, guard

TABLES = {"daily_cost", "daily_total"}


def test_adds_limit():
    assert guard("select * from daily_cost", TABLES, 50).lower().endswith("limit 50")


def test_keeps_smaller_limit():
    assert guard("select * from daily_cost limit 5", TABLES, 50).lower().endswith("limit 5")


def test_caps_larger_limit():
    assert guard("select * from daily_cost limit 5000", TABLES, 50).lower().endswith("limit 50")


def test_allows_cte_and_join():
    sql = """
        with top as (select service, sum(effective_cost) c from daily_cost group by 1)
        select t.service, d.date from top t join daily_cost d on d.service = t.service
    """
    assert "top" in guard(sql, TABLES)


@pytest.mark.parametrize(
    "sql",
    [
        "drop table daily_cost",
        "delete from daily_cost",
        "insert into daily_cost select * from daily_cost",
        "create table x as select 1",
        "select 1; select 2",
        "select * from focus_cost",
        "select * from main.daily_cost",
        "select * from read_parquet('data/focus/focus_cost.parquet')",
        "select * from read_csv('/etc/passwd')",
        "select * from '/etc/passwd'",
        "select getenv('HOME')",
        "select * from duckdb_settings()",
        "copy daily_cost to 'out.csv'",
        "attach 'other.db' as o",
        "pragma database_list",
        "set memory_limit = '10GB'",
        "select * from daily_cost where",
        "",
    ],
)
def test_rejects(sql):
    with pytest.raises(UnsafeSQLError):
        guard(sql, TABLES)
