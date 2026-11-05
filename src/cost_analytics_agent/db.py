import threading
from pathlib import Path

import duckdb
import pandas as pd

from cost_analytics_agent.sql_guard import guard

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "focus" / "focus_cost.parquet"
DB_PATH = ROOT / "data" / "cost.duckdb"

VIEWS = {
    "daily_cost": """
        select
            cast(ChargePeriodStart as date) as date,
            SubAccountName as account,
            SubAccountId as account_id,
            RegionId as region,
            ServiceName as service,
            ServiceCategory as service_category,
            x_UsageType as usage_type,
            ChargeCategory as charge_category,
            coalesce(PricingCategory, 'None') as pricing_category,
            json_extract_string(Tags, '$.team') as team,
            json_extract_string(Tags, '$.env') as env,
            json_extract_string(Tags, '$.app') as app,
            sum(EffectiveCost) as effective_cost,
            sum(BilledCost) as billed_cost,
            sum(ListCost) as list_cost,
            sum(PricingQuantity) as usage_quantity,
            any_value(PricingUnit) as pricing_unit
        from focus_cost
        group by all
    """,
    "daily_total": """
        select
            cast(ChargePeriodStart as date) as date,
            sum(EffectiveCost) as effective_cost,
            sum(BilledCost) as billed_cost
        from focus_cost
        group by all
        order by date
    """,
    "monthly_cost": """
        select
            cast(date_trunc('month', ChargePeriodStart) as date) as month,
            SubAccountName as account,
            RegionId as region,
            ServiceName as service,
            sum(EffectiveCost) as effective_cost,
            sum(BilledCost) as billed_cost
        from focus_cost
        group by all
    """,
    "commitment_utilization": """
        select
            cast(ChargePeriodStart as date) as date,
            CommitmentDiscountId as commitment_id,
            ServiceName as service,
            RegionId as region,
            sum(PricingQuantity) filter (where CommitmentDiscountStatus = 'Used') as used_hours,
            coalesce(sum(PricingQuantity) filter (where CommitmentDiscountStatus = 'Unused'), 0)
                as unused_hours,
            sum(EffectiveCost) as effective_cost
        from focus_cost
        where ChargeCategory = 'Usage' and CommitmentDiscountId is not null
        group by all
    """,
}
ALLOWED_TABLES = set(VIEWS)


def build_database(parquet: Path = PARQUET, db_path: Path = DB_PATH) -> None:
    db_path.unlink(missing_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute("create table focus_cost as select * from read_parquet(?)", [str(parquet)])
        for name, sql in VIEWS.items():
            con.execute(f"create view {name} as {sql}")


class QueryRunner:
    def __init__(self, db_path: Path = DB_PATH, timeout_s: float = 10.0, max_rows: int = 1000):
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.con = duckdb.connect(
            str(db_path),
            read_only=True,
            config={
                "enable_external_access": False,
                "autoload_known_extensions": False,
                "autoinstall_known_extensions": False,
                "memory_limit": "1GB",
                "lock_configuration": True,
            },
        )

    def run(self, sql: str) -> tuple[str, pd.DataFrame]:
        safe_sql = guard(sql, ALLOWED_TABLES, self.max_rows)
        timer = threading.Timer(self.timeout_s, self.con.interrupt)
        timer.start()
        try:
            df = self.con.execute(safe_sql).df()
        finally:
            timer.cancel()
        return safe_sql, df

    def schema(self) -> dict[str, list[tuple[str, str]]]:
        return {
            name: [(r[0], r[1]) for r in self.con.execute(f"describe {name}").fetchall()]
            for name in sorted(ALLOWED_TABLES)
        }

    def close(self) -> None:
        self.con.close()


if __name__ == "__main__":
    build_database()
    print(f"built {DB_PATH.relative_to(ROOT)}")
