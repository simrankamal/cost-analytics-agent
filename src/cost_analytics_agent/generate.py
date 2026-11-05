"""Generate a synthetic FOCUS cost dataset priced with real AWS list prices.

Usage: uv run python -m cost_analytics_agent.generate
Writes data/focus/focus_cost.parquet and data/focus/incidents.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cost_analytics_agent.focus_schema import COLUMNS, validate

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "config" / "scenario.yaml"
CATALOG = ROOT / "data" / "seed" / "price_catalog.csv"
OUT_DIR = ROOT / "data" / "focus"

HOURLY_UNITS = {"Hrs", "Hours", "LCU-Hrs"}
MONTH_END_DAYS = 3
AR_PHI = 0.6
ARN_SERVICE = {
    "ec2": "ec2",
    "ebs": "ec2",
    "nat": "ec2",
    "data": "ec2",
    "rds": "rds",
    "elasticache": "elasticache",
    "s3": "s3",
    "alb": "elasticloadbalancing",
    "lambda": "lambda",
    "dynamodb": "dynamodb",
    "eks": "eks",
    "cloudwatch": "logs",
}


def workload_key(item: dict) -> tuple[str, str, str]:
    return item["account"], item["region"], item["sku"]


def date_mask(dates: pd.DatetimeIndex, start, end) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def base_units(w: dict, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    months = (dates - dates[0]).days.to_numpy() / 30.44
    units = w["base"] * (1 + w.get("growth", 0.0)) ** months
    weekend = dates.dayofweek.to_numpy() >= 5
    units = np.where(weekend, units * w.get("weekend", 1.0), units)
    if "month_end" in w:
        month_end = dates.day.to_numpy() > dates.days_in_month.to_numpy() - MONTH_END_DAYS
        units = np.where(month_end, units * w["month_end"], units)
    z = rng.standard_normal(len(dates))
    sigma = w.get("noise", 0.0)
    if sigma:
        e = np.empty(len(dates))
        e[0] = sigma * z[0]
        scale = np.sqrt(1 - AR_PHI**2) * sigma
        for i in range(1, len(dates)):
            e[i] = AR_PHI * e[i - 1] + scale * z[i]
        units = units * np.exp(e - sigma**2 / 2)
    return units


def apply_incident(units: np.ndarray, inc: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    mask = date_mask(dates, inc["start"], inc["end"])
    if "factor" in inc:
        units = np.where(mask, units * inc["factor"], units)
    if "add" in inc:
        units = np.where(mask, units + inc["add"], units)
    return units


def to_quantity(units: np.ndarray, pricing_unit: str, dates: pd.DatetimeIndex) -> np.ndarray:
    if pricing_unit in HOURLY_UNITS:
        return units * 24
    if pricing_unit == "GB-Mo":
        return units / dates.days_in_month.to_numpy()
    return units


def committed_hours(commitments: list[dict], dates: pd.DatetimeIndex) -> tuple[np.ndarray, list]:
    hours = np.zeros(len(dates))
    ids: list = [None] * len(dates)
    for c in commitments:
        mask = date_mask(dates, c["start"], c["end"])
        hours = np.where(mask, hours + c["count"] * 24, hours)
        for i in np.flatnonzero(mask):
            ids[i] = c["id"]
    return hours, ids


def split_usage(qty: np.ndarray, committed: np.ndarray) -> tuple[np.ndarray, ...]:
    covered = np.minimum(qty, committed)
    return qty - covered, covered, committed - covered


def effective_cost(qty: np.ndarray, committed: np.ndarray, price: pd.Series) -> np.ndarray:
    ondemand, covered, unused = split_usage(qty, committed)
    rate = price["commit_unit_price"] if committed.any() else 0.0
    return ondemand * price["list_unit_price"] + (covered + unused) * rate


class Builder:
    def __init__(self, scenario: dict, dates: pd.DatetimeIndex):
        self.scenario = scenario
        self.dates = dates
        self.billing_start = dates.to_period("M").start_time
        self.billing_end = (dates.to_period("M") + 1).start_time
        self.frames: list[pd.DataFrame] = []

    def common(self, w: dict, price: pd.Series, mask: np.ndarray) -> dict:
        account = self.scenario["accounts"][w["account"]]
        billing = self.scenario["billing_account"]
        prefix = ARN_SERVICE[w["sku"].split("_")[0]]
        return {
            "BillingAccountId": billing["id"],
            "BillingAccountName": billing["name"],
            "SubAccountId": account["id"],
            "SubAccountName": w["account"],
            "BillingPeriodStart": self.billing_start[mask],
            "BillingPeriodEnd": self.billing_end[mask],
            "ChargePeriodStart": self.dates[mask],
            "ChargePeriodEnd": self.dates[mask] + pd.Timedelta(days=1),
            "ProviderName": "AWS",
            "PublisherName": "Amazon Web Services",
            "InvoiceIssuerName": "Amazon Web Services",
            "ServiceName": price["service_name"],
            "ServiceCategory": price["service_category"],
            "RegionId": price["region_id"],
            "RegionName": price["region_name"],
            "SkuId": price["sku_id"],
            "SkuPriceId": price["sku_price_id"],
            "PricingUnit": price["pricing_unit"],
            "ConsumedUnit": price["pricing_unit"],
            "BillingCurrency": "USD",
            "Tags": json.dumps({"team": account["team"], "env": account["env"], "app": w["app"]}),
            "x_UsageType": price["usage_type"],
            "x_Operation": price["operation"],
            "ResourceId": f"arn:aws:{prefix}:{price['region_id']}:{account['id']}:{w['app']}",
            "ResourceName": w["app"],
        }

    def usage_rows(self, w, price, qty, list_rate, effective_rate, billed_rate, **extra):
        mask = qty > 0
        if not mask.any():
            return
        q = qty[mask]
        consumed = None if extra.get("CommitmentDiscountStatus") == "Unused" else q
        list_cost = 0.0 if consumed is None else q * list_rate
        self.frames.append(
            pd.DataFrame(
                {
                    **self.common(w, price, mask),
                    "ChargeCategory": "Usage",
                    "ChargeFrequency": "Usage-Based",
                    "ChargeDescription": price["price_description"],
                    "PricingQuantity": q,
                    "ConsumedQuantity": consumed,
                    "ListUnitPrice": list_rate,
                    "ListCost": list_cost,
                    "ContractedUnitPrice": list_rate,
                    "ContractedCost": list_cost,
                    "BilledCost": q * billed_rate,
                    "EffectiveCost": q * effective_rate,
                    **{
                        k: (np.asarray(v, dtype=object)[mask] if isinstance(v, list) else v)
                        for k, v in extra.items()
                    },
                }
            )
        )

    def purchase_rows(self, w, price, commitment: dict):
        mask = date_mask(self.dates, commitment["start"], commitment["end"])
        if not mask.any():
            return
        hours = commitment["count"] * 24.0
        self.frames.append(
            pd.DataFrame(
                {
                    **self.common(w, price, mask),
                    "ResourceId": commitment["id"],
                    "ResourceName": commitment["id"],
                    "ChargeCategory": "Purchase",
                    "ChargeFrequency": "Recurring",
                    "ChargeDescription": "1-year No Upfront reservation recurring fee",
                    "PricingQuantity": hours,
                    "ConsumedQuantity": None,
                    "ListUnitPrice": price["commit_unit_price"],
                    "ListCost": hours * price["commit_unit_price"],
                    "ContractedUnitPrice": price["commit_unit_price"],
                    "ContractedCost": hours * price["commit_unit_price"],
                    "BilledCost": hours * price["commit_unit_price"],
                    "EffectiveCost": 0.0,
                    "CommitmentDiscountId": commitment["id"],
                    "CommitmentDiscountName": commitment["id"],
                    "CommitmentDiscountType": "Reserved Instance",
                    "CommitmentDiscountCategory": "Usage",
                }
            )
        )


def generate() -> tuple[pd.DataFrame, list[dict]]:
    scenario = yaml.safe_load(SCENARIO.read_text())
    catalog = pd.read_csv(CATALOG).set_index(["key", "region_id"], drop=False)
    dates = pd.date_range(scenario["start_date"], scenario["end_date"], freq="D")
    rng = np.random.default_rng(scenario["seed"])

    keys = [workload_key(w) for w in scenario["workloads"]]
    if len(keys) != len(set(keys)):
        raise ValueError("each (account, region, sku) may appear in only one workload")

    commitments: dict[tuple, list[dict]] = {}
    for c in scenario["commitments"]:
        commitments.setdefault(workload_key(c), []).append(c)
    incidents: dict[tuple, list[dict]] = {}
    for inc in scenario["incidents"]:
        incidents.setdefault(workload_key(inc), []).append(inc)

    builder = Builder(scenario, dates)
    truth = []
    for w in scenario["workloads"]:
        key = workload_key(w)
        price = catalog.loc[(w["sku"], w["region"])]
        baseline = base_units(w, dates, rng)
        w_incidents = incidents.get(key, [])
        units = baseline
        for inc in w_incidents:
            units = apply_incident(units, inc, dates)
        qty = to_quantity(units, price["pricing_unit"], dates)

        w_commitments = commitments.get(key, [])
        committed, commitment_ids = committed_hours(w_commitments, dates)
        ondemand, covered, unused = split_usage(qty, committed)
        list_rate = price["list_unit_price"]
        builder.usage_rows(
            w, price, ondemand, list_rate, list_rate, list_rate, PricingCategory="Standard"
        )
        if w_commitments:
            commit = {
                "PricingCategory": "Committed",
                "CommitmentDiscountId": commitment_ids,
                "CommitmentDiscountName": commitment_ids,
                "CommitmentDiscountType": "Reserved Instance",
                "CommitmentDiscountCategory": "Usage",
            }
            rate = price["commit_unit_price"]
            builder.usage_rows(
                w, price, covered, list_rate, rate, 0.0, CommitmentDiscountStatus="Used", **commit
            )
            builder.usage_rows(
                w, price, unused, list_rate, rate, 0.0, CommitmentDiscountStatus="Unused", **commit
            )
            for c in w_commitments:
                builder.purchase_rows(w, price, c)

        actual_cost = effective_cost(qty, committed, price)
        for inc in w_incidents:
            if inc["kind"] == "commitment_expiry":
                kept = [
                    {**c, "end": scenario["end_date"]} if c["id"] == inc["commitment"] else c
                    for c in w_commitments
                ]
                counterfactual = effective_cost(qty, committed_hours(kept, dates)[0], price)
            else:
                cf_units = baseline
                for other in w_incidents:
                    if other is not inc and other["kind"] != "commitment_expiry":
                        cf_units = apply_incident(cf_units, other, dates)
                cf_qty = to_quantity(cf_units, price["pricing_unit"], dates)
                counterfactual = effective_cost(cf_qty, committed, price)
            delta = (actual_cost - counterfactual)[date_mask(dates, inc["start"], inc["end"])]
            window = dates[date_mask(dates, inc["start"], inc["end"])]
            peak = int(np.argmax(np.abs(delta)))
            truth.append(
                {
                    "id": inc["id"],
                    "kind": inc["kind"],
                    "start": str(pd.Timestamp(inc["start"]).date()),
                    "end": str(pd.Timestamp(inc["end"]).date()),
                    "sub_account_name": w["account"],
                    "region_id": w["region"],
                    "service_name": price["service_name"],
                    "usage_type": price["usage_type"],
                    "sku_key": w["sku"],
                    "cause": inc["cause"],
                    "total_impact": round(float(delta.sum()), 2),
                    "mean_daily_impact": round(float(delta.mean()), 2),
                    "peak_daily_impact": round(float(delta[peak]), 2),
                    "peak_date": str(window[peak].date()),
                }
            )

    df = pd.concat(builder.frames, ignore_index=True).reindex(columns=COLUMNS)
    for col in [
        "PricingQuantity",
        "ConsumedQuantity",
        "ListCost",
        "ContractedCost",
        "BilledCost",
        "EffectiveCost",
    ]:
        df[col] = df[col].astype(float).round(6)
    df = df.sort_values(["ChargePeriodStart", "SubAccountName", "ServiceName", "RegionId"])
    return df.reset_index(drop=True), sorted(truth, key=lambda t: t["start"])


def main() -> None:
    df, truth = generate()
    errors = validate(df)
    if errors:
        raise SystemExit("FOCUS validation failed:\n" + "\n".join(errors))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "focus_cost.parquet", index=False)
    (OUT_DIR / "incidents.json").write_text(json.dumps(truth, indent=2) + "\n")
    days = df["ChargePeriodStart"].nunique()
    print(f"{len(df):,} rows, {days} days, ${df['EffectiveCost'].sum():,.0f} effective cost")
    print(f"{len(truth)} incidents written")


if __name__ == "__main__":
    main()
