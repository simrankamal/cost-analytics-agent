"""Extract a small unit-price catalog from the AWS Price List CSVs (Kaggle mirror).

Usage: uv run python scripts/build_price_catalog.py
Reads data/raw/pricing/*.csv, writes data/seed/price_catalog.csv.
"""

import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pricing"
OUT = ROOT / "data" / "seed" / "price_catalog.csv"


def quote(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def usage_pattern(usage_type: str, prefixes: list[str]) -> str:
    alternatives = "|".join(re.escape(p) for p in prefixes)
    return rf"^({alternatives})?{re.escape(usage_type)}$"


def load_sources(con, spec: dict) -> None:
    by_file: dict[str, list[dict]] = {}
    for sku in spec["skus"]:
        by_file.setdefault(sku["file"], []).append(sku)
    regions = spec["regions"]
    for file, skus in by_file.items():
        pattern = "|".join(f"({s['pattern']})" for s in skus)
        region_cols = {s.get("region_column", "Region Code") for s in skus}
        region_where = " or ".join(
            f"{quote(c)} in ({', '.join('?' for _ in regions)})" for c in region_cols
        )
        con.execute(
            f"""
            create table {quote(file)} as
            select * from read_csv(?, all_varchar = true, header = true)
            where regexp_matches(usageType, ?) and ({region_where})
            """,
            [str(RAW / f"{file}.csv"), pattern, *(regions * len(region_cols))],
        )


def fetch(con, sku: dict, region: str, term_where: str) -> pd.DataFrame:
    region_col = sku.get("region_column", "Region Code")
    location_col = "From Location" if region_col == "From Region Code" else "Location"
    where = [f"{quote(region_col)} = ?", "regexp_matches(usageType, ?)", term_where]
    args = [region, sku["pattern"]]
    for col, val in (sku.get("filters") or {}).items():
        where.append(f"{quote(col)} = ?")
        args.append(val)
    sql = f"""
        select SKU, RateCode, usageType, operation, Unit, serviceName,
               {quote(location_col)} as location, PriceDescription,
               cast(PricePerUnit as double) as price,
               cast(StartingRange as double) as starting_range
        from {quote(sku["file"])}
        where {" and ".join(where)}
        order by starting_range, usageType
    """
    return con.execute(sql, args).df()


def build() -> pd.DataFrame:
    spec = yaml.safe_load((ROOT / "config" / "catalog_spec.yaml").read_text())
    for sku in spec["skus"]:
        sku["pattern"] = usage_pattern(sku["usage_type"], spec["usage_type_prefixes"])
    con = duckdb.connect()
    load_sources(con, spec)
    rows = []
    for sku in spec["skus"]:
        for region in spec["regions"]:
            ondemand = fetch(con, sku, region, "TermType = 'OnDemand'")
            ondemand = ondemand[ondemand["starting_range"] == 0]
            if ondemand.empty:
                raise ValueError(f"no on-demand price for {sku['key']} in {region}")
            if ondemand["price"].nunique() > 1:
                raise ValueError(
                    f"ambiguous price for {sku['key']} in {region}:\n{ondemand.to_string()}"
                )
            first = ondemand.iloc[0]
            commit_price = None
            if sku.get("commitment"):
                term = (
                    "TermType = 'Reserved' and LeaseContractLength = '1yr' "
                    "and PurchaseOption = 'No Upfront' and Unit = 'Hrs'"
                )
                if "OfferingClass" in con.table(quote(sku["file"])).columns:
                    term += " and OfferingClass = 'standard'"
                reserved = fetch(con, sku, region, term)
                if reserved["price"].nunique() != 1:
                    raise ValueError(f"expected one 1yr No Upfront rate for {sku['key']} {region}")
                commit_price = float(reserved.iloc[0]["price"])
            rows.append(
                {
                    "key": sku["key"],
                    "service_name": sku.get("service_name", first["serviceName"]),
                    "service_category": sku["service_category"],
                    "region_id": region,
                    "region_name": first["location"],
                    "sku_id": first["SKU"],
                    "sku_price_id": first["RateCode"],
                    "usage_type": first["usageType"],
                    "operation": first["operation"],
                    "pricing_unit": first["Unit"],
                    "list_unit_price": float(first["price"]),
                    "commit_unit_price": commit_price,
                    "price_description": first["PriceDescription"],
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    catalog = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT, index=False)
    print(f"wrote {len(catalog)} rows to {OUT.relative_to(ROOT)}")
