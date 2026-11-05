import pandas as pd

COLUMNS = [
    "BillingAccountId",
    "BillingAccountName",
    "SubAccountId",
    "SubAccountName",
    "BillingPeriodStart",
    "BillingPeriodEnd",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "ChargeCategory",
    "ChargeClass",
    "ChargeFrequency",
    "ChargeDescription",
    "ProviderName",
    "PublisherName",
    "InvoiceIssuerName",
    "ServiceName",
    "ServiceCategory",
    "RegionId",
    "RegionName",
    "ResourceId",
    "ResourceName",
    "SkuId",
    "SkuPriceId",
    "PricingCategory",
    "PricingUnit",
    "PricingQuantity",
    "ConsumedUnit",
    "ConsumedQuantity",
    "ListUnitPrice",
    "ListCost",
    "ContractedUnitPrice",
    "ContractedCost",
    "BilledCost",
    "EffectiveCost",
    "BillingCurrency",
    "CommitmentDiscountId",
    "CommitmentDiscountName",
    "CommitmentDiscountType",
    "CommitmentDiscountCategory",
    "CommitmentDiscountStatus",
    "Tags",
    "x_UsageType",
    "x_Operation",
]

NOT_NULL = [
    "BillingAccountId",
    "BillingPeriodStart",
    "BillingPeriodEnd",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "ChargeCategory",
    "ChargeFrequency",
    "ProviderName",
    "PublisherName",
    "InvoiceIssuerName",
    "ServiceName",
    "ServiceCategory",
    "ListCost",
    "ContractedCost",
    "BilledCost",
    "EffectiveCost",
    "BillingCurrency",
]

ALLOWED = {
    "ChargeCategory": {"Usage", "Purchase", "Tax", "Credit", "Adjustment"},
    "ChargeFrequency": {"One-Time", "Recurring", "Usage-Based"},
    "PricingCategory": {"Standard", "Dynamic", "Committed", "Other"},
    "CommitmentDiscountCategory": {"Spend", "Usage"},
    "CommitmentDiscountStatus": {"Used", "Unused"},
    "ServiceCategory": {
        "AI and Machine Learning",
        "Analytics",
        "Business Applications",
        "Compute",
        "Databases",
        "Developer Tools",
        "Multicloud",
        "Identity",
        "Integration",
        "Internet of Things",
        "Management and Governance",
        "Media",
        "Migration",
        "Mobile",
        "Networking",
        "Security",
        "Storage",
        "Web",
        "Other",
    },
}


def validate(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    for col in NOT_NULL:
        n = int(df[col].isna().sum())
        if n:
            errors.append(f"{col}: {n} null values")

    for col, allowed in ALLOWED.items():
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            errors.append(f"{col}: values outside the allowed set: {sorted(bad)}")

    if (df["ChargePeriodEnd"] <= df["ChargePeriodStart"]).any():
        errors.append("ChargePeriodEnd must be after ChargePeriodStart")
    outside = (df["ChargePeriodStart"] < df["BillingPeriodStart"]) | (
        df["ChargePeriodEnd"] > df["BillingPeriodEnd"]
    )
    if outside.any():
        errors.append(f"{int(outside.sum())} charge periods fall outside their billing period")

    no_commitment = df["CommitmentDiscountId"].isna()
    if df.loc[no_commitment, "CommitmentDiscountStatus"].notna().any():
        errors.append("CommitmentDiscountStatus set on rows without CommitmentDiscountId")
    non_usage_status = (df["ChargeCategory"] != "Usage") & df["CommitmentDiscountStatus"].notna()
    if non_usage_status.any():
        errors.append("CommitmentDiscountStatus must be null when ChargeCategory is not Usage")
    unused = df["CommitmentDiscountStatus"] == "Unused"
    if df.loc[unused, "ConsumedQuantity"].notna().any():
        errors.append("ConsumedQuantity must be null for unused commitment rows")

    purchase_pricing = (df["ChargeCategory"] == "Purchase") & df["PricingCategory"].notna()
    if purchase_pricing.any():
        errors.append("PricingCategory must be null on Purchase rows in this dataset")

    no_credits = df["ChargeCategory"] != "Credit"
    for col in ["BilledCost", "EffectiveCost", "ListCost", "ContractedCost"]:
        if (df.loc[no_credits, col] < 0).any():
            errors.append(f"{col}: negative values on non-credit rows")
    return errors
