"""Data schema definitions and validation for the district-election dataset."""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Required column groups (prefix → description)
# ---------------------------------------------------------------------------

DEMO_PREFIX = "demo_"
PARTY_PREFIX = "party_"
MARKET_PREFIX = "market_"
RESULT_PREFIX = "result_"
GEO_PREFIX = "geo_"

# Minimum required columns in the merged dataset
REQUIRED_COLUMNS: list[str] = [
    "district_id",  # "{state_fips}-{district_number}"
    "election_cycle",  # e.g. 2024
    "geo_state_fips",
    "geo_district_num",
    "geo_region",
    "demo_pct_white",
    "demo_pct_black",
    "demo_pct_hispanic",
    "demo_pct_college",
    "demo_median_income",
    "party_pct_dem",
    "party_pct_rep",
    "party_pct_ind",
    "market_price_dem",  # Polymarket implied probability for Dem win (0-1)
    "market_volume_usd",  # Total USD volume traded
    "market_hhi",  # Herfindahl index of wallet positions
    "result_dem_win",  # 1 = Dem won, 0 = Rep won (binary)
    "result_dem_voteshare",  # Dem two-party vote share (0-1)
    "incumbent_party",  # "D", "R", or "O" (open seat)
    "is_uncontested",  # bool: only one major-party candidate
]

# Expected polars dtype for each required column
COLUMN_DTYPES: dict[str, type] = {
    "district_id": pl.String,
    "election_cycle": pl.Int64,
    "geo_state_fips": pl.String,
    "geo_district_num": pl.Int32,
    "geo_region": pl.String,
    "demo_pct_white": pl.Float64,
    "demo_pct_black": pl.Float64,
    "demo_pct_hispanic": pl.Float64,
    "demo_pct_college": pl.Float64,
    "demo_median_income": pl.Float64,
    "party_pct_dem": pl.Float64,
    "party_pct_rep": pl.Float64,
    "party_pct_ind": pl.Float64,
    "market_price_dem": pl.Float64,
    "market_volume_usd": pl.Float64,
    "market_hhi": pl.Float64,
    "result_dem_win": pl.Int32,
    "result_dem_voteshare": pl.Float64,
    "incumbent_party": pl.String,
    "is_uncontested": pl.Boolean,
}


def validate_schema(df: pl.DataFrame) -> list[str]:
    """Check that *df* has all required columns with acceptable dtypes.

    Parameters
    ----------
    df : pl.DataFrame
        Merged district-election dataset.

    Returns
    -------
    list[str]
        List of validation error messages.  Empty list means the schema is valid.
    """
    errors: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    for col, expected_dtype in COLUMN_DTYPES.items():
        if col not in df.columns:
            continue  # already reported above
        actual = df[col].dtype
        if actual != expected_dtype:
            errors.append(f"Column '{col}': expected {expected_dtype}, got {actual}")

    # Range checks
    float_pct_cols = [
        "demo_pct_white",
        "demo_pct_black",
        "demo_pct_hispanic",
        "demo_pct_college",
        "party_pct_dem",
        "party_pct_rep",
        "party_pct_ind",
        "market_price_dem",
        "result_dem_voteshare",
        "market_hhi",
    ]
    for col in float_pct_cols:
        if col not in df.columns:
            continue
        bad = df.filter((pl.col(col) < 0.0) | (pl.col(col) > 1.0)).height
        if bad > 0:
            errors.append(f"Column '{col}' has {bad} value(s) outside [0, 1]")

    # District ID format check
    if "district_id" in df.columns:
        bad_ids = df.filter(~pl.col("district_id").str.contains(r"^\d+-\d+$")).height
        if bad_ids > 0:
            errors.append(
                f"district_id has {bad_ids} value(s) not matching "
                r"'{state_fips}-{district_number}' format"
            )

    return errors


def build_district_id(state_fips: str, district_number: int) -> str:
    """Construct the canonical district identifier string.

    Parameters
    ----------
    state_fips : str
        Two-digit state FIPS code, e.g. ``"06"`` for California.
    district_number : int
        Congressional district number (1-based; 0 for at-large).

    Returns
    -------
    str
        District ID in ``"{state_fips}-{district_number}"`` format.
    """
    return f"{state_fips}-{district_number}"
