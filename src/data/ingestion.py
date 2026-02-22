"""Data ingestion: load and merge ACS demographic data with Polymarket prices.

All data is wrangled with polars/duckdb.  The output is a single polars
DataFrame with one row per (district, election_cycle), conforming to the
schema defined in :mod:`src.data.schema`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

from src.data.schema import build_district_id, validate_schema


def load_acs_demographics(path: str | Path) -> pl.DataFrame:
    """Load ACS (American Community Survey) demographic data from a CSV or Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the ACS data file.  Supported formats: ``.csv``, ``.parquet``.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns prefixed ``demo_*`` plus ``geo_state_fips``,
        ``geo_district_num``, ``geo_region``, and ``district_id``.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    # Ensure district_id is present
    if "district_id" not in df.columns:
        df = df.with_columns(
            pl.struct(["geo_state_fips", "geo_district_num"])
            .map_elements(
                lambda s: build_district_id(s["geo_state_fips"], s["geo_district_num"]),
                return_dtype=pl.String,
            )
            .alias("district_id")
        )
    return df


def load_polymarket_prices(path: str | Path) -> pl.DataFrame:
    """Load Polymarket prediction-market prices from a CSV or Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the Polymarket data file.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``district_id``, ``election_cycle``,
        ``market_price_dem``, ``market_volume_usd``, ``market_hhi``.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    # Clip market price to valid probability range
    df = df.with_columns(pl.col("market_price_dem").clip(0.0, 1.0))
    return df


def load_election_results(path: str | Path) -> pl.DataFrame:
    """Load historical election results from a CSV or Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the results data file.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``district_id``, ``election_cycle``,
        ``result_dem_win``, ``result_dem_voteshare``, ``incumbent_party``,
        ``is_uncontested``.

    Notes
    -----
    Third-party candidates are excluded from vote-share calculations
    (they constitute < 0.3 % of historical outcomes).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    # Exclude third-party rows if the column exists
    if "candidate_party" in df.columns:
        df = df.filter(pl.col("candidate_party").is_in(["D", "R"]))

    return df


def merge_datasets(
    acs: pl.DataFrame,
    market: pl.DataFrame,
    results: pl.DataFrame,
    party_registration: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Merge ACS, Polymarket, election results, and optional party-registration data.

    The merge is performed via DuckDB for efficiency on large datasets.

    Parameters
    ----------
    acs : pl.DataFrame
        ACS demographic data (see :func:`load_acs_demographics`).
    market : pl.DataFrame
        Polymarket price data (see :func:`load_polymarket_prices`).
    results : pl.DataFrame
        Election results (see :func:`load_election_results`).
    party_registration : pl.DataFrame, optional
        Party-registration percentages with columns ``district_id``,
        ``election_cycle``, ``party_pct_dem``, ``party_pct_rep``,
        ``party_pct_ind``.

    Returns
    -------
    pl.DataFrame
        Fully merged dataset with one row per (district, cycle) and all
        columns required by :data:`src.data.schema.REQUIRED_COLUMNS`.

    Raises
    ------
    ValueError
        If the merged dataset fails schema validation.
    """
    con = duckdb.connect()
    con.register("acs_tbl", acs)
    con.register("market_tbl", market)
    con.register("results_tbl", results)

    if party_registration is not None:
        con.register("party_tbl", party_registration)
        party_join = """
            LEFT JOIN party_tbl AS p
                ON r.district_id = p.district_id
               AND r.election_cycle = p.election_cycle
        """
        party_cols = "p.party_pct_dem, p.party_pct_rep, p.party_pct_ind"
    else:
        party_join = ""
        party_cols = (
            "CAST(NULL AS DOUBLE) AS party_pct_dem,"
            " CAST(NULL AS DOUBLE) AS party_pct_rep,"
            " CAST(NULL AS DOUBLE) AS party_pct_ind"
        )

    query = f"""
        SELECT
            r.district_id,
            r.election_cycle,
            a.geo_state_fips,
            a.geo_district_num,
            a.geo_region,
            a.demo_pct_white,
            a.demo_pct_black,
            a.demo_pct_hispanic,
            a.demo_pct_college,
            a.demo_median_income,
            {party_cols},
            m.market_price_dem,
            m.market_volume_usd,
            m.market_hhi,
            r.result_dem_win,
            r.result_dem_voteshare,
            r.incumbent_party,
            r.is_uncontested
        FROM results_tbl AS r
        LEFT JOIN acs_tbl AS a
            ON r.district_id = a.district_id
        LEFT JOIN market_tbl AS m
            ON r.district_id = m.district_id
           AND r.election_cycle = m.election_cycle
        {party_join}
    """

    merged = con.execute(query).pl()
    con.close()

    errors = validate_schema(merged)
    if errors:
        raise ValueError(
            "Merged dataset failed schema validation:\n" + "\n".join(errors)
        )

    return merged


def load_merged_dataset(path: str | Path) -> pl.DataFrame:
    """Load a pre-merged dataset directly from a Parquet or CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the merged dataset file.

    Returns
    -------
    pl.DataFrame
        Merged dataset conforming to :data:`src.data.schema.REQUIRED_COLUMNS`.

    Raises
    ------
    ValueError
        If the dataset fails schema validation.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    errors = validate_schema(df)
    if errors:
        raise ValueError("Dataset failed schema validation:\n" + "\n".join(errors))

    return df
