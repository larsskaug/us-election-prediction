"""Shared pytest fixtures for the test suite.

Synthetic datasets use 50-100 rows with known edge cases:
  - Missing districts
  - Uncontested races
  - Zero-volume markets
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.engineering import FeatureArrays, build_feature_arrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_district_df(
    n: int = 80,
    seed: int = 0,
    include_uncontested: bool = True,
    include_zero_volume: bool = True,
) -> pl.DataFrame:
    """Create a synthetic merged district dataset."""
    rng = np.random.default_rng(seed)

    # Generate state FIPS codes (a handful of states)
    state_fips_pool = ["06", "12", "48", "36", "17", "39"]
    state_fips = [state_fips_pool[i % len(state_fips_pool)] for i in range(n)]

    district_nums = list(range(1, n + 1))
    district_ids = [f"{s}-{d}" for s, d in zip(state_fips, district_nums)]

    # Demographic features
    pct_white = rng.uniform(0.2, 0.9, n)
    pct_black = rng.uniform(0.02, 0.4, n)
    pct_hispanic = rng.uniform(0.05, 0.5, n)
    pct_college = rng.uniform(0.15, 0.65, n)
    median_income = rng.uniform(35000.0, 120000.0, n)

    # Party registration
    party_dem = rng.uniform(0.25, 0.55, n)
    party_rep = 1.0 - party_dem - rng.uniform(0.05, 0.15, n)
    party_rep = np.clip(party_rep, 0.0, 1.0)
    party_ind = np.clip(1.0 - party_dem - party_rep, 0.0, 1.0)

    # Market features
    market_price = rng.uniform(0.1, 0.9, n)
    market_volume = rng.uniform(1000.0, 500000.0, n)
    market_hhi = rng.uniform(0.0, 0.8, n)

    # Incumbent party
    incumbent_choices = ["D", "R", "O"]
    incumbent_party = [incumbent_choices[i % 3] for i in range(n)]

    # Outcomes (correlated with market price for realism)
    y = (rng.uniform(0, 1, n) < market_price).astype(int)

    # Vote share (two-party)
    voteshare = np.clip(market_price + rng.normal(0, 0.05, n), 0.01, 0.99)

    # Uncontested: last 5 rows
    is_uncontested = np.zeros(n, dtype=bool)
    if include_uncontested and n >= 5:
        is_uncontested[-5:] = True
        y[-5:] = 1  # known winner

    # Zero-volume market: rows 10-12
    if include_zero_volume and n > 12:
        market_volume[10:13] = 0.0
        market_hhi[10:13] = 0.0

    return pl.DataFrame(
        {
            "district_id": district_ids,
            "election_cycle": [2024] * n,
            "geo_state_fips": state_fips,
            "geo_district_num": pl.Series(district_nums, dtype=pl.Int32),
            "geo_region": [
                "South" if s in ("12", "48") else "Northeast" for s in state_fips
            ],
            "demo_pct_white": pct_white,
            "demo_pct_black": pct_black,
            "demo_pct_hispanic": pct_hispanic,
            "demo_pct_college": pct_college,
            "demo_median_income": median_income,
            "party_pct_dem": party_dem,
            "party_pct_rep": party_rep,
            "party_pct_ind": party_ind,
            "market_price_dem": market_price,
            "market_volume_usd": market_volume,
            "market_hhi": market_hhi,
            "result_dem_win": pl.Series(y.tolist(), dtype=pl.Int32),
            "result_dem_voteshare": voteshare,
            "incumbent_party": incumbent_party,
            "is_uncontested": pl.Series(is_uncontested.tolist()),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    """Standard 80-row synthetic dataset."""
    return _make_district_df(n=80, seed=42)


@pytest.fixture
def synthetic_df_small() -> pl.DataFrame:
    """Small 50-row synthetic dataset for fast tests."""
    return _make_district_df(n=50, seed=7)


@pytest.fixture
def feature_arrays(synthetic_df: pl.DataFrame) -> FeatureArrays:
    """FeatureArrays built from the standard 80-row synthetic dataset."""
    return build_feature_arrays(synthetic_df, fit_scalers=True)


@pytest.fixture
def feature_arrays_small(synthetic_df_small: pl.DataFrame) -> FeatureArrays:
    """FeatureArrays built from the small 50-row synthetic dataset."""
    return build_feature_arrays(synthetic_df_small, fit_scalers=True)
