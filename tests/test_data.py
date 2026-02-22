"""Tests for the data pipeline: schema validation, ingestion, edge cases."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.schema import (
    REQUIRED_COLUMNS,
    build_district_id,
    validate_schema,
)
from src.features.engineering import build_feature_arrays, train_test_split_districts

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestBuildDistrictId:
    def test_basic(self):
        assert build_district_id("06", 12) == "06-12"

    def test_zero_for_at_large(self):
        assert build_district_id("02", 0) == "02-0"

    def test_leading_zero_preserved(self):
        assert build_district_id("01", 1) == "01-1"


class TestValidateSchema:
    def test_valid_df_passes(self, synthetic_df):
        errors = validate_schema(synthetic_df)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_column_reported(self, synthetic_df):
        df = synthetic_df.drop("market_hhi")
        errors = validate_schema(df)
        assert any("market_hhi" in e for e in errors)

    def test_out_of_range_probability(self, synthetic_df):
        df = synthetic_df.with_columns(pl.lit(1.5).alias("market_price_dem"))
        errors = validate_schema(df)
        assert any("market_price_dem" in e for e in errors)

    def test_bad_district_id_format(self, synthetic_df):
        df = synthetic_df.with_columns(pl.lit("bad-format-XX").alias("district_id"))
        errors = validate_schema(df)
        assert any("district_id" in e for e in errors)

    def test_all_required_columns_present(self, synthetic_df):
        for col in REQUIRED_COLUMNS:
            assert col in synthetic_df.columns, f"Fixture missing: {col}"


# ---------------------------------------------------------------------------
# Edge cases: uncontested districts
# ---------------------------------------------------------------------------


class TestUncontestedDistricts:
    def test_uncontested_rows_in_df(self, synthetic_df):
        uncontested = synthetic_df.filter(pl.col("is_uncontested"))
        assert uncontested.height > 0

    def test_feature_arrays_flag_uncontested(self, feature_arrays):
        assert feature_arrays.is_uncontested.sum() > 0

    def test_contested_count_correct(self, synthetic_df, feature_arrays):
        assert feature_arrays.X_demo.shape[0] == len(feature_arrays.district_ids)
        assert len(feature_arrays.is_uncontested) == synthetic_df.height


# ---------------------------------------------------------------------------
# Edge cases: zero-volume markets
# ---------------------------------------------------------------------------


class TestZeroVolumeMarkets:
    def test_zero_volume_rows_exist(self, synthetic_df):
        zero_vol = synthetic_df.filter(pl.col("market_volume_usd") == 0.0)
        assert zero_vol.height > 0

    def test_feature_arrays_handle_zero_volume(self, feature_arrays):
        # market_hhi should be clipped to 0 for zero-volume rows
        assert np.all(feature_arrays.market_hhi >= 0.0)
        assert np.all(feature_arrays.market_hhi <= 1.0)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestBuildFeatureArrays:
    def test_shapes(self, synthetic_df):
        fa = build_feature_arrays(synthetic_df)
        n = synthetic_df.height
        assert fa.X_demo.shape == (n, 5)
        assert fa.X_party.shape == (n, 3)
        assert fa.incumbent_idx.shape == (n,)
        assert fa.state_idx.shape == (n,)
        assert fa.market_price.shape == (n,)
        assert fa.market_hhi.shape == (n,)
        assert fa.y.shape == (n,)

    def test_market_price_clipped(self, feature_arrays):
        assert np.all(feature_arrays.market_price > 0.0)
        assert np.all(feature_arrays.market_price < 1.0)

    def test_incumbent_idx_range(self, feature_arrays):
        assert np.all(feature_arrays.incumbent_idx >= 0)
        assert np.all(feature_arrays.incumbent_idx <= 2)

    def test_state_idx_range(self, feature_arrays):
        assert np.all(feature_arrays.state_idx >= 0)
        assert np.all(feature_arrays.state_idx < feature_arrays.n_states)

    def test_state_labels_count(self, feature_arrays, synthetic_df):
        n_unique_states = synthetic_df["geo_state_fips"].n_unique()
        assert feature_arrays.n_states == n_unique_states

    def test_inference_mode_uses_existing_scaler(self, synthetic_df):
        fa_train = build_feature_arrays(synthetic_df, fit_scalers=True)
        fa_test = build_feature_arrays(
            synthetic_df,
            demo_scaler=fa_train.demo_scaler,
            party_scaler=fa_train.party_scaler,
            fit_scalers=False,
        )
        # Scalers are preserved
        assert fa_test.demo_scaler is fa_train.demo_scaler


class TestTrainTestSplit:
    def test_no_overlap(self, synthetic_df):
        train, test = train_test_split_districts(synthetic_df, test_size=0.2)
        train_ids = set(train["district_id"].to_list())
        test_ids = set(test["district_id"].to_list())
        assert train_ids.isdisjoint(test_ids)

    def test_sizes(self, synthetic_df):
        train, test = train_test_split_districts(synthetic_df, test_size=0.2)
        assert train.height + test.height == synthetic_df.height
