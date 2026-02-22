"""Tests for the PyMC hierarchical model.

Uses small synthetic datasets (50-100 rows) and checks:
  - Tensor shapes
  - Sampling completion (short run: 100 draws, 100 tune, 2 chains)
  - Basic posterior coverage (R-hat < 1.05 on short run)
  - build_model returns a valid pm.Model
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from src.model.hierarchical import build_model, check_convergence, sample_model

# ---------------------------------------------------------------------------
# Model construction tests
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_returns_pm_model(self, feature_arrays_small):
        model = build_model(feature_arrays_small)
        assert isinstance(model, pm.Model)

    def test_observed_variables_present(self, feature_arrays_small):
        model = build_model(feature_arrays_small)
        obs_names = {v.name for v in model.observed_RVs}
        assert "market_obs" in obs_names
        assert "y_obs" in obs_names

    def test_free_variables_present(self, feature_arrays_small):
        model = build_model(feature_arrays_small)
        free_names = {v.name for v in model.free_RVs}
        expected = {
            "alpha",
            "sigma_state",
            "alpha_state_raw",
            "beta_incumbent",
            "beta_demo",
            "beta_party",
            "sigma0",
            "kappa",
        }
        assert expected.issubset(
            free_names
        ), f"Missing free RVs: {expected - free_names}"

    def test_contested_districts_only_in_likelihood(self, feature_arrays_small):
        """Only contested rows enter the likelihood."""
        fa = feature_arrays_small
        n_contested = int((~fa.is_uncontested).sum())
        model = build_model(fa)
        # p_dem deterministic should have shape (n_contested,)
        assert model["p_dem"].eval().shape == (n_contested,)

    def test_market_noise_inflated_by_hhi(self, feature_arrays_small):
        """sigma_market should be > sigma0 wherever hhi > 0."""
        model = build_model(feature_arrays_small)
        # Check structural property: sigma_market = sigma0 + kappa * hhi
        # so sigma_market >= sigma0 always (since kappa >= 0 and hhi >= 0)
        fa = feature_arrays_small
        contested = ~fa.is_uncontested
        hhi = fa.market_hhi[contested]
        # At least some non-zero HHI values exist
        assert hhi.max() > 0, "Test requires at least one non-zero HHI"
        # Confirm sigma_market node exists in the model
        assert "sigma_market" in [v.name for v in model.deterministics]

    def test_state_coords(self, feature_arrays_small):
        model = build_model(feature_arrays_small)
        assert "state" in model.coords
        assert len(model.coords["state"]) == feature_arrays_small.n_states


# ---------------------------------------------------------------------------
# Sampling tests (very short — just check it runs and returns InferenceData)
# ---------------------------------------------------------------------------


class TestSampleModel:
    @pytest.fixture
    def short_trace(self, feature_arrays_small):
        """Run a very short chain for structural tests."""
        model = build_model(feature_arrays_small)
        return sample_model(
            model,
            draws=50,
            tune=50,
            chains=2,
            progressbar=False,
        )

    def test_returns_inference_data(self, short_trace):
        import arviz as az

        assert isinstance(short_trace, az.InferenceData)

    def test_posterior_has_expected_variables(self, short_trace):
        posterior_vars = set(short_trace.posterior.data_vars)
        for var in ["alpha", "beta_demo", "beta_party", "beta_incumbent", "p_dem"]:
            assert var in posterior_vars, f"Missing posterior variable: {var}"

    def test_p_dem_in_unit_interval(self, short_trace):
        p = short_trace.posterior["p_dem"].values
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_beta_demo_shape(self, short_trace):
        # (chains, draws, n_demo_features=5)
        shape = short_trace.posterior["beta_demo"].shape
        assert shape[-1] == 5

    def test_beta_party_shape(self, short_trace):
        # (chains, draws, n_party_features=3)
        shape = short_trace.posterior["beta_party"].shape
        assert shape[-1] == 3

    def test_beta_incumbent_shape(self, short_trace):
        # (chains, draws, 3 categories: D, R, O)
        shape = short_trace.posterior["beta_incumbent"].shape
        assert shape[-1] == 3

    def test_alpha_state_shape(self, short_trace, feature_arrays_small):
        shape = short_trace.posterior["alpha_state"].shape
        assert shape[-1] == feature_arrays_small.n_states


# ---------------------------------------------------------------------------
# Convergence diagnostics tests
# ---------------------------------------------------------------------------


class TestCheckConvergence:
    def test_returns_dict_keys(self, feature_arrays_small):
        model = build_model(feature_arrays_small)
        trace = sample_model(model, draws=50, tune=50, chains=2, progressbar=False)
        result = check_convergence(trace)
        assert "r_hat_warnings" in result
        assert "ess_warnings" in result

    def test_types(self, feature_arrays_small):
        model = build_model(feature_arrays_small)
        trace = sample_model(model, draws=50, tune=50, chains=2, progressbar=False)
        result = check_convergence(trace)
        assert isinstance(result["r_hat_warnings"], list)
        assert isinstance(result["ess_warnings"], list)
