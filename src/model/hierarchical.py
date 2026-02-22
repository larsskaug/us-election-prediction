"""Hierarchical Bayesian model for US House district election forecasting.

Uses PyMC 5+ with NUTS sampling.  The model fuses:

* ACS demographic predictors (scaled)
* Party-registration predictors (scaled)
* Incumbent-party fixed effects
* Polymarket price as a noisy observation (heteroscedastic noise inflated by HHI)
* State-level random effects (partial pooling)

Uncontested districts are excluded from the likelihood — their outcomes
are set to the known result directly without any uncertainty.
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import pymc as pm

from src.features.engineering import FeatureArrays


def build_model(fa: FeatureArrays) -> pm.Model:
    """Construct the hierarchical PyMC model.

    The latent Democratic win probability for each contested district follows a
    logistic regression with:

    * State-level intercepts (non-centred parameterisation)
    * Demographic linear predictor
    * Party-registration linear predictor
    * Incumbent-party fixed effect (separate from demographics)

    The Polymarket price is modelled as a noisy observation of the latent
    probability, with observation noise inflated by the HHI concentration
    metric:

        sigma_market_i = sigma0 + kappa * hhi_i

    Parameters
    ----------
    fa : FeatureArrays
        Preprocessed feature arrays produced by
        :func:`src.features.engineering.build_feature_arrays`.

    Returns
    -------
    pm.Model
        Compiled PyMC model (not yet sampled).

    Notes
    -----
    Only contested (non-uncontested) districts enter the likelihood.
    Uncontested districts have their probability clamped to 0.99 (known win)
    or 0.01 (known loss).
    """
    contested = ~fa.is_uncontested
    n_demo = fa.X_demo.shape[1]
    n_party = fa.X_party.shape[1]

    coords = {
        "state": fa.state_labels,
        "demo_feature": [f"demo_{i}" for i in range(n_demo)],
        "party_feature": [f"party_{i}" for i in range(n_party)],
        "incumbent_category": ["D", "R", "O"],
        "district": [d for d, c in zip(fa.district_ids, contested) if c],
    }

    with pm.Model(coords=coords) as model:
        # ------------------------------------------------------------------
        # Data containers (swappable for posterior predictive)
        # ------------------------------------------------------------------
        X_demo_data = pm.Data(
            "X_demo",
            fa.X_demo[contested],
            dims=("district", "demo_feature"),
        )
        X_party_data = pm.Data(
            "X_party",
            fa.X_party[contested],
            dims=("district", "party_feature"),
        )
        incumbent_data = pm.Data(
            "incumbent_idx",
            fa.incumbent_idx[contested],
            dims="district",
        )
        state_data = pm.Data(
            "state_idx",
            fa.state_idx[contested],
            dims="district",
        )
        market_price_data = pm.Data(
            "market_price",
            fa.market_price[contested],
            dims="district",
        )
        market_hhi_data = pm.Data(
            "market_hhi",
            fa.market_hhi[contested],
            dims="district",
        )
        y_data = pm.Data(
            "y",
            fa.y[contested].astype(float),
            dims="district",
        )

        # ------------------------------------------------------------------
        # Priors
        # ------------------------------------------------------------------
        # Global intercept
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # State-level random intercepts (non-centred)
        sigma_state = pm.HalfNormal("sigma_state", sigma=0.5)
        alpha_state_raw = pm.Normal("alpha_state_raw", mu=0.0, sigma=1.0, dims="state")
        alpha_state = pm.Deterministic(
            "alpha_state", alpha + alpha_state_raw * sigma_state, dims="state"
        )

        # Incumbent-party fixed effects (separate from demographics)
        beta_incumbent = pm.Normal(
            "beta_incumbent", mu=0.0, sigma=1.5, dims="incumbent_category"
        )

        # Demographic coefficients
        beta_demo = pm.Normal("beta_demo", mu=0.0, sigma=1.0, dims="demo_feature")

        # Party-registration coefficients
        beta_party = pm.Normal("beta_party", mu=0.0, sigma=1.0, dims="party_feature")

        # ------------------------------------------------------------------
        # Linear predictor
        # ------------------------------------------------------------------
        mu_logit = (
            alpha_state[state_data]
            + beta_incumbent[incumbent_data]
            + pm.math.dot(X_demo_data, beta_demo)
            + pm.math.dot(X_party_data, beta_party)
        )
        p_dem = pm.Deterministic("p_dem", pm.math.sigmoid(mu_logit), dims="district")

        # ------------------------------------------------------------------
        # Market-price likelihood (heteroscedastic noise)
        # sigma_market = sigma0 + kappa * hhi
        # ------------------------------------------------------------------
        sigma0 = pm.HalfNormal("sigma0", sigma=0.1)
        kappa = pm.HalfNormal("kappa", sigma=0.2)
        sigma_market = pm.Deterministic(
            "sigma_market",
            sigma0 + kappa * market_hhi_data,
            dims="district",
        )
        pm.Normal(
            "market_obs",
            mu=p_dem,
            sigma=sigma_market,
            observed=market_price_data,
            dims="district",
        )

        # ------------------------------------------------------------------
        # Election outcome likelihood
        # ------------------------------------------------------------------
        pm.Bernoulli("y_obs", p=p_dem, observed=y_data, dims="district")

    return model


def sample_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: Optional[int] = 42,
    progressbar: bool = True,
) -> az.InferenceData:
    """Sample the posterior using NUTS.

    Parameters
    ----------
    model : pm.Model
        PyMC model as returned by :func:`build_model`.
    draws : int
        Number of posterior samples per chain.
    tune : int
        Number of tuning steps.
    chains : int
        Number of MCMC chains.
    target_accept : float
        Target acceptance rate for NUTS (0.95 recommended for hierarchical models).
    random_seed : int, optional
        Random seed for reproducibility.
    progressbar : bool
        Show a progress bar during sampling.

    Returns
    -------
    az.InferenceData
        Posterior trace with ``posterior``, ``sample_stats``, and
        ``observed_data`` groups.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=progressbar,
        )
    return trace


def check_convergence(
    trace: az.InferenceData, r_hat_threshold: float = 1.01, ess_threshold: float = 400.0
) -> dict[str, list[str]]:
    """Run ArviZ convergence diagnostics.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior trace from :func:`sample_model`.
    r_hat_threshold : float
        Maximum acceptable R-hat value (default 1.01).
    ess_threshold : float
        Minimum acceptable bulk ESS (default 400).

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys ``"r_hat_warnings"`` and ``"ess_warnings"``,
        each containing a list of variable names that failed the check.
    """
    summary = az.summary(trace, round_to=4)
    warnings: dict[str, list[str]] = {"r_hat_warnings": [], "ess_warnings": []}

    if "r_hat" in summary.columns:
        bad_rhat = summary[summary["r_hat"] > r_hat_threshold].index.tolist()
        warnings["r_hat_warnings"] = bad_rhat

    if "ess_bulk" in summary.columns:
        bad_ess = summary[summary["ess_bulk"] < ess_threshold].index.tolist()
        warnings["ess_warnings"] = bad_ess

    return warnings


def posterior_predictive_check(
    model: pm.Model,
    trace: az.InferenceData,
    fa_new: Optional[FeatureArrays] = None,
) -> az.InferenceData:
    """Run posterior predictive checks, optionally on new data.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model.
    trace : az.InferenceData
        Posterior samples.
    fa_new : FeatureArrays, optional
        New feature arrays to swap in for out-of-sample predictions.
        If ``None``, uses the training data stored in the model.

    Returns
    -------
    az.InferenceData
        Trace extended with ``posterior_predictive`` group.
    """
    contested = ~fa_new.is_uncontested if fa_new is not None else None

    with model:
        if fa_new is not None and contested is not None:
            pm.set_data(
                {
                    "X_demo": fa_new.X_demo[contested],
                    "X_party": fa_new.X_party[contested],
                    "incumbent_idx": fa_new.incumbent_idx[contested],
                    "state_idx": fa_new.state_idx[contested],
                    "market_price": fa_new.market_price[contested],
                    "market_hhi": fa_new.market_hhi[contested],
                    "y": fa_new.y[contested].astype(float),
                }
            )
        ppc = pm.sample_posterior_predictive(trace)

    return ppc
