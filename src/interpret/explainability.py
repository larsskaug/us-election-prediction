"""Posterior analysis and explainability helpers.

Uses ArviZ for trace diagnostics and posterior summaries.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from src.features.engineering import DEMO_FEATURES, PARTY_FEATURES


def summarize_posterior(
    trace: az.InferenceData,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return a tidy ArviZ summary DataFrame for selected variables.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples from :func:`src.model.hierarchical.sample_model`.
    var_names : list[str], optional
        Variable names to include.  If ``None``, summarises all scalar/vector
        variables except large deterministics.

    Returns
    -------
    pd.DataFrame
        ArviZ summary with columns ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
        ``r_hat``, ``ess_bulk``.
    """
    if var_names is None:
        var_names = [
            "alpha",
            "alpha_state",
            "sigma_state",
            "beta_incumbent",
            "beta_demo",
            "beta_party",
            "sigma0",
            "kappa",
        ]

    return az.summary(trace, var_names=var_names, round_to=4)


def feature_importance(trace: az.InferenceData) -> pd.DataFrame:
    """Compute feature importance as mean absolute posterior coefficient.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature`` and ``mean_abs_coef``, sorted
        descending by importance.
    """
    rows: list[dict] = []

    posterior = trace.posterior

    if "beta_demo" in posterior:
        beta_demo = posterior["beta_demo"].values  # (chain, draw, feature)
        mean_abs = np.abs(beta_demo).mean(axis=(0, 1))
        for name, val in zip(DEMO_FEATURES, mean_abs):
            rows.append({"feature": name, "mean_abs_coef": float(val)})

    if "beta_party" in posterior:
        beta_party = posterior["beta_party"].values
        mean_abs = np.abs(beta_party).mean(axis=(0, 1))
        for name, val in zip(PARTY_FEATURES, mean_abs):
            rows.append({"feature": name, "mean_abs_coef": float(val)})

    if "beta_incumbent" in posterior:
        beta_inc = posterior["beta_incumbent"].values
        mean_abs = np.abs(beta_inc).mean(axis=(0, 1))
        for name, val in zip(["incumbent_D", "incumbent_R", "incumbent_O"], mean_abs):
            rows.append({"feature": name, "mean_abs_coef": float(val)})

    df = pd.DataFrame(rows).sort_values("mean_abs_coef", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


def district_win_probabilities(
    trace: az.InferenceData,
    district_ids: list[str],
) -> pd.DataFrame:
    """Extract posterior win probabilities for each contested district.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples (must include the ``p_dem`` deterministic).
    district_ids : list[str]
        Ordered list of contested district IDs matching the ``district``
        coordinate in the trace.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``district_id``, ``p_dem_mean``,
        ``p_dem_hdi_low``, ``p_dem_hdi_high``.
    """
    posterior = trace.posterior
    if "p_dem" not in posterior:
        raise KeyError("'p_dem' not found in posterior. Check model build.")

    p_dem = posterior["p_dem"].values  # (chain, draw, district)
    p_mean = p_dem.mean(axis=(0, 1))
    hdi = az.hdi(trace, var_names=["p_dem"])["p_dem"].values  # (district, 2)

    return pd.DataFrame(
        {
            "district_id": district_ids,
            "p_dem_mean": p_mean,
            "p_dem_hdi_low": hdi[:, 0],
            "p_dem_hdi_high": hdi[:, 1],
        }
    )
