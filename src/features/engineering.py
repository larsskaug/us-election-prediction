"""Feature engineering: preprocessing and train/test split utilities.

Uses scikit-learn only for scaling and splits — never for the core model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Demographic feature columns used as predictors
# ---------------------------------------------------------------------------

DEMO_FEATURES: list[str] = [
    "demo_pct_white",
    "demo_pct_black",
    "demo_pct_hispanic",
    "demo_pct_college",
    "demo_median_income",
]

PARTY_FEATURES: list[str] = [
    "party_pct_dem",
    "party_pct_rep",
    "party_pct_ind",
]

# Separate from demographics per domain rules
INCUMBENT_COL = "incumbent_party"


@dataclass
class FeatureArrays:
    """Container for model-ready numpy arrays.

    Attributes
    ----------
    X_demo : np.ndarray, shape (n, d_demo)
        Scaled demographic features.
    X_party : np.ndarray, shape (n, d_party)
        Scaled party-registration features.
    incumbent_idx : np.ndarray, shape (n,), dtype int
        Integer encoding of incumbent party: 0 = Dem, 1 = Rep, 2 = Open.
    state_idx : np.ndarray, shape (n,), dtype int
        Integer index into the list of unique states.
    market_price : np.ndarray, shape (n,)
        Polymarket implied probability for Dem win, clipped to (0, 1).
    market_hhi : np.ndarray, shape (n,)
        Herfindahl index for market-noise inflation (0-1).
    y : np.ndarray, shape (n,), dtype int
        Binary outcome: 1 = Dem win, 0 = Rep win.
    n_states : int
        Number of unique states.
    state_labels : list[str]
        Ordered list of state FIPS codes corresponding to ``state_idx``.
    district_ids : list[str]
        Ordered district IDs matching row order.
    is_uncontested : np.ndarray, shape (n,), dtype bool
        Mask for uncontested districts (should be handled separately).
    """

    X_demo: np.ndarray
    X_party: np.ndarray
    incumbent_idx: np.ndarray
    state_idx: np.ndarray
    market_price: np.ndarray
    market_hhi: np.ndarray
    y: np.ndarray
    n_states: int
    state_labels: list[str]
    district_ids: list[str]
    is_uncontested: np.ndarray
    demo_scaler: StandardScaler = field(repr=False)
    party_scaler: StandardScaler = field(repr=False)


_INCUMBENT_ENCODING: dict[str, int] = {"D": 0, "R": 1, "O": 2}


def encode_incumbent(series: pl.Series) -> np.ndarray:
    """Encode incumbent party strings to integer indices.

    Parameters
    ----------
    series : pl.Series
        String series with values ``"D"``, ``"R"``, or ``"O"`` (open seat).

    Returns
    -------
    np.ndarray
        Integer array with 0 = Dem, 1 = Rep, 2 = Open.
    """
    return np.array(
        [_INCUMBENT_ENCODING.get(v, 2) for v in series.to_list()], dtype=np.int32
    )


def build_feature_arrays(
    df: pl.DataFrame,
    demo_scaler: Optional[StandardScaler] = None,
    party_scaler: Optional[StandardScaler] = None,
    fit_scalers: bool = True,
) -> FeatureArrays:
    """Convert a merged polars DataFrame into model-ready numpy arrays.

    Uncontested districts are *included* in the arrays but flagged via
    ``is_uncontested`` so the model can handle them separately.

    Parameters
    ----------
    df : pl.DataFrame
        Merged dataset as returned by :func:`src.data.ingestion.merge_datasets`.
    demo_scaler : StandardScaler, optional
        Pre-fitted scaler for demographic features.  If ``None`` and
        ``fit_scalers=True``, a new scaler is fitted.
    party_scaler : StandardScaler, optional
        Pre-fitted scaler for party-registration features.
    fit_scalers : bool
        Whether to fit the scalers on *df*.  Set to ``False`` at inference time
        and provide pre-fitted scalers.

    Returns
    -------
    FeatureArrays
        Named container with all arrays needed by the PyMC model.
    """
    # Fill any missing party registration with 0 (will be handled by model)
    for col in PARTY_FEATURES:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Demographic features
    X_demo_raw = df.select(DEMO_FEATURES).fill_null(0.0).to_numpy()
    if demo_scaler is None:
        demo_scaler = StandardScaler()
    if fit_scalers:
        X_demo = demo_scaler.fit_transform(X_demo_raw)
    else:
        X_demo = demo_scaler.transform(X_demo_raw)

    # Party registration features
    X_party_raw = df.select(PARTY_FEATURES).fill_null(0.0).to_numpy()
    if party_scaler is None:
        party_scaler = StandardScaler()
    if fit_scalers:
        X_party = party_scaler.fit_transform(X_party_raw)
    else:
        X_party = party_scaler.transform(X_party_raw)

    # Incumbent party encoding
    incumbent_idx = encode_incumbent(df[INCUMBENT_COL])

    # State index
    states = df["geo_state_fips"].to_list()
    state_labels = sorted(set(states))
    state_map = {s: i for i, s in enumerate(state_labels)}
    state_idx = np.array([state_map[s] for s in states], dtype=np.int32)

    # Market features
    market_price = df["market_price_dem"].clip(1e-6, 1 - 1e-6).to_numpy()
    market_hhi = df["market_hhi"].fill_null(0.0).clip(0.0, 1.0).to_numpy()

    # Target
    y = df["result_dem_win"].to_numpy().astype(np.int32)

    # Uncontested flag
    is_uncontested = df["is_uncontested"].to_numpy().astype(bool)

    return FeatureArrays(
        X_demo=X_demo,
        X_party=X_party,
        incumbent_idx=incumbent_idx,
        state_idx=state_idx,
        market_price=market_price,
        market_hhi=market_hhi,
        y=y,
        n_states=len(state_labels),
        state_labels=state_labels,
        district_ids=df["district_id"].to_list(),
        is_uncontested=is_uncontested,
        demo_scaler=demo_scaler,
        party_scaler=party_scaler,
    )


def train_test_split_districts(
    df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the dataset into train and test sets at the district level.

    Parameters
    ----------
    df : pl.DataFrame
        Merged dataset.
    test_size : float
        Fraction of districts to hold out for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (train_df, test_df)
    """
    district_ids = df["district_id"].unique().to_list()
    train_ids, test_ids = train_test_split(
        district_ids, test_size=test_size, random_state=random_state
    )
    train_set = set(train_ids)
    test_set = set(test_ids)
    train_df = df.filter(pl.col("district_id").is_in(train_set))
    test_df = df.filter(pl.col("district_id").is_in(test_set))
    return train_df, test_df
