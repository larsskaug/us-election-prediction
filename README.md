# us-election-prediction

Bayesian prediction of U.S. congressional district-level election results integrating demographic features with prediction market signals from Polymarket.

## Motivation

Traditional election forecasting relies on either demographic/polling models or prediction market prices, but rarely combines both in a principled way. Demographic models (like those using Census ACS data) capture structural factors—race, income, education, age—that shape district-level partisanship, but they can miss fast-moving shifts in the political environment. Prediction markets like Polymarket aggregate real-time beliefs into prices, but those prices reflect a noisy mix of conviction bets, financial hedges, and capital-driven whale movements—not a clean probabilistic signal.

This project uses a Bayesian hierarchical model in **PyMC** to fuse these two information sources:

- **Demographic priors** derived from Census Bureau American Community Survey district profiles (informed by interpretable rule-based patterns from SR4-Fit-style analysis)
- **Market-informed likelihoods** constructed from Polymarket contract prices, with explicit modeling of noise, capital concentration, and the "prediction laundering" problem where heterogeneous motives are flattened into a single price

The Bayesian framework lets us maintain calibrated uncertainty rather than collapsing everything into a point estimate, and makes it possible to reason about *where* the market signal is trustworthy versus where demographic fundamentals should dominate.

## Project Structure

```
us-election-prediction/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── census.py          # ACS demographic data ingestion & cleaning
│   │   ├── elections.py        # MIT Election Lab results parsing
│   │   ├── polymarket.py       # Polymarket contract price scraping/API
│   │   └── merge.py            # Join demographics, results, and market data by district
│   ├── features/
│   │   ├── __init__.py
│   │   ├── demographic.py      # Feature engineering from ACS variables
│   │   └── market.py           # Market signal extraction (price, volume, concentration)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── priors.py           # Demographic-informed prior construction
│   │   ├── likelihood.py       # Market-signal likelihood with noise model
│   │   ├── hierarchical.py     # Full hierarchical PyMC model
│   │   └── diagnostics.py      # Convergence checks, posterior predictive checks
│   ├── interpret/
│   │   ├── __init__.py
│   │   ├── rules.py            # Rule extraction for interpretability (SR4-Fit inspired)
│   │   └── market_quality.py   # Capital concentration & signal reliability metrics
│   └── evaluate/
│       ├── __init__.py
│       ├── calibration.py      # Calibration plots, Brier scores
│       └── backtest.py         # Historical backtesting harness
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_market_signal_analysis.ipynb
│   └── 04_results.ipynb
├── tests/
│   └── ...
└── data/
    └── raw/                    # Gitignored; see data download instructions below
```

## Model Overview

The core model is a Bayesian hierarchical logistic regression:

```
For district d in state s:

  # Demographic prior on district lean
  α_s ~ Normal(μ_national, σ_state)          # state-level intercept
  β ~ Normal(0, σ_β)                          # demographic coefficients
  θ_demo_d = α_s + X_d · β                    # demographic log-odds

  # Market signal (observed with noise)
  θ_market_d = logit(p_market_d)              # raw market log-odds
  σ_market_d ~ f(volume_d, concentration_d)   # noise scales with capital concentration

  # Fusion
  θ_d ~ Normal(θ_demo_d, σ_demo)             # structural prior
  p_market_d ~ Normal(σ(θ_d), σ_market_d)    # market price as noisy observation

  # Outcome
  y_d ~ Bernoulli(σ(θ_d))                    # election result
```

Key modeling choices:
- **Hierarchical state effects** partially pool district estimates toward state and national means
- **Heteroscedastic market noise** — districts with thin markets or high whale concentration get wider market variance (directly addressing the "architectural masking" problem)
- **Demographic features** include registered party percentages, age/race/education/income distributions from ACS
- **Interpretable summaries** — posterior rule extraction identifies which demographic + market conditions most shift predictions

## Setup

### Prerequisites

- Python ≥ 3.11
- Git

### Installation

```bash
git clone https://github.com/<your-username>/us-election-prediction.git
cd us-election-prediction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Dependencies (managed in pyproject.toml)

Core: `pymc >= 5.10`, `arviz`, `pandas`, `numpy`, `scikit-learn`, `requests`
Dev: `pytest`, `jupyter`, `matplotlib`, `seaborn`, `black`, `ruff`

### Data

1. **ACS Demographics**: Download district-level tables from [data.census.gov](https://data.census.gov/) (Table S0601, 1-year estimates). Place CSVs in `data/raw/acs/`.
2. **Election Results**: Download from [MIT Election Data Lab](https://doi.org/10.7910/DVN/IG0UN2). Place in `data/raw/elections/`.
3. **Polymarket Data**: Run `python -m src.data.polymarket` to fetch contract histories via the Polymarket API (or supply cached JSON in `data/raw/polymarket/`).

## Usage

```bash
# Fetch and merge all data sources
python -m src.data.merge

# Fit the model
python -m src.model.hierarchical --election-year 2024

# Run diagnostics
python -m src.model.diagnostics --trace-path outputs/trace.nc

# Evaluate calibration on held-out cycles
python -m src.evaluate.backtest --years 2018 2020 2022
```

## Copilot Instructions

When contributing to this repository, follow these conventions:

- **Modeling**: All probabilistic models use PyMC v5+ syntax (`pm.Model()` context manager). Use ArviZ for all posterior analysis and plotting.
- **Data**: Pandas DataFrames are the interchange format. District identifiers use the pattern `{state_fips}-{district_number}` (e.g., `"06-12"` for California's 12th).
- **Type hints**: All function signatures should include type annotations.
- **Docstrings**: NumPy-style docstrings on all public functions.
- **Tests**: `pytest`. Model tests should use small synthetic data and check shape/convergence, not exact values.
- **Style**: `black` for formatting, `ruff` for linting.

## References

- Krishnan & Hougen (2026). "SR4-Fit: An Interpretable and Informative Classification Algorithm Applied to Prediction of U.S. House of Representatives Elections." arXiv:2602.06229.
- Rohanifar, Ahmed, & Sultana (2026). "Prediction Laundering: The Illusion of Neutrality, Transparency, and Governance in Polymarket." arXiv:2602.05181.
- Salvatier, Wiecki, & Fonnesbeck (2016). "Probabilistic programming in Python using PyMC3." PeerJ Computer Science.

## License

MIT