# English Football: Total Goals Model

Predicts the **total goals** in an English football match (Premier League,
Championship, League One; seasons 2018/19 to date) with a Poisson team-strength
GLM. 

## Findings

**The full summary of findings is in [`FINDINGS.md`](FINDINGS.md)**: modelling
rationale, data-quality handling, leakage treatment, validation design, results,
and limitations.

Each match is split into two team-rows, a single Poisson GLM predicts each team's
goals from pre-match form and opponent strength, and the two rates are summed to
a `Poisson(lambda_total)` match total. Validated by expanding-window
walk-forward (7 folds, 9,756 out-of-sample matches over 2019/20–2025/26). Pooled
headline result:

| Total goals | Model | Naive baseline |
|---|---|---|
| MAE | 1.2864 | 1.2990 |
| RMSE | 1.6005 | 1.6101 |
| Poisson NLL | 1.8486 | 1.8541 |

The model beats a constant-mean forecast by ~1% and does so in **every fold**,
the expected, robust result for a low-count, high-variance target. Full per-fold,
per-league, and pooled metrics are written to `data/processed/metrics.csv`; see
[`FINDINGS.md`](FINDINGS.md) for the reasoning.

## Setup and run

Python 3.12. The `data/` folder is included in the repo (`data/raw/` inputs and
`data/processed/` generated outputs), so the results can be inspected directly
and the pipeline re-run end to end. Install into a `.venv` (which the `Makefile`
expects):

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy statsmodels scipy matplotlib
```

Run the pipeline in order (`make <name>` runs `src/<name>.py`):

```bash
make clean       # data/raw/match_data.csv        -> data/raw/match_data_clean.csv
make features    # data/raw/match_data_clean.csv  -> data/processed/features.csv
make train       # data/processed/features.csv    -> data/processed/predictions.csv
make evaluate    # prints metrics, writes         -> data/processed/metrics.csv
make eda         # optional EDA plots -> plots/
```
