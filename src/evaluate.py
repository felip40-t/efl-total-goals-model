"""
evaluate.py

Score the walk-forward predictions from train.py against actual total goals,
per fold (test season) and pooled across all out-of-sample matches.
Since total goals is the sum of two independent Poissons, the actual total is
Poisson(lambda_total), so the count metrics below use that distribution.
Every total-goals metric is shown beside a naive constant baseline (each fold's
own training-mean total) so the model's lift is legible.
Two basic independence checks (home/away residual correlation and the total's
variance ratio) sanity-test the summed-Poisson assumption the model relies on.
No plots. Run as `python src/evaluate.py` from the repo root.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PRED_PATH = Path("data/processed/predictions.csv")
CLEAN_PATH = Path("data/raw/match_data_clean.csv")
METRICS_PATH = Path("data/processed/metrics.csv")

# Over/Under line to score the model against as a betting market.
OU_LINE = 2.5


def point_metrics(actual: np.ndarray, mu: np.ndarray) -> dict:
    """Accuracy of lambda_total as a point forecast of total goals."""
    error = mu - actual
    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "bias (mean pred - actual)": np.mean(error),
    }


def count_metrics(actual: np.ndarray, mu: np.ndarray) -> dict:
    """Proper metrics for a Poisson count model."""
    nll = -stats.poisson.logpmf(actual, mu).mean()
    # actual * log(actual / mu) -> 0 as actual -> 0; compute only on positive
    # rows so the log(0) case is never evaluated.
    log_term = np.zeros_like(mu, dtype=float)
    pos = actual > 0
    log_term[pos] = actual[pos] * np.log(actual[pos] / mu[pos])
    deviance = 2 * np.mean(log_term - (actual - mu))
    return {"Poisson NLL": nll, "mean Poisson deviance": deviance}


def side_metrics(pred: pd.DataFrame) -> dict:
    """Evaluate the model on its own target, goals_for, by stacking both sides."""
    actual = np.concatenate([pred["actual_home_goals"], pred["actual_away_goals"]])
    mu = np.concatenate([pred["lambda_home"], pred["lambda_away"]])
    return {**point_metrics(actual, mu), **count_metrics(actual, mu)}


def over_under_metrics(actual: np.ndarray, mu: np.ndarray) -> dict:
    """Score the implied Over/Under OU_LINE market. The majority-class accuracy is
    the no-model floor for `accuracy @0.5`: always calling the more common
    outcome, i.e. max(base rate, 1 - base rate)."""
    p_over = 1 - stats.poisson.cdf(np.floor(OU_LINE), mu)
    actual_over = (actual > OU_LINE).astype(float)
    base = actual_over.mean()
    return {
        f"Over{OU_LINE} base rate": base,
        f"Over{OU_LINE} mean predicted P": p_over.mean(),
        f"Over{OU_LINE} Brier": np.mean((p_over - actual_over) ** 2),
        f"Over{OU_LINE} accuracy @0.5": np.mean((p_over >= 0.5) == actual_over),
        f"Over{OU_LINE} majority-class accuracy": max(base, 1 - base),
    }


def independence_diagnostics(pred: pd.DataFrame) -> None:
    """Two basic checks of the home/away conditional-independence assumption the
    summed-rates model relies on (total = Poisson(lambda_home + lambda_away)).
    Both are conditional on the fitted rates, so they test the dependence left
    after the model has explained each side's scoring, not the raw-goal
    association that shared covariates create even under independence.

    - Residual correlation: correlate the Pearson residuals of the two sides,
      r = (goals - lambda) / sqrt(lambda). Independent Poisson => about 0.
    - Total variance ratio: mean((T - lamT)^2) / mean(lamT), which is 1 when the
      total is Poisson(lamT), i.e. the two independent sides add cleanly."""
    lh = pred["lambda_home"].to_numpy(float)
    la = pred["lambda_away"].to_numpy(float)
    rh = (pred["actual_home_goals"].to_numpy(float) - lh) / np.sqrt(lh)
    ra = (pred["actual_away_goals"].to_numpy(float) - la) / np.sqrt(la)
    corr = float(np.corrcoef(rh, ra)[0, 1])
    total = pred["actual_total_goals"].to_numpy(float)
    lam_total = pred["lambda_total"].to_numpy(float)
    var_ratio = float(np.mean((total - lam_total) ** 2) / np.mean(lam_total))
    print("Independence checks (home/away, conditional on fitted rates):")
    print(f"  residual correlation   {corr:>8.4f}   (0 = independent Poisson)")
    print(f"  total variance ratio   {var_ratio:>8.4f}   (1 = independent Poisson)")
    print()


def naive_baseline(clean_total: pd.Series, clean_seasons: pd.Series,
                   clean_leagues: pd.Series, test_season: str,
                   league: str = None) -> float:
    """Mean total goals over the seasons strictly before test_season, i.e. the
    constant a no-feature forecaster fit on that walk-forward fold's expanding
    training window would predict. Season labels sort chronologically, so the
    string comparison selects exactly the earlier seasons and never the test
    season or later. When `league` is given the mean is taken within that league
    only, so each division is benchmarked against its own historical scoring."""
    prior = clean_seasons < test_season
    if league is not None:
        prior = prior & (clean_leagues == league)
    return float(clean_total[prior].mean())


def total_goals_comparison(actual: np.ndarray, mu_model: np.ndarray,
                           mu_naive: np.ndarray) -> None:
    """Total-goals metrics for the model beside the naive constant baseline. The
    gap is what the features actually buy; lower is better for every row except
    bias, which is diagnostic (train-vs-holdout drift in the mean)."""
    model = {**point_metrics(actual, mu_model), **count_metrics(actual, mu_model)}
    naive = {**point_metrics(actual, mu_naive), **count_metrics(actual, mu_naive)}
    print("Total goals: model vs naive baseline")
    print(f"  {'':<28} {'model':>10} {'naive':>10}")
    for name in model:
        print(f"  {name:<28} {model[name]:>10.4f} {naive[name]:>10.4f}")
    print()
    return model, naive


def collect_rows(season: str, league: str, fold: pd.DataFrame, actual: np.ndarray,
                 mu: np.ndarray, mu_naive: np.ndarray) -> list:
    """Every metric row for one slice (a fold, a league, or the pooled set),
    tagged with `season` and `league`."""
    tg_model = {**point_metrics(actual, mu), **count_metrics(actual, mu)}
    tg_naive = {**point_metrics(actual, mu_naive), **count_metrics(actual, mu_naive)}
    side = side_metrics(fold)
    ou = over_under_metrics(actual, mu)
    return (
        [(season, league, "total_goals", "model", k, v) for k, v in tg_model.items()]
        + [(season, league, "total_goals", "naive", k, v) for k, v in tg_naive.items()]
        + [(season, league, "per_side_goals", "model", k, v) for k, v in side.items()]
        + [(season, league, f"over_under_{OU_LINE}", "model", k, v) for k, v in ou.items()]
    )


def write_metrics(rows: list) -> None:
    """Persist every metric as tidy long-format rows, one per
    (season, league, group, model, metric). `season` is a walk-forward fold (a
    test season) or 'pooled'; `league` is a division or 'all'. Long format keeps
    each row fully populated: the naive baseline only carries total-goals
    metrics, and per-side / Over-Under only apply to the model."""
    out = pd.DataFrame(
        rows, columns=["season", "league", "group", "model", "metric", "value"])
    out["value"] = out["value"].round(4)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_PATH, index=False)
    print(f"Wrote metrics to {METRICS_PATH}")


def main() -> None:
    pred = pd.read_csv(PRED_PATH)
    clean = pd.read_csv(CLEAN_PATH)
    clean_total = clean["home_goals"] + clean["away_goals"]
    clean_seasons = clean["season"]
    clean_leagues = clean["league_name"]

    folds = sorted(pred["season"].unique())
    naive_by_season = {s: naive_baseline(clean_total, clean_seasons, clean_leagues, s)
                       for s in folds}
    rows = []

    # --- per fold, all leagues ---
    print(f"Walk-forward: {len(folds)} folds, {len(pred)} out-of-sample matches\n")
    print(f"  {'fold (test season)':<20}{'matches':>9}{'MAE':>9}{'naive MAE':>11}{'NLL':>9}")
    for season in folds:
        fold = pred[pred["season"] == season]
        actual = fold["actual_total_goals"].to_numpy()
        mu = fold["lambda_total"].to_numpy()
        mu_naive = np.full_like(actual, naive_by_season[season], dtype=float)
        rows += collect_rows(season, "all", fold, actual, mu, mu_naive)
        pm, cm = point_metrics(actual, mu), count_metrics(actual, mu)
        pmn = point_metrics(actual, mu_naive)
        print(f"  {season:<20}{len(fold):>9}{pm['MAE']:>9.4f}"
              f"{pmn['MAE']:>11.4f}{cm['Poisson NLL']:>9.4f}")
    print()

    # --- pooled, all leagues: every match scored against its own fold's baseline ---
    actual = pred["actual_total_goals"].to_numpy()
    mu = pred["lambda_total"].to_numpy()
    mu_naive = pred["season"].map(naive_by_season).to_numpy()
    rows += collect_rows("pooled", "all", pred, actual, mu, mu_naive)

    print("Pooled (all folds, all leagues):")
    total_goals_comparison(actual, mu, mu_naive)
    for title, metrics in [
        ("Per-side goals (model target: goals_for)", side_metrics(pred)),
        (f"Over/Under {OU_LINE} market", over_under_metrics(actual, mu)),
    ]:
        print(title)
        for name, value in metrics.items():
            print(f"  {name:<32} {value:.4f}")
        print()

    independence_diagnostics(pred)

    # --- pooled per league: each match vs its own (fold, league) baseline ---
    print(f"  {'league':<20}{'matches':>9}{'MAE':>9}{'naive MAE':>11}{'NLL':>9}")
    for league in sorted(pred["league_name"].unique()):
        sub = pred[pred["league_name"] == league]
        actual = sub["actual_total_goals"].to_numpy()
        mu = sub["lambda_total"].to_numpy()
        nb = {s: naive_baseline(clean_total, clean_seasons, clean_leagues, s, league)
              for s in folds}
        mu_naive = sub["season"].map(nb).to_numpy()
        rows += collect_rows("pooled", league, sub, actual, mu, mu_naive)
        pm, cm = point_metrics(actual, mu), count_metrics(actual, mu)
        pmn = point_metrics(actual, mu_naive)
        print(f"  {league:<20}{len(sub):>9}{pm['MAE']:>9.4f}"
              f"{pmn['MAE']:>11.4f}{cm['Poisson NLL']:>9.4f}")
    print()

    write_metrics(rows)


if __name__ == "__main__":
    main()
