"""
train.py

Fit a Poisson team-strength GLM with expanding-window walk-forward validation
and predict match goal rates out of sample.

One model over both sides of every match: the response is goals_for (goals
scored by a team in a match) and is_home carries the home advantage. Each fold
trains on every season strictly before a test season and predicts that season;
predictions are recombined to match level (lambda_home + lambda_away) and stacked
across folds. Run as `python src/train.py` from the repo root.
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

import features as F

FEATURES_PATH = Path("data/processed/features.csv")
OUT_PATH = Path("data/processed/predictions.csv")


def load_features() -> tuple[pd.DataFrame, list[str]]:
    """Read the persisted feature matrix and the module's feature list."""
    df = pd.read_csv(FEATURES_PATH)
    return df, F.FEATURE_COLS


def build_design(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """FEATURE_COLS (already includes is_home) plus one-hot league_name."""
    league = pd.get_dummies(df["league_name"], prefix="league", drop_first=True)
    X = pd.concat([df[feature_cols], league], axis=1).astype(float)
    return sm.add_constant(X)


def walk_forward_folds(df: pd.DataFrame):
    """Expanding-window walk-forward folds. Seasons sort chronologically, so for
    every season from the second onward yield (train_mask, test_mask,
    test_season): the model trains on all strictly-earlier seasons and is scored
    on that season. No fold ever trains on its own test season or later."""
    seasons = sorted(df["season"].unique())
    for i in range(1, len(seasons)):
        test_season = seasons[i]
        train_mask = df["season"].isin(seasons[:i])
        test_mask = df["season"] == test_season
        yield train_mask, test_mask, test_season


def fit_poisson(y: pd.Series, X: pd.DataFrame):
    return sm.GLM(y, X, family=sm.families.Poisson()).fit()


def recombine(holdout: pd.DataFrame) -> pd.DataFrame:
    """Join the two team-rows of each match into one row with both lambdas and
    both sides' actual goals (goals_for on the home/away row respectively)."""
    home = holdout[holdout["venue"] == "home"][
        ["match_id", "date", "season", "league_name", "lam", "goals_for", "total_goals"]
    ].rename(columns={"lam": "lambda_home", "goals_for": "actual_home_goals",
                      "total_goals": "actual_total_goals"})
    away = holdout[holdout["venue"] == "away"][["match_id", "lam", "goals_for"]].rename(
        columns={"lam": "lambda_away", "goals_for": "actual_away_goals"}
    )
    matches = home.merge(away, on="match_id")
    matches["lambda_total"] = matches["lambda_home"] + matches["lambda_away"]
    return matches[
        ["match_id", "date", "season", "league_name",
         "lambda_home", "lambda_away", "lambda_total",
         "actual_home_goals", "actual_away_goals", "actual_total_goals"]
    ]


def main() -> None:
    df, feature_cols = load_features()
    X = build_design(df, feature_cols)
    y = df["goals_for"]

    fold_preds = []
    print(f"{'test season':<14}{'train seasons':>14}{'matches':>9}")
    for train_mask, test_mask, test_season in walk_forward_folds(df):
        model = fit_poisson(y[train_mask], X[train_mask])
        holdout = df[test_mask].copy()
        holdout["lam"] = model.predict(X[test_mask])
        matches = recombine(holdout)
        fold_preds.append(matches)
        n_train_seasons = df.loc[train_mask, "season"].nunique()
        print(f"{test_season:<14}{n_train_seasons:>14}{len(matches):>9}")

    predictions = pd.concat(fold_preds, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT_PATH, index=False)
    print(f"\nWalk-forward folds: {len(fold_preds)}  |  "
          f"out-of-sample matches: {len(predictions)}")
    print(f"Wrote predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()
