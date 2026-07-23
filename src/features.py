"""
features.py

Build a leakage-safe feature matrix for a Poisson team-strength GLM.
Output is two rows per match (one per team), response ``goals_for``.
Leakage safety holds because every rolling mean reads strictly earlier dates
(see ``_roll_mean``); cold-start gaps are filled with a constant prior.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/match_data_clean.csv")
OUT_PATH = Path("data/processed/features.csv")

# For earliest matches with no prior history.
COLD_START_PRIOR = {"goals_for": 1.35, "goals_against": 1.35}
FORM_WINDOW = 5
MIN_MATCHES = 5
BASE_STATS = ["goals_for", "goals_against", "xg_for", "xg_against"]
LEAK_COLS = ["goals_for", "goals_against", "xg_for", "xg_against", "total_goals"]
META_COLS = ["match_id", "date", "season", "league_name", "game_week",
             "team_id", "team", "opp_team_id", "opp_team", "venue"]
LABEL_COLS = ["goals_for", "goals_against", "total_goals"]
LEAGUE_RANK = {"Premier League": 3, "Championship": 2, "League One": 1}

# Wide to long column mapping for home and away teams
SIDE_COLS = {
    "home": {"home_id": "team_id", "home_team": "team",
             "home_goals": "goals_for", "away_goals": "goals_against",
             "home_xg": "xg_for", "away_xg": "xg_against"},
    "away": {"away_id": "team_id", "away_team": "team",
             "away_goals": "goals_for", "home_goals": "goals_against",
             "away_xg": "xg_for", "home_xg": "xg_against"},
}

def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build long-form DataFrame with one row per team per match.
    """
    common = ["match_id", "date", "league_name", "season", "game_week",
                "is_crowds"]

    sides = []
    for venue, mapping in SIDE_COLS.items():
        side_df = df[common + list(mapping.keys())].rename(columns=mapping)
        side_df["venue"] = venue
        sides.append(side_df)
    long = pd.concat(sides, ignore_index=True)
    return long.sort_values(["team_id", "date", "venue"]).reset_index(drop=True)

def _roll_mean(s: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling mean shifted by 1 to avoid data leakage.
    """
    # min_periods = window // 2 lets a partial early-season history contribute
    # while still requiring a couple of prior matches. The cold-start NaNs this
    # leaves behind are filled by fill_cold_start.
    return s.shift(1).rolling(window, min_periods=window // 2).mean()

def add_rolling(long: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean features to the long-form DataFrame.

    Form resets at each season boundary: the rolling means are grouped by
    (team_id, season), so a promoted or relegated side never carries its old
    division's scoring rate into the new one. The season-opening rows this
    leaves NaN are handled by the cold-start path (team_cold is likewise
    per-season, and fill_cold_start supplies the constant prior).
    """
    long = long.copy()
    by_team = long.groupby(["team_id", "season"], sort=False)
    for stat in BASE_STATS:
        long[f"roll{FORM_WINDOW}_{stat}"] = by_team[stat].transform(
            lambda s: _roll_mean(s, FORM_WINDOW))

    # Cold start teams (per season, so every season opener is flagged)
    long["team_cold"] = by_team.cumcount() < MIN_MATCHES
    return long

def apply_xg_fallback(long: pd.DataFrame) -> pd.DataFrame:
    """
    Apply xG fallback for missing values of xG features.
    """
    long = long.copy()
    long["xg_missing"] = False
    for xg, goals in [("xg_for", "goals_for"), ("xg_against", "goals_against")]:
        xg_col, goals_col = f"roll{FORM_WINDOW}_{xg}", f"roll{FORM_WINDOW}_{goals}"
        # Flag means "rolling xG was unavailable" regardless of whether we
        # can substitute goals; the cold-start path (both NaN) still counts.
        missing = long[xg_col].isna()
        long["xg_missing"] = long["xg_missing"] | missing
        fillable = missing & long[goals_col].notna()
        long.loc[fillable, xg_col] = long.loc[fillable, goals_col]
    return long

def fill_cold_start(long: pd.DataFrame) -> pd.DataFrame:
    """
    Replace every remaining NaN rolling value with the constant COLD_START_PRIOR.
    These are the earliest matches in a team's season, which have no prior
    history to average; the xG columns fall back to the goals prior. A constant
    is used deliberately: any data-derived prior would have to be computed from
    strictly earlier dates to stay leakage-safe, and it buys no measurable
    accuracy on total goals.
    """
    long = long.copy()
    underlying = {"goals_for": "goals_for", "goals_against": "goals_against",
                  "xg_for": "goals_for", "xg_against": "goals_against"}
    for stat, source in underlying.items():
        col = f"roll{FORM_WINDOW}_{stat}"
        long[col] = long[col].fillna(COLD_START_PRIOR[source])
    return long


def add_promotion_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add promoted/relegated flags to the DataFrame.

    A team's division for a season is resolved deterministically as the division
    it played the most matches in that season, ties broken by the latest match
    date, so a mid-season anomaly is never decided by input row order.
    """
    h = df[["home_id", "season", "league_name", "date"]].rename(columns={"home_id": "team_id"})
    a = df[["away_id", "season", "league_name", "date"]].rename(columns={"away_id": "team_id"})
    division = (pd.concat([h, a], ignore_index=True)
                .groupby(["team_id", "season", "league_name"], as_index=False)
                .agg(_n=("date", "size"), _last=("date", "max"))
                .sort_values(["team_id", "season", "_n", "_last"],
                             ascending=[True, True, False, False])
                .drop_duplicates(["team_id", "season"]))
    team_seasons = (division[["team_id", "season", "league_name"]]
                    .sort_values(["team_id", "season"]))

    team_seasons["division_rank"] = team_seasons.league_name.map(LEAGUE_RANK)
    prev_rank = team_seasons.groupby("team_id")["division_rank"].shift(1)
    team_seasons["promoted"] = (prev_rank.notna() & (team_seasons.division_rank > prev_rank)).astype(int)
    team_seasons["relegated"] = (prev_rank.notna() & (team_seasons.division_rank < prev_rank)).astype(int) 
    team_seasons = team_seasons[["team_id", "season", "promoted", "relegated"]]

    for prefix, id_col in [("home", "home_id"), ("away", "away_id")]:
        side_flags = team_seasons.rename(columns={
            "team_id": id_col,
            "promoted": f"{prefix}_promoted", "relegated": f"{prefix}_relegated"})
        df = df.merge(side_flags, on=[id_col, "season"], how="left")
    return df


def _attach_promotion(long: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the wide per-side promoted/relegated flags produced by
    ``add_promotion_flag`` onto each long team-row by (match_id, team_id) so
    every long row carries the flags exactly once, with no row-count change.
    """
    parts = []
    for prefix, id_col in [("home", "home_id"), ("away", "away_id")]:
        part = df[["match_id", id_col, f"{prefix}_promoted", f"{prefix}_relegated"]].rename(
            columns={id_col: "team_id",
                     f"{prefix}_promoted": "promoted",
                     f"{prefix}_relegated": "relegated"})
        parts.append(part)
    flags = pd.concat(parts, ignore_index=True).drop_duplicates(["match_id", "team_id"])
    return long.merge(flags, on=["match_id", "team_id"], how="left", validate="many_to_one")


def _team_feature_cols() -> list[str]:
    return ([f"roll{FORM_WINDOW}_{stat}" for stat in BASE_STATS]
            + ["promoted", "relegated", "team_cold", "xg_missing"])
 
 
def add_opponent(long: pd.DataFrame) -> pd.DataFrame:
    """
    Self-join on match_id so each team-match row also carries its opponent's
    pre-kickoff features (the GLM needs opponent defence to predict own scoring).

    The self-join pairs every team-row with both rows of its match; the mirror
    filter then DROPS the self-pair (team vs itself), leaving exactly the two
    real team rows per match.
    """
    # A match whose two team-rows share a team_id (home_id == away_id) would be
    # erased entirely by the mirror filter below. Detect it explicitly instead.
    collision = long.duplicated(["match_id", "team_id"], keep=False)
    if collision.any():
        bad = sorted(long.loc[collision, "match_id"].unique().tolist())
        raise ValueError(f"match(es) with home_id == away_id (self-play): {bad}")
    mirror = ["team_id", "team"] + _team_feature_cols()
    other = long[["match_id"] + mirror].rename(columns={c: f"opp_{c}" for c in mirror})
    merged = long.merge(other, on="match_id", how="inner", validate="many_to_many")
    return merged[merged.team_id != merged.opp_team_id].reset_index(drop=True)

def _feature_cols() -> list[str]:
    """
    The model's feature list, derived purely from the module constants above.
    Mirrors the transformations build_features applies: every team feature is
    emitted twice (own_/opp_), team_cold and xg_missing collapse to the pair
    is_cold_start / xg_fallback, and the venue/context flags are appended.
    league_name and game_week stay in META_COLS, so they are not repeated here.
    """
    mirrored = [c for c in _team_feature_cols() if c not in ("team_cold", "xg_missing")]
    context = ["is_crowds"]
    return ([f"own_{c}" for c in mirrored] + [f"opp_{c}" for c in mirrored]
            + context + ["is_home", "is_cold_start", "xg_fallback"])


FEATURE_COLS: list[str] = _feature_cols()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "match_id" not in df.columns:
        # Deterministic, stable key so equivalent inputs (any row order) get the
        # same ids: a bare single-key sort over heavily tied dates is not stable.
        df = df.sort_values(["date", "home_id", "away_id"], kind="stable").reset_index(drop=True)
        df["match_id"] = np.arange(len(df))
    if not df["match_id"].is_unique:
        dups = sorted(df.loc[df["match_id"].duplicated(), "match_id"].unique().tolist())
        raise ValueError(f"Duplicate match_id values in input: {dups}")
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    df = add_promotion_flag(df)  # needs the wide frame (home_id / away_id)

    long = build_long(df)
    long = _attach_promotion(long, df)
    long = add_rolling(long)
    long = apply_xg_fallback(long)
    long = fill_cold_start(long)
 
    team_feats = _team_feature_cols()
    out = add_opponent(long)
    out = out.rename(columns={c: f"own_{c}" for c in team_feats})
    out = out.merge(df[["match_id", "total_goals"]], on="match_id", how="left",
                    validate="many_to_one")

    # is_crowds never passes through fill_cold_start. Fill and cast to a stable
    # int dtype before FEATURE_COLS is frozen. A missing is_crowds is the COVID
    # behind-closed-doors gap, so it defaults to 0 (no crowd).
    out["is_crowds"] = out["is_crowds"].fillna(0).astype(int)

    out["is_home"] = (out.venue == "home").astype(int)
    out["is_cold_start"] = (out.own_team_cold | out.opp_team_cold).astype(int)
    out["xg_fallback"] = (out.own_xg_missing | out.opp_xg_missing).astype(int)
    out = out.drop(columns=["own_team_cold", "opp_team_cold",
                            "own_xg_missing", "opp_xg_missing"])
 
    out = out.drop(columns=["xg_for", "xg_against"], errors="ignore")
    selector = list(dict.fromkeys(META_COLS + FEATURE_COLS + LABEL_COLS))
    out = out[[c for c in selector if c in out.columns]]
    out = out.sort_values(["date", "match_id", "venue"]).reset_index(drop=True)

    return out

def main() -> None:
    features = build_features(pd.read_csv(DATA_PATH))
    features.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()