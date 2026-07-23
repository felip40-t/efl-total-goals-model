"""
clean.py

Script to clean data file to make sure all data can be used.

"""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/raw/match_data.csv")
OUTPUT_PATH = Path("data/raw/match_data_clean.csv")

STRING_COLS = ["league_name", "season", "home_team", "away_team"]
FLAG_COLS = ["is_neutral_venue", "is_crowds"]
GOAL_COLS = ["home_goals", "away_goals"]
XG_COLS = ["home_xg", "away_xg"]
ABSURD_GOAL_THRESHOLD = 20

def clean(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Inspect data.
    """
    print("\n" + "=" * 20 + f"\nClean\n" + "=" * 20)
    dropped = {}

    # date: parse, sort ascending, reset index
    s = df["date"].astype("string").str.strip()
    s = s.str.replace("Z", "", regex=False)        # drop the Zulu marker
    df["date"] = pd.to_datetime(s.str.slice(0, 10), format="%Y-%m-%d", errors="coerce").dt.date
    if df["date"].isna().any():
        print(f"[date] {df['date'].isna().sum()} unparseable date(s) left as NaT.")
    df = df.sort_values("date", na_position="last").reset_index(drop=True)
    print("[date] parsed, sorted ascending, index reset.")

    # Strip whitespace from string columns
    for col in STRING_COLS:
        df[col] = df[col].astype("string").str.strip()
    print("[string] whitespace stripped from string columns.")
    # Check league names and seasons
    print("  league_name:", sorted(df["league_name"].dropna().unique()))
    print("  season:", sorted(df["season"].dropna().unique()))

    # Check goals and game week 
    for col in GOAL_COLS + ["game_week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in XG_COLS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        lost = coerced.isna() & df[col].notna()
        if lost.any():
            print(f"[{col}] {lost.sum()} unparseable value(s) set to NaN.")
        df[col] = coerced.astype("Float64")
    
    # Check flags
    for col in FLAG_COLS:
        num = pd.to_numeric(df[col], errors="coerce")
        valid = num.isin([0, 1])
        bad = (~valid) & num.notna()
        if bad.any():
            print(f"[{col}] {bad.sum()} non-0/1 value(s).")
        if num.isna().any():
            print(f"[{col}] {num.isna().sum()} unparseable value(s) set to NaN.")

    # team names and ids - id is canonical, standardize names to id, keep most frequent name for each id
    full = pd.concat([
        df[["home_id", "home_team"]].rename(columns={"home_id": "team_id", "home_team": "team_name"}),
        df[["away_id", "away_team"]].rename(columns={"away_id": "team_id", "away_team": "team_name"})
    ]).dropna(subset=["team_id", "team_name"])
    counts = full.groupby(["team_id", "team_name"]).size().rename("count").reset_index()
    canon = (counts.sort_values(["team_id", "count"], ascending=[True, False])
                  .drop_duplicates(subset=["team_id"])
                  .set_index("team_id")["team_name"])

    names_per_id = counts.groupby("team_id")["team_name"].nunique()
    conflicted_ids = names_per_id[names_per_id > 1].index.tolist()
    if conflicted_ids:
        print(f"[team] {len(conflicted_ids)} team_id(s) with multiple names:")
        for team_id in conflicted_ids:
            variants = counts.loc[counts["team_id"] == team_id]
            print(f"  {team_id}: {variants['team_name'].tolist()} (counts: {variants['count'].tolist()})")
        df["home_team"] = df["home_id"].map(canon).fillna(df["home_team"])
        df["away_team"] = df["away_id"].map(canon).fillna(df["away_team"])
        print("[team] standardized team names to most frequent name per team_id.")
    else:
        print("[team] all team_ids have a single name; no standardization needed.")

    # Check for one name with multiple ids
    ids_per_name = counts.groupby("team_name")["team_id"].nunique()
    clashing_names = ids_per_name[ids_per_name > 1].index.tolist()
    if clashing_names:
        print(f"[team] {len(clashing_names)} team_name(s) with multiple ids:")
        for team_name in clashing_names:
            variants = counts.loc[counts["team_name"] == team_name]
            print(f"  {team_name}: {variants['team_id'].tolist()} (counts: {variants['count'].tolist()})")

    # drop exact duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped["exact_duplicate"] = before - len(df)
    print(f"\n[dedup] dropped {before - len(df)} exact-duplicate row(s).")

    # missing goals = unplayed/void exclude (but keep rows missing only xG)
    missing = df[GOAL_COLS].isna().any(axis=1)
    dropped["missing_goal_outcome"] = int(missing.sum())
    df = df[~missing].reset_index(drop=True)
    print(f"[void] excluded {missing.sum()} row(s) missing goals; xG-only-missing kept.")

    # Check for absurd scores and flag, drop negative goals
    neg = (df[GOAL_COLS] < 0).any(axis=1)
    dropped["negative_goals"] = int(neg.sum())
    df = df[~neg].reset_index(drop=True)
    print(f"[goals] dropped {neg.sum()} row(s) with negative goals.")
    absurd = (df[GOAL_COLS] > ABSURD_GOAL_THRESHOLD).any(axis=1)
    if absurd.any():
        print(f"[goals] {absurd.sum()} row(s) with absurdly high goals (> {ABSURD_GOAL_THRESHOLD}) flagged but kept.")
    
    return df, dropped


def write(df: pd.DataFrame) -> None:
    """
    Write cleaned data to CSV.
    """
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote cleaned data to {OUTPUT_PATH.resolve()}.")

def summary(df: pd.DataFrame, dropped: dict) -> None:
    """
    Print summary of cleaning.
    """
    print("\n" + "=" * 20 + f"\nSummary\n" + "=" * 20)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("\nNull count per column:\n" + df.isna().sum().to_string())
    print("\nDropped rows by reason:")
    for reason, count in dropped.items():
        print(f"  {reason}: {count}")

    
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}. "
            f"Place match_data.csv next to this script."
        )
    df = pd.read_csv(INPUT_PATH)
    cleaned_df, dropped = clean(df)
    write(cleaned_df)
    summary(cleaned_df, dropped)

if __name__ == "__main__":
    main()