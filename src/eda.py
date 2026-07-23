"""
eda.py - exploratory data analysis

Script to perform exploratory data analysis on the match data.

"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = Path("data/raw/match_data_clean.csv")
PLOT_PATH = Path("plots")

def total_goals_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of total goals scored in matches.
    """
    print("\n" + "=" * 20 + f"\nTotal Goals Distribution\n" + "=" * 20)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    mean, var = df["total_goals"].mean(), df["total_goals"].var()
    print(f"mean={mean:.2f}, variance={var:.2f}")
    print(f"variance/mean ratio: {var/mean:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["total_goals"], bins=20, edgecolor="black")
    ax.set_xlabel("Total Goals")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Total Goals")
    fig.tight_layout()
    fig.savefig(PLOT_PATH / "total_goals_distribution.png")

def total_xg_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of total expected goals (xG) in matches.
    """
    print("\n" + "=" * 20 + f"\nTotal xG Distribution\n" + "=" * 20)
    df["total_xg"] = df["home_xg"] + df["away_xg"]
    mean, var = df["total_xg"].mean(), df["total_xg"].var()
    print(f"mean={mean:.2f}, variance={var:.2f}")
    print(f"variance/mean ratio: {var/mean:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["total_xg"], bins=30, edgecolor="black")
    ax.set_xlabel("Total Expected Goals (xG)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Total Expected Goals (xG)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH / "total_xg_distribution.png")

def goal_rates_by_league(df: pd.DataFrame) -> None:
    """
    Plot the average goal rates by league.
    """
    print("\n" + "=" * 20 + f"\nGoal Rates by League\n" + "=" * 20)
    goal_rates = df.groupby("league_name")[["home_goals", "away_goals"]].mean()
    goal_rates["total_goals"] = goal_rates["home_goals"] + goal_rates["away_goals"]
    print(goal_rates.to_string())

def home_adv_by_season(df: pd.DataFrame) -> None:
    """
    Plot the home advantage (home goals - away goals) by season.
    """
    print("\n" + "=" * 20 + f"\nHome Advantage by Season\n" + "=" * 20)
    print("overall home advantage (mean):", (df["home_goals"] - df["away_goals"]).mean())
    by_season = ( df.groupby("season")
                     .agg(home_goals=("home_goals", "mean"),
                          away_goals=("away_goals", "mean"),
                          crowd_share=("is_crowds", "mean"))
                     .assign(home_adv=lambda x: x["home_goals"] - x["away_goals"])
    )
    print("home advantage by season:\n", by_season.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(by_season.index, by_season["home_adv"], marker="o",
            label="Home Advantage (Home Goals - Away Goals)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home Advantage")
    ax.set_title("Home Advantage by Season")
    # mark empty crowds seasons
    empty_crowds = by_season[by_season["crowd_share"] < 0.5]
    ax.scatter(empty_crowds.index, empty_crowds["home_adv"], color="red", label="Empty Crowds Season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH / "home_adv_by_season.png")

def xg_missingness_by_league_season(df: pd.DataFrame) -> None:
    """
    Print the missingness of xG by league and season.
    """
    print("\n" + "=" * 20 + f"\nxG Missingness by League and Season\n" + "=" * 20)
    xg = df[["league_name", "season"]].copy()
    xg["matches"] = 1
    xg["home_xg_null"] = df["home_xg"].isna().astype(int)
    xg["away_xg_null"] = df["away_xg"].isna().astype(int)
    xg["either_null"] = (df["home_xg"].isna() | df["away_xg"].isna()).astype(int)
    xg = xg.groupby(["league_name", "season"]).sum()
    xg["pct_either_null"] = (100 * xg["either_null"] / xg["matches"]).round(1)
    print(xg.to_string())



def main() -> None:
    df = pd.read_csv(DATA_PATH)
    total_goals_distribution(df)
    total_xg_distribution(df)
    goal_rates_by_league(df)
    home_adv_by_season(df)
    xg_missingness_by_league_season(df)

if __name__ == "__main__":
    main()