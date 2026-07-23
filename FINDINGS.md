# Total Goals Model: Summary of Findings

A model predicting the **total goals** in an English football match (Premier
League, Championship, League One; 2018/19 to date), built for the bet365
Quantitative Analysis (Sports) assessment.

**Headline result:** across an expanding-window walk-forward (7 folds, 9,756
out-of-sample matches over 2019/20 to 2025/26) the model's pooled total-goals
MAE is **1.286**, against a naive constant-mean baseline of **1.299**. It is
below the baseline on each of the 7 folds. The difference is small (~1%). The
aim was a calibrated model and a leakage-free validation design rather than a
large accuracy gain.

All figures below come from `data/processed/metrics.csv`, produced by
`src/evaluate.py`.

## Modelling decision and rationale

Total goals is modelled **indirectly**: each match is split into two team-rows
with response `goals_for`, a single Poisson GLM (`statsmodels`) is fit over all
team-rows with an `is_home` term for home advantage, and the two predicted rates
are summed to `lambda_total = lambda_home + lambda_away`. Because the sum of two
independent Poissons is Poisson, the match total is `Poisson(lambda_total)`,
which yields a full predictive distribution for the Over/Under market, not only
a point forecast. 

## Data quality issues found and how they were handled

Handled in `src/clean.py`; each step is logged.

- **Dates** carried a trailing `Z` (Zulu marker) and were stored as strings.
  Parsed to dates, unparseable values left as `NaT`, frame sorted ascending.
- **xG is missing for 37.3% of 2018/19 matches** (≈5% overall, near-complete
  from 2019/20 on). xG is never imputed at the raw level; the gap is absorbed
  downstream by a rolling-goals fallback (see below).
- **`is_neutral_venue` is constant 0** across all rows and carries no
  information, so it was dropped from the feature set entirely rather than fed to
  the GLM as a dead column.
- **`is_crowds` = 0 for 1,601 matches**, the COVID behind-closed-doors period.
  This is real signal (it enters the model); a small number of missing values
  default to 0 (no crowd), consistent with that gap.
- **Team identity**: `team_id` is treated as canonical and names are standardised
  to the most frequent spelling per id. On this data no id had conflicting names
  and no name mapped to multiple ids, so the step changed nothing here; it
  handles name drift if it appears later.
- **Voids / duplicates / impossible scores**: rows missing either goals value are
  dropped as unplayed (none occurred here), exact duplicates dropped (none),
  negative goals dropped (none), and scores above 20 are flagged. Goals are never
  imputed.

## xG and leakage treatment

Same-match xG (`home_xg`, `away_xg`) is a **post-match** quantity (known only
after kick-off), so it is treated as strictly off-limits as a feature. It
enters the model **only** as a team's rolling average over its **previous**
matches (`shift(1)` within `(team_id, season)`), never for the match being
predicted. The raw `home_xg` / `away_xg` columns and the same-match goals never
enter `FEATURE_COLS`, which is built only from pre-match rolling features and
context flags. Where a team's rolling xG is unavailable (the
2018/19 gap), it falls back to that team's rolling *goals* and sets an
`xg_fallback` flag, so a missing pre-match average is marked, not silently
zeroed. All rolling means use `shift(1)`, so no feature ever reads the current or
a future match.

## Validation design

**Expanding-window walk-forward** (`train.py:walk_forward_folds`): for every
season from the second onward, the model trains on all strictly-earlier seasons
and is scored on that season, giving 7 folds (test 2019/20 through 2025/26,
9,756 out-of-sample matches). The training window grows each fold: fold 1 trains
on 2018/19 alone, the last on 2018/19 to 2024/25. No fold ever trains on its own
test season or later, and combined with the `shift(1)` feature construction the
evaluation is predictive at every step. The naive baseline is computed
per fold from that fold's training seasons only, so it never sees the future
either. 

## Results

From `data/processed/metrics.csv` (one row per
`season`/`league`/`group`/`model`/`metric`; `season` is a fold or `pooled`,
`league` a division or `all`). Lower is better for every metric except `bias`
(diagnostic) and the Over/Under base/predicted rates (calibration checks). The
tables below are the **pooled** figures over all 9,756 out-of-sample matches.

**Total goals, model vs naive baseline** (`Poisson(lambda_total)` vs each fold's
training-mean constant):

| Metric | Model | Naive | Why this metric |
|---|---|---|---|
| MAE | 1.2864 | 1.2990 | Average error in goals, directly interpretable on the target's scale. |
| RMSE | 1.6005 | 1.6101 | Penalises large misses (blowouts) more than MAE. |
| bias | 0.0045 | 0.0121 | Systematic over/under-prediction; near-zero indicates little mean drift. |
| Poisson NLL | 1.8486 | 1.8541 | Proper scoring rule for counts; rewards the whole predictive distribution, not only the point. |
| Mean Poisson deviance | 1.0797 | 1.0908 | GLM goodness-of-fit on the response scale; comparable to training deviance. |

**Per-fold total-goals MAE** (model MAE below the naive baseline in every fold):

| Fold (test season) | Model MAE | Naive MAE |
|---|---|---|
| 2019/2020 | 1.2882 | 1.2899 |
| 2020/2021 | 1.3168 | 1.3239 |
| 2021/2022 | 1.2985 | 1.3066 |
| 2022/2023 | 1.3093 | 1.3226 |
| 2023/2024 | 1.2793 | 1.3082 |
| 2024/2025 | 1.2687 | 1.2868 |
| 2025/2026 | 1.2237 | 1.2329 |

**Per-league total goals** (pooled; each division scored against its own
historical scoring rate, since scoring differs by division):

| League | Matches | Model MAE | Naive MAE | Model NLL | Naive NLL |
|---|---|---|---|---|---|
| Championship | 3,692 | 1.2579 | 1.2639 | 1.8179 | 1.8197 |
| League One | 3,524 | 1.2977 | 1.3017 | 1.8536 | 1.8518 |
| Premier League | 2,540 | 1.3120 | 1.3233 | 1.8861 | 1.8949 |

The result is not uniform across divisions. The model is below the
league-specific baseline on MAE in all three, by the largest margin in the
Premier League. In League One it is marginally higher (worse) on the
distributional metrics (NLL and RMSE), where its per-match probabilities are
close to the division's historical average.

**Per-side goals** (pooled; the model's actual target, `goals_for`, before
recombining, a diagnostic on the two-Poisson construction):

| Metric | Model | Why this metric |
|---|---|---|
| MAE | 0.9086 | Per-team scoring error in goals. |
| RMSE | 1.1562 | Large-miss sensitivity at team level. |
| bias | 0.0023 | Per-side calibration of the mean rate. |
| Poisson NLL | 1.4541 | Distributional fit on the fitted response. |
| Mean Poisson deviance | 1.1592 | GLM fit on the response the model is actually estimating. |

**Over/Under 2.5** (pooled; the implied totals betting market, from
`1 − Poisson.cdf(2, lambda_total)`):

| Metric | Model | Why this metric |
|---|---|---|
| Over2.5 base rate | 0.4963 | Realised frequency of Over 2.5, the reference. |
| Over2.5 mean predicted P | 0.4915 | Mean predicted probability; compared with the base rate for average calibration. |
| Over2.5 Brier | 0.2485 | Proper score for the probability forecast (0.25 is the coin-flip reference). |
| Over2.5 accuracy @0.5 | 0.5314 | Hit rate of the 0.5-threshold Over/Under call; compare with the majority-class floor below. |
| Over2.5 majority-class accuracy | 0.5037 | No-model floor: always calling the more common outcome (Under). The model is ~2.8pp above it. |

## Strengths

- **Leak-free construction.** Features are forward-only (`shift(1)` rolling means
  within `(team_id, season)`), and the walk-forward baselines are computed per
  fold from earlier seasons only, so no step reads the current or a future match.
- **Consistency across folds.** The model is below the naive baseline on all 7
  folds, so the difference does not rest on a single test season.
- **Full predictive distribution.** The Poisson total gives Over/Under
  probabilities, not only a point forecast.
- **Baseline comparison.** Every total-goals metric is reported beside a naive
  baseline.
- **Small model.** 16 features and one GLM.

## Limitations

- **Small lift on totals.** Form and pre-match xG separate the model from the
  mean only slightly (~1%); most match-total variance is irreducible at this
  feature depth.
- **Independence assumption.** Home and away goals are modelled as independent
  Poissons. A basic check in `evaluate.py` supports this for totals: conditional
  on the fitted rates the two sides' residuals correlate only **-0.04**
  (negligible, and negative), and the total's variance ratio is **0.97** (not
  overdispersed), so the assumption costs nothing measurable on total goals at
  this feature depth. These are first-pass checks; deeper ones are left as future work.
- **Thin early folds.** Fold 1 trains on 2018/19 alone (a season with 37% xG
  missing), and the final fold tests a partial 2025/26 (1,004 matches), so the
  extreme folds lean harder on the goals fallback and cold-start prior than the
  middle ones.
- **No team-strength parameters.** Strength is proxied by short-window rolling
  form, which is noisy for teams with few recent matches and resets every season.
- **Constant cold-start prior.** Season-openers with no history get a single
  league-agnostic constant (1.35), ignoring divisional scoring differences.

## Future refinements

- **Bivariate Poisson / Dixon–Coles** to model goal correlation and low-score
  structure. The independence checks above show little to gain on *totals*, so
  the value here is in the scoreline and correlated-outcome markets (exact
  score, both-teams-to-score) rather than the total itself.
- **Team-level random effects (a hierarchical / mixed model)** with shrinkage,
  replacing rolling form with stable attack/defence strengths.
- **Elo-style or league-relative priors** for promoted/relegated and cold-start
  teams instead of a flat constant.

## Pipeline and implementation

`make <name>` runs `src/<name>.py`. Order: `clean → features → train → evaluate`
(`eda` is optional and off the model path). The `data/` folder is committed
(`data/raw/` inputs, `data/processed/` generated outputs), so every figure here
is reproducible without re-running the pipeline.

- **`clean.py`**: `data/raw/match_data.csv` → `data/raw/match_data_clean.csv`. Typing,
  date parsing, team-name canonicalisation, void/duplicate removal; goals never
  imputed.
- **`features.py`**: `…_clean.csv` → `data/processed/features.csv`. Reshapes to two rows
  per match, builds `shift(1)` rolling form per `(team_id, season)`, applies the
  xG-goals fallback and constant cold-start fill, attaches promotion/relegation
  and opponent features. Emits 16 features.
- **`train.py`**: `features.csv` → `data/processed/predictions.csv`. One-hot
  league plus the feature set, an expanding-window walk-forward that fits a fresh
  Poisson GLM per fold, predictions recombined to match-level lambdas and stacked
  across folds (tagged by test season).
- **`evaluate.py`**: `predictions.csv` (+ clean data for the baselines) →
  prints per-fold, per-league and pooled metrics plus two home/away
  independence checks (residual correlation and total variance ratio), and
  writes **`data/processed/metrics.csv`** (tidy long format: one row per
  season/league/group/model/metric, with per-league baselines computed within
  each division).
- **`eda.py`**: distributions, home advantage by season, and xG-missingness
  tables/plots supporting the notes above.

**Metrics CSV:** `data/processed/metrics.csv`, the single source for the
Results tables.
