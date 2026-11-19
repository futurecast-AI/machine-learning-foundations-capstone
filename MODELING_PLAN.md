# Modeling plan, recorded before training

Author: AI Foundations (Edward Hickman)

## Problem and scope

Supervised regression: estimate recorded hourly Capital Bikeshare rental counts
(`cnt`) using calendar variables and contemporaneously observed weather.
This is retrospective conditional prediction, not a deployed advance forecast:
actual weather for a future hour would not be available without forecasts.
Do not claim to estimate unmet demand, individual behavior, or causal effects.

## Data integrity and leakage prevention

Use the real UCI Bike Sharing `hour.csv`, unchanged, beside the notebook.
Initial inspection established 17,379 rows, 17 columns, 731 dates, no missing
cells and no repeated date/hour keys; model scores have not been inspected.
Use the file and source Readme row count rather than the landing page's
inconsistent 17,389 figure. The dataset differs from Titanic and Student Performance.

Predictors: season, yr, mnth, hr, holiday, weekday, workingday, weathersit,
temp, atemp, hum and windspeed (12 input features).
Exclude cnt from inputs; exclude casual and registered because their sum is
exactly cnt; exclude instant because it is a row identifier. Use dteday only
to create timestamps and split dates, not as an unprocessed predictor.
Retain provider-normalized weather values without fitting any extra scaling.
Missing clock hours are audited, not invented as zero-rental observations.

## Split and model

Reserve the last 20% of distinct dates (rounded up) as the test period.
Keep entire calendar days together. Fit/evaluate only on earlier dates during
development. No random row splitting and no tuning using the final test.

Use three expanding-window folds on development dates, with a one-calendar-day
gap before each validation fold. These folds are diagnostics, not a randomized
sampling distribution. Fit fresh preprocessing and models on each fold's
training subset only.

Compare a training-mean DummyRegressor with a fixed RandomForestRegressor:
150 trees, max_depth=14, min_samples_leaf=2, max_features=1.0,
bootstrap=True, criterion="squared_error", random_state=20261119, n_jobs=1.
Apply OneHotEncoder(handle_unknown="ignore", sparse_output=False) to the eight
calendar/weather categorical predictors and pass through the four numeric
weather predictors. No imputation is needed if the verified input is complete.
No feature selection, model-family search or hyperparameter tuning is planned.
Retain and report this fixed model even if it loses to the baseline.

After development cross-validation, fit both models on the complete development
period. Evaluate the untouched later test period once per clean notebook run.
Re-running an unchanged notebook for reproducibility is not additional tuning.

## Evaluation and communication

Primary metric: RMSE, in rentals per recorded hour, to weight large misses more
heavily. Secondary metrics: MAE (mean absolute miss in the same units) and R²
(squared-error improvement relative to the evaluation period's mean, not
"percent accuracy"). Report training and test metrics, the simple baseline,
every development fold, and the train/test gap without treating differences
as a pure estimate of overfitting; temporal distribution shift also matters.

Figures: test RMSE comparison, all test actual/predicted pairs, and test MAE
by clock hour with group counts. Do not cherry-pick a successful test week.
No prediction intervals, demographic fairness claims, statistical significance
claims or production-readiness claims are planned.

## Limitations and sources

Use Breiman (2001) for the ensemble approach, Kapoor and Narayanan (2023) for
leakage precautions, Hodson (2022) for error metrics and official scikit-learn
documentation for API-specific definitions. Attribute the UCI dataset and cite
its requested Fanaee-T and Gama publication. Discuss historical/city coverage,
unrecorded hours, peak misses, temporal dependence, shifting usage and the
availability of observed weather. Disclose Replit AI assistance.
