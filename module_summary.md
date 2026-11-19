# Hourly Bike-Rental Prediction

AI Foundations (Edward Hickman)

## Overview

This project uses supervised regression to estimate recorded hourly Capital Bikeshare rentals from calendar variables and contemporaneously observed weather. It trains a random-forest regressor on the real UCI Bike Sharing dataset and compares it with a training-mean baseline on later held-out dates (Fanaee-T, 2013; Fanaee-T & Gama, 2014). The dataset is available at https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset; this evaluation does not demonstrate an advance forecast using weather forecasts.

## Dataset Description

The unchanged hour.csv contains 17,379 recorded hours and 17 columns covering Washington, D.C.'s Capital Bikeshare system in 2011–2012, with weather and calendar information (Fanaee-T & Gama, 2014). The target cnt is total observed rental count; twelve predictors comprise season, year, month, hour, holiday, weekday, working-day status, weather category, temperature, apparent temperature, humidity and wind speed. The raw file has no missing cells or repeated date/hour keys, but 165 clock hours are absent from its complete time span and are not invented as zero observations. The raw columns also include a row identifier, date and the casual/registered rental subtotals, which are not model inputs. The source Readme and actual file agree on 17,379 records, while the UCI landing page's 17,389 count is inconsistent; all computations use the included file.

## Modeling Approach

The notebook loads the local CSV, displays its first rows and data types, and verifies its fingerprint, missingness, key uniqueness and relevant ranges. The twelve input features exclude the target, the row identifier and raw date. Crucially, casual and registered are excluded because their sum is exactly cnt; including them would reveal the answer rather than demonstrate useful prediction. This explicit feature-availability audit addresses leakage risks described by Kapoor and Narayanan (2023).

A ColumnTransformer one-hot encodes eight categorical fields and passes through four numerical weather fields. Encoders are fitted on each training subset only; unseen categories use all-zero encodings rather than categories learned from validation or test data. No missing-value imputation is needed for the verified input. Extra scaling is not used for tree thresholds; the provider-normalized weather values remain unchanged. Those released normalizations are source preprocessing, not transformations newly fitted on the full dataset by this analysis.

The last 20% of distinct dates, rounded up, form the test set; entire days remain together. Development covers 2011-01-01–2012-08-06 (13,891 rows, 584 dates), and testing covers 2012-08-07–2012-12-31 (3,488 rows, 147 dates). Three expanding-window development folds each leave one whole calendar day between training and validation. Fresh preprocessing and models are fitted within every fold. Chronology prevents future training relative to each evaluated period; it does not make temporally correlated records independent or make one future period representative of all deployment conditions (Kapoor & Narayanan, 2023).

The fixed RandomForestRegressor averages 150 bootstrapped regression trees, uses squared-error splitting, a maximum depth of 14 and a minimum leaf size of two, with seed 20261119 and a single worker. All available transformed features are considered at each split (max_features=1.0). Tree ensembles can represent nonlinear relationships and interactions without specifying a linear functional form (Breiman, 2001). Depth and leaf constraints limit complexity but do not prove the model generalizes. The training-mean DummyRegressor is a transparent, deliberately simple comparison, not an optimized seasonal forecasting benchmark. No hyperparameter search or test-based model selection was performed; both models were refitted on all development records before final evaluation.

RMSE is the primary metric because squared errors emphasize large rental-count misses; taking the root returns the error to rentals per recorded hour. MAE supplies a complementary mean absolute miss in the same units. These losses express different priorities; neither is universally superior (Hodson, 2022). R² is 1 minus the model's sum of squared errors divided by that of the evaluation period's actual mean; it can be negative and is not percent accuracy (scikit-learn developers, n.d.). The evaluation uses empirical losses, not a claim that errors are normally distributed, independent or homoscedastic.

## Results

The forest's development-fold RMSE values were 106.61, 89.91, 132.99 rentals per recorded hour. All three folds are retained in the notebook and saved table; their variation reflects different training sizes and time periods, not independent repetitions of the same experiment.

On the 3,488 later held-out records, the forest achieved RMSE 104.65, MAE 72.57 and R² 0.775. The training-mean baseline achieved RMSE 232.56, MAE 174.97 and R² -0.112. Figure 1 shows the resulting 55.00% RMSE reduction relative to this baseline. The negative baseline R² is possible because the development mean used for predictions differs from the test period's mean.

Forest training RMSE was 54.05, training MAE 36.67 and training R² 0.895, substantially better than the held-out scores. This gap is consistent with limited generalization and possibly overfitting, but time-period differences prevent attributing it entirely to overfitting. Mean recorded rentals rose from 174.62 in development to 248.56 in testing.

Figure 2 plots every held-out actual/predicted pair, rather than a selected successful week, and exposes departures from the exact-prediction diagonal. The mean actual-minus-predicted residual was 43.91, so the model underpredicted the test period on average. Figure 3 shows the largest clock-hour MAE at hour 16:00, with 133.03 rentals per recorded hour. Group counts are labeled so readers can inspect coverage. No final-test categorical values were absent from the full development encoder's categories.

## Interpretation for a Non-Technical Audience

The model's absolute miss averaged about 73 rentals per recorded hour in the later test period; individual errors could be much larger or smaller. It performed better than using one historical average for every hour, but this does not show that it beats a carefully designed seasonal baseline or every alternative model. Its test R² is not an accuracy of 77.5%; it summarizes squared-error performance relative to a particular reference prediction. Large average errors at some hours and overall underprediction could matter for staffing or bike allocation, so a favorable overall score alone would not justify automated decisions. Because the test supplied observed weather, the scores should not be advertised as the performance of an advance forecasting service.

## Limitations and Potential Bias

The two historical years represent one city's system and recorded rentals, not all travel needs. Unmet demand, station outages and people unable to access the service are not directly measured, and the absence of a row does not establish zero demand. Using observed rentals to withdraw service from low-count areas could reinforce existing access differences; the file contains neither station-level nor demographic information sufficient to test geographic or demographic fairness. A proposed mitigation is to collect service-availability and station/neighborhood coverage data, audit allocation impacts and retain human review before deployment.

Only one later test block is evaluated, and its seasonal and usage mix differs from development. Temporal dependence, changing membership, extreme weather or service changes may invalidate future performance expectations. The fixed random forest has limited extrapolation ability, and correlated calendar/weather inputs make simple feature-importance rankings difficult to interpret causally. No feature-importance, causal or prediction-interval claims are made. The one-day development gap is a precaution, not evidence that all dependence has vanished.

Chronological separation, excluded outcome-derived columns and training-only encoding reduce identifiable leakage risks, following Kapoor and Narayanan (2023). They do not validate weather availability for advance use: a real forecast would require weather forecasts available at prediction time and a new prospective evaluation. Future work should also compare stronger seasonal baselines, evaluate later rolling periods and investigate large hour-specific errors. RMSE/MAE do not establish appropriate business loss weights or error distributions by themselves (Hodson, 2022).

## Reproducibility and Responsible Practice

The package includes the original CSV and fingerprint, the executed modeling.ipynb, saved split assignments, fold boundaries, predictions, metrics and three figures. requirements.txt was generated by pip freeze, and the README explains dependency installation and restarting the kernel before running all cells. Fixed model parameters and random seeds make the computational choices explicit; package pins are not a guarantee of identical results on every operating system. Re-execution of the unchanged notebook is a reproducibility check, not additional test-driven tuning.

Replit AI assisted with planning, coding, debugging, verification and report drafting. All reported metrics come from executed outputs; no hypothetical experiments are presented as results. The learner remains responsible for reviewing, understanding and revising the work and complying with the course's academic-integrity and AI-use policies.

## Conclusion

The fixed forest improved on the training-mean baseline under the documented later-period evaluation, but appreciable errors, underprediction and a training/test gap remain. The central lesson is to define the prediction setting and preserve feature and time boundaries before interpreting a favorable metric. This is a transparent educational ML workflow, not a validated causal explanation or production bike-allocation system.

## References

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Fanaee-T, H. (2013). Bike Sharing [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894

Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2, 113–127. https://doi.org/10.1007/s13748-013-0040-3

Hodson, T. O. (2022). Root-mean-square error (RMSE) or mean absolute error (MAE): When to use them or not. Geoscientific Model Development, 15, 5481–5487. https://doi.org/10.5194/gmd-15-5481-2022

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

scikit-learn developers. (n.d.). r2_score. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

