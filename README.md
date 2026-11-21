# hotel-booking-cancellation-prediction
## Predicting Booking Cancellations at Reservation Time: A Temporal Validation Study with Leakage Control on Hotel Demand Data.
### Course: DS5002: Data Science Tools & Technologies (NUCES)
### Instructor: Dr. Asma Ahmad | TA: Burhan Hadi Butt
### Group: Nurmeen Shahid (26L-8010), Ayesha Mazher (26L-8005)

---
## Table of Contents
1. Overview
2. Research Question
3. Dataset
4. Problem Definition
5. Repository contents
6. How to run
7. Project pipeline
8. Part1: Data preparation, Leakage audit and EDA
9. Part 2: Modelling and Evaluation
10. Answers to the research question
11. References

---

## 1. Overview
Dealing with late cancellations is one of the most frustrating parts of a business, especially for hotels because of the number of preparations and resources used for a single guest. In this regard, a room that was prepared and held for a guest who never arrives is lost income and wasted resources.
Therefore, this project asks;
> Can information available **at reservation time** identify hotel bookings at elevated risk of cancellation, when models are trained on earlier bookings, tested on later ones, and every outcome-revealing variable is excluded?

### The project is done in 2 parts:
Part 1 covers the following 6 parts:
1. Download the data and check that its structure is what we expect.
2. Deal with missing values, duplicates, outliers and invalid records.
3. Sort every column by when its value becomes known (the leakage audit).
4. Build new features and define three feature sets for Part 2.
5. Set up a time-based split that keeps 2017 untouched as the test period.
6. Describe the cancellation patterns, always showing sample sizes and 95% confidence intervals.

Similarly, Part 2 covers the following 6 parts:
1. Rebuild the cleaned data and define six feature sets (the strict main set, three variations and two leakage benchmarks), so Part 2 can run on its own.
2. Set up the time-based validation: 2017 stays untouched as the test period, and 2015-2016 is split into three rolling-origin folds.
3. Train the baselines (prior rate, one rule, logistic regression) and the models (L1 logistic, random forest, XGBoost), tune them on the rolling folds with PR-AUC, and choose the final model on validation only.
4. Measure how much leakage and the split design change the score: the feature-set comparison (RQ1), the random split against the time split, the score for each test month, and a check with duplicate rows removed (RQ2).
5. Turn the risk scores into decisions: compare doing nothing, acting on every booking and acting above a cost-based threshold, test how sensitive this is to the cost assumptions, and build the risk tiers (RQ3).
6. Find the factors behind cancellation with SHAP, check that they are stable over time and across hotels (RQ4), then save the final pipeline, which scores new bookings into risk tiers, and report the final results and limits.

### What makes this project different from other standard "Churn Prediction" models?
1. Leakage is measured and not just ignored in this project. Every column is classified by when its value becomes known, and the effect of leaky columns on the score is quantified.
2. Validation respects time. Models are tuned with rolling-origin folds and tested on 2017, which no decision ever touched.
3. Predictions lead to a decision (raw data -> wisdom). Risk scores are turned into a cost-based action policy and three risk tiers.

---

# 2. Research Question

| # | Question |
|---|----------|
| RQ1 | How much of the apparent predictive performance disappears when outcome-revealing and post-booking variables are removed? |
| RQ2 | How large is the gap between a random split and a chronological split, and how does performance vary by test month? |
| RQ3 | How much does acting on the model's risk scores reduce expected cost compared with acting on no bookings or on all bookings? |
| RQ4 | Which reservation-time factors are associated with cancellation, are they stable between earlier and later periods, and does a model trained on one hotel transfer to the other? |

---

# 3. Dataset
| Item | Detail |
|------|--------|
| Source | Hotel Booking Demand, Kaggle: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand |
| Original paper | Antonio, N., Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41-49. |
| Size | 119,390 bookings, 32 columns |
| Hotels | City Hotel (79,330) and Resort Hotel (40,060), both in Portugal |
| Period | Arrivals from 1 July 2015 to 31 August 2017 |
| Target | `is_canceled` (1 = cancelled or no-show, 0 = checked out), 37.04% positive |
| Identifiers | None. There is no booking ID or guest ID, and no personal information. |
| Contents | Booking timing, stay and guests, channel and market, guest history, price and deposit policy, room and requests, and the final reservation status |

Each row is one booking. Columns cover booking timing, stay and guests, channel and market, guest history, price and deposit policy, room and requests, and the final reservation status.

---

# 4. Problem Definition
| Component          | Description                                                      |
|--------------------|------------------------------------------------------------------|
| Task               | Supervised Binary classification, Booking level Churn Prediction |
| Class label        | 'is_canceled' (qualitative, asymmetric binary attribute)         |
| Positive class (1) | cancelled booking, or no show                                    |
| Negative class (0) | Checked out booking                                              |
| Prediction Point   | Reservation Time                                                 |
| Main Challenge     | Preventing Data Leakage and evaluating on a later time period, and turning risk scores into decisions    |
| Primary metric | PR-AUC (average precision) on the cancelled class |
| Phase 1 analysis   | Descriptive and Diagnostic analysis                              |
| Phase 2 analysis   | Predictive and Prescriptive analysis                             |

---

# 5. Repository contents
| File | Description |
|------|-------------|
| `Part 1_Term_Project_(8005,8010).ipynb` | Part 1 notebook: data preparation, leakage audit, EDA |
| `<Part 2 notebook name>.ipynb` | Part 2 notebook: models, evaluation, cost analysis, final pipeline |
| `Hotel_Booking_Project_Proposal__26-L__8010_8005_.pdf` | Approved project proposal |
| `hotel_bookings_cleaned.csv` | Cleaned data saved by Part 1 (created when the notebook runs) |
| `hotel_cancellation_model.joblib` | Saved final pipeline (created when Part 2 runs) |
| `README.md` | This file |

# 6. How To Run
1. Open the notebook in Google Colab, JupyterLab or Jupyter Notebook.
2. Run all cells from top to bottom in a fresh session. The dataset is downloaded with `kagglehub`, so internet access is required.
3. Install the requirements:

   ```bash
   pip install numpy pandas seaborn matplotlib scipy scikit-learn xgboost shap joblib kagglehub
   ```

4. `RANDOM_STATE = 42` is set throughout for reproducibility.
5. Part 2 repeats the Part 1 cleaning in short, so it can also run on its own.

---

# 7. Project Pipeline: 

| Step | Part | What is done |
|------|------|--------------|
| 1. Acquire | 1 | Download the data, validate the schema, keep the untouched copy in `df_raw` |
| 2. Split | 1 | 2015-2016 for train/validation (78,703 rows), 2017 for test (40,687 rows) |
| 3. Clean | 1 | Missing values, duplicates, outliers, categorical consistency |
| 4. Features | 1 | Booking-level features, with every threshold learned on 2015-2016 only |
| 5. Leakage audit | 1 | Classify every column by when it becomes known |
| 6. Preprocessing | 1 | Impute, log-transform, scale and one-hot encode, fitted on training data only |
| 7. EDA | 1 | Cancellation rates with sample sizes and 95% Wilson intervals, drift analysis |
| 8. Baselines and models | 2 | Prior rate, one-rule, logistic, L1 logistic, random forest, XGBoost |
| 9. Tuning and selection | 2 | Rolling-origin CV on PR-AUC, final model chosen on validation only |
| 10. Experiments | 2 | Leakage ablation, random vs time split, cost analysis, SHAP, stability, hotel transfer |
| 11. Final pipeline | 2 | Refit on all data, saved with `joblib`, scores new bookings into risk tiers |

---

# 8. Part 1: data preparation, leakage audit and EDA

### 8.1 Cleaning decisions

| Issue | Decision | Reason |
|-------|----------|--------|
| `company` missing (94.31%) | Filled with 0 (no company) | Missing means no company was involved |
| `agent` missing (13.69%) | Filled with 0 (no agent) | Same reason |
| `country` missing (0.41%) | "Unknown" category | Too few rows to impute |
| `children` missing (4 rows) | Filled with 0 | Most bookings have no children |
| Exact duplicates (31,994 rows, 26.8%) | **Kept**, reported both ways | No booking ID, so identical rows may be separate bookings. Dropping them lowers the cancel rate from 37.04% to 27.49%. |
| One negative ADR | Replaced by the 2015-2016 median | Impossible value |
| Extreme ADR (max 5,400) | Capped at the 2015-2016 99.9th percentile (305) | Cap learned on training period only |
| Zero guests (180) and zero nights (715) | Kept for sensitivity checks | Not clearly errors |
| `meal` "Undefined" | Merged into "SC" | Both mean no meal package |

No rows are removed, so the target distribution is unchanged.

### 8.2 Leakage audit

| Decision | Columns |
|----------|---------|
| Excluded (reveal or follow the outcome) | `reservation_status`, `reservation_status_date`, `assigned_room_type` |
| Ambiguous (may be recorded after booking) | `booking_changes`, `days_in_waiting_list`, `required_car_parking_spaces`, `total_of_special_requests` |
| Candidate, tested with and without | `adr`, `deposit_type` |
| Target | `is_canceled` |
| Not a predictor | `arrival_date_year` (defines the split) |

`reservation_status` maps one-to-one onto the target, so using it would hand the model the answer. No booking that needed parking was ever cancelled, which supports treating parking as post-booking information.

### 8.3 Engineered features

`total_nights`, `total_guests`, `is_family`, `weekend_stay`, `is_domestic`, `high_adr`, `lead_time_group`, `has_agent`, `has_company`, `was_waitlisted`, `has_previous_cancel`, `arrival_weekday`, `nights_group`, `booking_changes_group`. Thresholds, such as the ADR 75th percentile, are learned on 2015-2016 only.

### 8.4 Key Part 1 findings
1. **Imbalance:** 37.04% of bookings are cancelled. Always predicting "not cancelled" would score 62.96% accuracy, so accuracy is misleading and PR-AUC is the primary metric.
2. **Hotel:** the City Hotel cancels 41.7% of bookings and the Resort Hotel 27.8%.
3. **Lead time:** cancellations rise from 9.6% (0-7 days) to 67.7% (more than 365 days), in both hotels.
4. **Segment:** Groups cancel at 61.1%, while Direct (15.3%) and Corporate (18.7%) are low. Lead time and segment interact strongly: City Hotel Groups rise from 19% to 86% as lead time grows.
5. **Deposit type:** Non Refund bookings are cancelled 99.4% of the time. This is implausible for a refund-penalty policy and looks like a recording artefact linked to Portuguese group and agent bookings (97.2% are Portuguese).
6. **Special requests:** 47.7% cancel with no requests and 5.0% with five. The field may be recorded after booking, so it is ambiguous.
7. **Country:** Portugal is 40.7% of bookings and cancels at 56.6%. The gap remains within segments and deposit types, so country, segment and deposit type interact.
8. **Drift:** for January to August, the cancellation rate rose from 34.7% (2016) to 38.7% (2017), with shifts in segment mix, ADR and domestic share. This supports chronological validation.

---

# 9. Part 2: modelling and evaluation

### 9.1 Validation design

- **Test set:** all 2017 bookings (40,687 rows). It is never used to choose a model, a feature set or a threshold.
- **Train and validation:** 2015-2016 (78,703 rows), with three rolling-origin folds. Each fold trains on everything before a date and validates on the next three months.

| Fold | Training period | Train rows | Validation period | Validation rows |
|------|-----------------|-----------:|-------------------|----------------:|
| 1 | Jul 2015 to Mar 2016 | 32,959 | Apr to Jun 2016 | 16,198 |
| 2 | Jul 2015 to Jun 2016 | 49,157 | Jul to Sep 2016 | 15,029 |
| 3 | Jul 2015 to Sep 2016 | 64,186 | Oct to Dec 2016 | 14,517 |

- **Leakage control:** imputation, log transform, scaling and encoding live inside the pipeline, so they are refitted on each training fold.
- **Metrics:** PR-AUC (primary), ROC-AUC, F1 at a threshold chosen on validation predictions, and the Brier score for calibration.

### 9.2 Feature sets

| Set | Columns | Purpose |
|-----|--------:|---------|
| A strict | 20 | Main model: only information known at reservation time |
| A without adr | 19 | Tests dependence on price, which can be updated after booking |
| A plus deposit | 21 | Tests the suspicious `deposit_type` column |
| A paper-safe | 14 | Drops country, meal, guests, nights and weekend stay, which the data paper says may change at check-in |
| B plus ambiguous | 24 | A plus four columns that may be filled after booking |
| C leaky | 26 | B plus `assigned_room_type` and `reservation_status`, on purpose, to measure leakage |

The main set A was chosen by the timing rule, not by which set scores best.

### 9.3 Models and tuning

| Model | Tuning | Selected settings |
|-------|--------|-------------------|
| Prior rate | None | Constant score equal to the training cancel rate |
| One rule | Lead-time cut-off | Cancel if lead time is above 14 days |
| Logistic regression | None | Strict features |
| Logistic L1 | Grid over C | C = 0.1 |
| Random forest | Random search, 6 combinations | 250 trees, max depth 14, min leaf 3, `sqrt` features |
| XGBoost (default) | None | Library defaults |
| XGBoost (tuned) | Random search, 12 combinations | 300 trees, depth 3, learning rate 0.05, subsample 0.9, colsample 0.6 |

All tuning uses the rolling folds with PR-AUC as the score. No class weights are used, so the probabilities stay honest for the cost analysis.

### 9.4 Results

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 | Test Brier |
|-------|----------:|------------:|-------------:|--------:|-----------:|
| Prior rate | 0.371 | 0.387 | 0.500 | 0.000 | 0.238 |
| One rule (lead time) | 0.424 | 0.447 | 0.608 | 0.608 | 0.466 |
| Logistic regression | 0.770 | 0.768 | 0.826 | 0.675 | 0.164 |
| Logistic L1 (tuned) | 0.770 | 0.768 | 0.826 | 0.676 | 0.164 |
| **Random forest (tuned)** | **0.795** | 0.787 | 0.838 | 0.689 | 0.164 |
| XGBoost (default) | 0.771 | 0.781 | 0.834 | 0.666 | 0.161 |
| XGBoost (tuned) | 0.788 | **0.792** | **0.848** | **0.703** | **0.156** |

**Final model: random forest.** It was chosen only on the highest mean validation PR-AUC (0.795) and the best validation Brier score (0.159). Tuned XGBoost is slightly higher on the 2017 test set (0.792 against 0.787), but the test set was not used to choose. The gap between the two is smaller than the differences between folds.

Fold-level validation PR-AUC shows a seasonal effect:

| Model | Fold 1 (Apr-Jun) | Fold 2 (Jul-Sep) | Fold 3 (Oct-Dec) |
|-------|-----------------:|-----------------:|-----------------:|
| Logistic regression | 0.807 | 0.714 | 0.788 |
| Random forest | 0.846 | 0.730 | 0.809 |
| XGBoost (tuned) | 0.822 | 0.732 | 0.809 |

Every model finds July to September the hardest to predict.

### 9.5 RQ1: how much does leakage inflate the score?

Same tuned XGBoost on all six feature sets:

| Feature set | Columns | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|-------------|--------:|----------:|------------:|-------------:|
| A strict | 20 | 0.788 | 0.792 | 0.848 |
| A without adr | 19 | 0.785 | 0.787 | 0.844 |
| A plus deposit | 21 | 0.792 | 0.796 | 0.848 |
| A paper-safe | 14 | 0.694 | 0.724 | 0.782 |
| B plus ambiguous | 24 | 0.852 | 0.829 | 0.878 |
| C leaky | 26 | 1.000 | 1.000 | 1.000 |

- Leaky columns add **0.208** PR-AUC (1.000 against 0.792), which is fake performance.
- Post-booking columns (set B) add about 0.037, so they stay out of the main model.
- Deposit type adds only 0.004, because country and segment already carry most of its signal. Removing price costs only 0.005.
- Without the columns the data paper warns about, the score is 0.724, which is still far above the no-skill level of 0.387. A realistic range for a booking-time model is therefore about **0.72 to 0.79**.

---

# 10. Answers to the research questions

| RQ | Answer |
|----|--------|
| RQ1 Leakage | With reservation-time columns only, test PR-AUC is about 0.79. Adding the outcome columns gives 1.000, so leakage adds 0.21. Post-booking columns add about 0.04. Without the columns the data paper warns about, the score is 0.724. |
| RQ2 Split design | A random split overstates PR-AUC by about 0.049 (0.841 against 0.792). Performance is strongest in April to June (up to 0.86) and weakest in July and August (0.61 to 0.68). |
| RQ3 Cost | Under the assumed costs, the model saves 24.8% of the cancellation loss, compared with 24.4% for acting on every booking. It is clearly better and safer when actions are costly or work less well. |
| RQ4 Factors | Country, lead time, customer type and market segment matter most. Their ranking is stable between earlier and later periods (rank correlation 0.84), and a model trained on one hotel works on the other with a small loss in lift. |

# 11. References
1. Antonio, N., Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49.
2. Gaudreault, J. G., & Branco, P. (2024). Empirical analysis of performance assessment for imbalanced classification. Machine Learning, 113, 5533-5575.
3. Kmiecik, M., & Antonio, N. (2026). Cancel, rebook, save: Revenue leakage from price cuts in hotels. International Journal of Hospitality Management, 135.
