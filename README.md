# Diabetes 30-Day Hospital Readmission Prediction

A healthcare machine-learning portfolio project exploring whether routinely collected hospital-encounter information can help identify encounters associated with readmission within 30 days.

> **Project status:** Core analysis and model experiments completed. This project is for educational and portfolio purposes only. It is not a clinical decision-support system and must not be used to make medical decisions.

## Project at a glance

| Item | Description |
|---|---|
| Domain | Healthcare analytics and applied machine learning |
| Problem type | Imbalanced binary classification |
| Dataset | Diabetes 130-US Hospitals for Years 1999–2008 |
| Dataset size | 101,766 encounters; 71,518 unique patients |
| Prediction target | Readmission within 30 days |
| Final model | Random Forest classifier |
| Final feature set | 9 raw features |
| Primary ranking metric | Average precision (PR-AUC) |
| Test ROC-AUC | 0.6726 |
| Test PR-AUC | 0.2198 |
| Main focus | Patient-aware evaluation, leakage prevention, model comparison, feature selection, and error analysis |

## 1. Research question and motivation

Hospital readmission is an important healthcare quality and resource-planning concern. Historical encounter data may contain patterns associated with subsequent readmission, such as previous inpatient or emergency utilization, admission characteristics, and discharge information.

This project investigates the following question:

**Can routinely collected patient and hospital-encounter characteristics help distinguish encounters followed by a readmission within 30 days?**

The project is framed as a predictive analysis, not a causal study. Associations identified by the models do not establish that a feature causes readmission.

## 2. Dataset and target definition

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

- **101,766** hospital encounters
- **50** raw columns
- **71,518** unique patients
- Data from **130** U.S. hospitals / integrated delivery networks
- Records covering **1999–2008**

**Official dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

### Target variable

The binary target, `readmitted_30d`, was derived from the original `readmitted` column:

```python
data["readmitted_30d"] = (
    data["readmitted"] == "<30"
).astype(int)
```

| Value | Meaning |
|---|---|
| `1` | Readmitted within 30 days |
| `0` | Not readmitted within 30 days (`>30` or `NO`) |

The target distribution was:

| Class | Encounters | Percentage |
|---|---:|---:|
| No readmission within 30 days | 90,409 | 88.84% |
| Readmission within 30 days | 11,357 | 11.16% |

The positive class is the minority class, so accuracy alone would not be an adequate measure of model quality.

### Prediction point

The project defines the prediction point as **hospital discharge**, using information intended to be available by that point. Identifier columns (`encounter_id`, `patient_nbr`) and target-related columns (`readmitted`, `readmitted_30d`) were excluded from the predictive feature set.

Discharge-related variables can be informative at discharge, but they would not be available for a model intended to predict risk earlier in the hospital stay. Any future version must retain a clear prediction-time definition and audit every feature against it.

## 3. Workflow and technical approach

The project was developed as a sequence of data-science experiments rather than a single model-fitting exercise.

### Data validation and exploratory data analysis

The analysis included:

- Dataset dimensions, column types, unique values, and numerical ranges
- Missing-value analysis and categorical-value inspection
- Duplicate encounter checks
- Repeated-patient analysis
- Outlier and distribution inspection
- Target distribution and class imbalance
- Analysis of hospital utilization, diagnoses, admission and discharge characteristics
- Descriptive analysis of HbA1c-result categories and medication-related variables
- Review of the supplied ID mapping file
- A predictive-feature and leakage audit

The dataset contains substantial missingness in several variables. Examples include `weight` (96.86% missing), `max_glu_serum` (94.75%), `A1Cresult` (83.28%), and `medical_specialty` (49.08%). The project excluded `weight` and `payer_code` based on project-specific judgment. Missing numeric values were imputed using the median, while categorical missing values were represented by `"Missing"`.

The dataset contains a categorical `A1Cresult` field, not continuous HbA1c measurements. The analysis did not invent continuous values from those categories.

### Preprocessing

A scikit-learn pipeline was used to keep preprocessing within model fitting.

**Numeric features**
- Median imputation
- Standard scaling

**Categorical features**
- Constant imputation with `"Missing"`
- One-hot encoding with `handle_unknown="ignore"`

The initial 45-feature representation expanded to 2,328 encoded features after preprocessing. The final nine-feature representation expanded to 126 encoded features.

### Patient-aware splitting and leakage prevention

The dataset contains repeated encounters for some patients:

- Unique patients: **71,518**
- Patients with multiple encounters: **16,773**
- Maximum encounters for one patient: **40**

To reduce the risk of patient leakage, the workflow used `StratifiedGroupKFold` with `patient_nbr` as the grouping variable. The recorded train/test split contained 81,412 training encounters and 20,354 test encounters, with zero patients shared between those partitions. The training data was further divided into a training subset of 65,129 encounters and a validation set of 16,283 encounters, also with zero shared patients.

Grouping by patient is important because encounters from the same patient may share information. If a patient's records appear in both training and evaluation sets, performance estimates can be overly optimistic.

### Class imbalance and evaluation metrics

The positive class accounts for approximately 11.16% of encounters. The final Random Forest used `class_weight="balanced"` rather than synthetic oversampling.

The evaluation included:

- Average precision / PR-AUC
- ROC-AUC
- Precision
- Recall
- F1 score
- Balanced accuracy
- Confusion-matrix and error analysis

PR-AUC was used as the primary model-ranking metric because the positive class is uncommon. Metrics at the 0.50 classification threshold were also examined, while recognizing that a threshold should ultimately depend on the intended use and the relative costs of false positives and false negatives.

## 4. Model development and selection

### Initial model comparison

The initial candidates included Logistic Regression and Random Forest. HistGradientBoosting was explored but was not included in the final comparison because the current one-hot-encoded preprocessing produced sparse input that did not fit the tested configuration.

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6517 | 0.2113 |
| Random Forest | 0.6731 | 0.2262 |

The Random Forest achieved the higher validation PR-AUC among these two candidates.

### Random Forest hyperparameter tuning

A grid search evaluated 27 hyperparameter combinations using five-fold grouped stratified cross-validation, for 135 model fits. The search optimized average precision.

The selected configuration was:

```python
{
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
}
```

The best cross-validation PR-AUC recorded during the search was **0.2201**.

### Feature selection

Feature selection was explored to reduce dimensionality and assess whether a smaller set of raw features could retain useful predictive performance.

1. Permutation importance ranked candidate raw features using average precision.
2. The top 20 candidates were passed to sequential forward selection.
3. Grouped five-fold cross-validation evaluated candidate subsets.
4. The best recorded subset contained nine raw features, with a cross-validation PR-AUC of 0.2243.

The selected features were:

1. `number_inpatient`
2. `discharge_disposition`
3. `medical_specialty`
4. `admission_source`
5. `number_emergency`
6. `admission_type`
7. `num_procedures`
8. `number_diagnoses`
9. `max_glu_serum`

This reduced the raw feature count from 45 to 9 (**80% fewer raw features**) and the encoded feature count from 2,328 to 126 (**approximately 94.6% fewer encoded features**).

> **Selection caveat:** Permutation importance and feature selection used validation data, and the nine-feature and 45-feature models were subsequently compared on the test set. As a result, the current test results should be treated as a useful project evaluation, but not as a fully untouched estimate after all model-selection decisions. A future rigorous evaluation should reserve a new holdout set or use a carefully designed nested cross-validation procedure.

## 5. Final model performance

The final portfolio model is a Random Forest trained on the nine selected raw features. The reported classification threshold is 0.50.

### Held-out test metrics

| Metric | Result |
|---|---:|
| ROC-AUC | 0.6726 |
| PR-AUC / average precision | 0.2198 |
| Precision | 0.1763 |
| Recall | 0.6193 |
| F1 score | 0.2745 |
| Balanced accuracy | 0.6279 |

The test-set positive-class prevalence was approximately 11.16%. The observed PR-AUC of 0.2198 is above that baseline prevalence, suggesting some ability to rank positive cases above negative cases. The ROC-AUC indicates moderate discrimination rather than strong separation.

At the 0.50 threshold, recall of 61.93% means the model identified about 62% of positive encounters in this evaluation. Precision of 17.63% means that fewer than one in five flagged encounters was positive. This is a substantial false-positive burden and is an important limitation, not a result to hide.

### Comparison with the 45-feature reference model

| Metric | 45-feature test result | 9-feature test result |
|---|---:|---:|
| ROC-AUC | 0.6770 | 0.6726 |
| PR-AUC | 0.2088 | 0.2198 |
| Precision | 0.1868 | 0.1763 |
| Recall | 0.5299 | 0.6193 |
| F1 score | 0.2762 | 0.2745 |
| Balanced accuracy | 0.6200 | 0.6279 |

In the recorded comparison, the nine-feature model had higher PR-AUC, recall, and balanced accuracy, with a small reduction in ROC-AUC and F1. It also used substantially fewer raw and encoded features. These results do not establish that the smaller model is clinically superior; they describe the trade-offs observed in this experiment.

## 6. Interpretation and error analysis

Exploratory analysis and model-error review found descriptive patterns including:

- Prior inpatient utilization was an important predictive signal.
- Prior emergency utilization was positively associated with observed readmission patterns.
- Readmission rates varied across discharge dispositions and primary diagnosis groups.
- Observed readmission rates increased across the model's risk bands in the recorded analysis.
- The highest risk band had an observed readmission rate of approximately 26%, compared with approximately 3.4% in the lowest risk band.

These are descriptive associations. They do not establish causality, guarantee performance in another population, or show that acting on the predictions would improve outcomes. Predicted scores should not be interpreted as calibrated probabilities; calibration would require a separate assessment.

## 7. Reproducibility and project artifacts

The project includes scripts, notebooks, dependency information, and saved model artifacts.

The final nine-feature model was saved as:

```text
models/final_random_forest_9_features.joblib
```

The 45-feature reference model was also preserved:

```text
models/final_random_forest.joblib
```

The saved nine-feature artifact was successfully loaded as a scikit-learn `Pipeline`. However, the notebook's exact element-by-element comparison of predictions before and after saving returned `False`. Exact floating-point equality can be too strict, but this result should be investigated with a numerical-tolerance check before claiming that prediction equivalence has been verified.

The environment's dependencies are recorded in `requirements.txt`. For reproducible results, users should also record the dataset version, split strategy, feature construction, preprocessing, model parameters, random seeds, evaluation protocol, and code revision.

### Repository structure

The repository is organized around data, notebooks, source scripts, model artifacts, and dependency documentation. A representative structure is:

```text
diabetes-readmission-ml/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── final_random_forest.joblib
│   └── final_random_forest_9_features.joblib
├── notebooks/
│   ├── diabetes_readmission_analysis.ipynb
│   └── model.ipynb
├── src/
│   ├── download_data.py
│   └── prepare_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

This is a representative structure; update it if the repository contents differ. The local `.venv/` environment should generally remain excluded from version control. Check GitHub's file-size limits and consider Git LFS or release assets if model files make the repository difficult to clone.

## 8. Skills demonstrated

| Skill area | Evidence from the project |
|---|---|
| Python data analysis | Dataset validation, transformation, aggregation, and analysis |
| pandas and NumPy | Data inspection, missingness analysis, feature preparation, and metric calculations |
| Exploratory data analysis | Class distribution, feature distributions, missing values, and target relationships |
| Data preprocessing | Separate numeric and categorical pipelines, imputation, scaling, and encoding |
| scikit-learn pipelines | Reusable preprocessing and modeling workflow |
| Supervised machine learning | Logistic Regression and Random Forest classification |
| Imbalanced classification | Class weighting and PR-AUC-centered model comparison |
| Model evaluation | ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, and confusion matrix |
| Cross-validation and tuning | Grouped stratified cross-validation and grid search |
| Leakage awareness | Exclusion of identifiers/target columns and patient-grouped evaluation |
| Feature selection | Permutation importance and sequential forward selection |
| Model interpretation | Feature analysis, risk-band review, and error analysis |
| Reproducibility practices | Dependency file, saved pipeline artifacts, and model reload checks |
| Scientific reasoning | Clear research question, distinction between association and causation, and documented limitations |

The project demonstrates applied practice in these areas; it should not be interpreted as proof of production-level clinical ML expertise.

## 9. Limitations and responsible use

Important limitations include:

- The data covers 1999–2008 and may not reflect current clinical practice.
- Several variables have substantial missingness.
- The dataset has limited clinical and social context.
- The target represents observed readmission in this dataset, not every possible measure of readmission risk.
- Predictive performance is moderate and precision is limited at the selected threshold.
- The current project has no external validation or temporal holdout evaluation.
- Model selection involved validation-based feature selection and comparison of test-set results; the reported test metrics are therefore not a fully independent final estimate.
- Calibration and clinical utility have not been established.
- Discharge-related features are only appropriate for a prediction made at discharge.
- Associations found in observational data do not imply causality.

**This model is for education and portfolio demonstration only. It must not be used for diagnosis, treatment, discharge decisions, or other clinical decision-making.** External validation, careful calibration, clinical review, fairness analysis, and evaluation of real-world utility would be required before any clinical application could be considered.

## 10. Future improvements

Potential next steps include:

1. Resolve the saved-model prediction-equivalence check using numerical tolerances and investigate any meaningful discrepancy.
2. Strengthen evaluation with a new untouched holdout set or nested cross-validation.
3. Assess probability calibration and document threshold-selection criteria.
4. Add confidence intervals or variability estimates for key metrics.
5. Evaluate performance across relevant patient subgroups, with careful attention to sample size and fairness.
6. Add a clean inference script that validates input columns and uses the saved pipeline.
7. If developing a demonstration application, keep application code separate from training and evaluation code, and clearly label it as educational.
8. Document the environment setup, data preparation, notebook execution order, and expected outputs.
9. Explore whether the research question can be refined into a more focused, reproducible healthcare analysis.

## 11. Research context

The dataset is associated with the following publication:

Strack, B., DeShazo, J. P., Gennaro, M., et al. (2014). “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.” *BioMed Research International*, Article ID 781670.

The paper provides healthcare-domain context for readmission, HbA1c measurement, diagnoses, medication information, and repeated encounters. **This portfolio project does not claim to reproduce the paper's methodology or results.** The modeling workflow and feature-selection decisions described here are project-specific.

## Conclusion

This project represents an end-to-end applied machine-learning exercise using a large, imperfect healthcare dataset. It goes beyond fitting a classifier by examining missingness, repeated-patient structure, leakage risks, class imbalance, model-selection trade-offs, feature reduction, and errors.

The results suggest that the selected model contains some ability to rank encounters by observed 30-day readmission outcome, but its moderate discrimination and low precision limit the claims that can be made. The most valuable outcome is the documented workflow and the practical lessons it provides for further work in healthcare analytics and responsible machine learning.
