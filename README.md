# Diabetes 30-Day Hospital Readmission Prediction

## Project Overview

This project analyzes hospital encounter data to investigate factors associated with 30-day hospital readmission among patients with diabetes.

**Primary question:** Can routinely collected hospital encounter information identify patients at elevated risk of readmission within 30 days?

This is a **portfolio data-science project**. The referenced research paper is used for healthcare-domain context, not as the methodology to reproduce.

The project demonstrates reproducible healthcare data analysis, EDA, leakage prevention, patient-aware evaluation, imbalanced classification, model comparison, feature selection, hyperparameter tuning, threshold analysis, model interpretation, error analysis, and model artifact creation.

> This project is for research and educational purposes. It is not a clinical decision-support system and should not be used for medical decisions.

---

## Research Context

The dataset is the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository.

The dataset is associated with:

> Strack, B., DeShazo, J. P., Gennaro, M., et al. (2014).  
> "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records."  
> *BioMed Research International*, Article ID 781670.

The paper provides healthcare-domain context for hospital readmission, HbA1c measurement, ICD-9 diagnosis codes, hospital utilization, medication information, and repeated encounters.

**This project does not reproduce the paper's methodology.** The predictive modeling approach was developed independently as a portfolio data-science project.

---

## Dataset

**Diabetes 130-US Hospitals for Years 1999-2008**

- 101,766 hospital encounters
- 50 raw columns
- 71,518 unique patients
- 130 U.S. hospitals / integrated delivery networks
- data covering 1999-2008

Raw files:

```text
data/raw/
├── diabetes_130_us_hospitals.zip
├── diabetic_data.csv
└── IDS_mapping.csv
```

Official UCI source:

https://archive.ics.uci.edu/dataset/296/diabetes%2B130-u

---

## Target Variable

The target is:

```text
readmitted_30d
```

Definition:

```text
1 = readmitted within 30 days
0 = not readmitted within 30 days
```

Created from the original `readmitted` field:

```python
data["readmitted_30d"] = (
    data["readmitted"] == "<30"
).astype(int)
```

Target distribution:

```text
0 = 90,409 encounters (88.84%)
1 = 11,357 encounters (11.16%)
```

The positive class is substantially imbalanced.

---

## Prediction Point

**Hospital discharge**

The model uses information available by discharge to estimate whether the encounter will be followed by a readmission within 30 days.

- Admission variables are available before or at admission.
- Hospital-course variables are available during the stay.
- Discharge variables are available at discharge.
- `readmitted_30d` is never used as a feature.
- `encounter_id` and `patient_nbr` are identifiers, not predictive measurements.

This prediction point is a project-specific design decision.

---

## Project Structure

```text
diabetes-readmission-ml/
├── .venv/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── final_random_forest.joblib
│   └── final_random_forest_9_features.joblib
├── notebooks/
│   └── model.ipynb
├── src/
│   ├── download_data.py
│   └── prepare_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Environment

```text
Python 3.14.6
pandas 3.0.6
numpy 2.5.3
scikit-learn 1.9.1
matplotlib 3.11.2
```

Exact dependencies are recorded in `requirements.txt`.

---

## Data Validation and Missing Data

Validation included:

- dataset shape and columns
- data types
- missing values
- unique values
- target validation
- duplicate encounter checking
- repeated-patient analysis
- categorical inspection
- numerical ranges
- outliers
- ID mapping
- leakage audit

Duplicate encounters:

```text
Duplicate encounter_id values: 0
```

Important missingness:

```text
weight              96.86%
max_glu_serum       94.75%
A1Cresult           83.28%
medical_specialty   49.08%
payer_code          39.56%
race                 2.23%
diag_3               1.40%
diag_2               0.35%
diag_1               0.02%
```

`weight` and `payer_code` were excluded based on project-specific data-science judgment.

Numeric missing values use median imputation. Categorical missing values use `"Missing"`. Preprocessing is fitted only on appropriate training data.

The integer-coded admission/discharge/source fields were decoded using `IDS_mapping.csv`. IDs 17 for admission source and 18 for discharge disposition have missing descriptions in the supplied mapping; no descriptions were invented.

---

## Initial Feature Set

The initial model candidate contained **45 features**.

### Numeric

```text
time_in_hospital
num_lab_procedures
num_procedures
num_medications
number_outpatient
number_emergency
number_inpatient
number_diagnoses
```

### Categorical

```text
race
gender
age
admission_type
admission_source
diag_1
diag_2
diag_3
max_glu_serum
A1Cresult
metformin
repaglinide
nateglinide
chlorpropamide
glimepiride
acetohexamide
glipizide
glyburide
tolbutamide
pioglitazone
rosiglitazone
acarbose
miglitol
troglitazone
tolazamide
examide
citoglipton
insulin
glyburide-metformin
glipizide-metformin
glimepiride-pioglitazone
metformin-rosiglitazone
metformin-pioglitazone
change
diabetesMed
medical_specialty
discharge_disposition
```

Excluded from predictive features:

```text
encounter_id
patient_nbr
readmitted
readmitted_30d
weight
payer_code
```

`diag_1_category` was explored but is not part of the final model.

---

## Diagnosis and HbA1c Analysis

Diagnosis variables are ICD-9 codes:

```text
diag_1 = primary diagnosis
diag_2 = additional diagnosis
diag_3 = additional diagnosis
```

A grouped `diag_1_category` representation was explored descriptively.

The public dataset does not contain continuous HbA1c measurements. `A1Cresult` is categorical:

```text
>8
>7
Norm
missing
```

No continuous HbA1c values were invented. HbA1c and medication-change analyses were descriptive and do not establish causality.

---

## Exploratory Findings

EDA covered target distribution, missingness, categorical and numerical distributions, outliers, repeated-patient structure, utilization, admissions, discharge, demographics, diagnoses, HbA1c, medications, correlations, and target relationships.

Important observations:

- 30-day readmission prevalence is 11.16%.
- `number_inpatient` has a positive descriptive relationship with readmission.
- `number_emergency` also shows positive descriptive patterns.
- `number_diagnoses` generally increases in observed readmission rate across common values.
- `time_in_hospital` shows a modest positive relationship with readmission.
- Discharge disposition shows larger differences between categories.
- Primary diagnosis groups have varying observed readmission rates.
- HbA1c categories have relatively similar observed readmission rates while missingness is substantial.

These are associations, not causal findings.

---

## Leakage Prevention and Patient-Aware Evaluation

Repeated encounters create a major risk of patient leakage.

```text
Unique patients: 71,518
Patients with multiple encounters: 16,773
Maximum encounters for one patient: 40
```

The workflow uses:

```text
StratifiedGroupKFold
n_splits = 5
shuffle = True
random_state = 42
groups = patient_nbr
```

Train/test:

```text
Training: 81,412 encounters
Test:     20,354 encounters
Shared patients: 0
```

Training/validation subdivision:

```text
Training subset: 65,129 encounters
Validation:      16,283 encounters
Shared patients: 0
```

The encounter-level target is used for stratification while `patient_nbr` defines the groups.

Forbidden predictive variables:

```text
encounter_id
patient_nbr
readmitted
readmitted_30d
```

The workflow also prevents target leakage, identifier leakage, patient leakage, preprocessing leakage, cross-fold leakage, and test-set tuning.

---

## Preprocessing

Numeric pipeline:

```text
median imputation
↓
standard scaling
```

Categorical pipeline:

```text
constant imputation with "Missing"
↓
one-hot encoding
```

`OneHotEncoder(handle_unknown="ignore")` is used.

Initial 45-feature representation:

```text
2,328 encoded features
```

Final selected representation:

```text
126 encoded features
```

---

## Class Imbalance

The positive class is only:

```text
11.16%
```

of encounters.

The final Random Forest uses:

```python
class_weight="balanced"
```

This increases the influence of the minority class during model fitting without synthetically oversampling the data.

No SMOTE or other synthetic resampling was applied.

Validation and test sets retain the natural class distribution.

---

## Model Comparison

Initial candidates:

```text
Logistic Regression
Random Forest
HistGradientBoosting
```

HistGradientBoosting was not included in the final comparison because the current one-hot preprocessing produces sparse data while the installed implementation required dense input for this configuration.

Validation ranking:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6517 | 0.2113 |
| Random Forest | 0.6731 | **0.2262** |

Because the positive class is uncommon, PR-AUC is particularly important.

---

## Random Forest Tuning

Grouped stratified 5-fold cross-validation was used.

Search space:

```python
rf_param_grid = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 5, 10],
}
```

This produced:

```text
27 combinations
5 folds
135 fits
```

Objective:

```text
Average Precision / PR-AUC
```

Best configuration:

```text
n_estimators = 500
max_depth = None
min_samples_leaf = 5
```

Best CV PR-AUC:

```text
0.2201
```

A worker warning occurred during the parallel search, but the search completed and returned a valid configuration.

---

## Feature Selection

Feature selection was performed after establishing the tuned 45-feature Random Forest reference.

### Permutation Importance

Permutation importance used:

```text
scoring = average_precision
n_repeats = 10
random_state = 42
```

The top 20 permutation-ranked features became the SFS candidate pool.

### Sequential Forward Selection

Grouped 5-fold sequential forward selection evaluated candidate features incrementally.

Best step:

```text
9 features
CV PR-AUC = 0.2243
```

Final selected features:

```text
number_inpatient
discharge_disposition
medical_specialty
admission_source
number_emergency
admission_type
num_procedures
number_diagnoses
max_glu_serum
```

Dimensionality reduction:

```text
45 → 9 raw features
2,328 → 126 processed features
```

That is:

```text
80% fewer raw features
~94.6% fewer processed features
```

### Feature Selection Caveat

Permutation importance and SFS development used the validation set. Therefore, validation metrics for the selected feature set are selection-biased.

The test set remained untouched during feature selection and was evaluated only after the 9-feature set was frozen.

---

## Final Model Configuration

```text
Model: Random Forest
n_estimators: 500
max_depth: None
min_samples_leaf: 5
class_weight: balanced
random_state: 42
threshold: 0.50
```

Final raw features:

```text
number_inpatient
discharge_disposition
medical_specialty
admission_source
number_emergency
admission_type
num_procedures
number_diagnoses
max_glu_serum
```

---

## Validation Results

9-feature model:

```text
ROC-AUC:           0.6708
PR-AUC:            0.2278
Precision:         0.1746
Recall:            0.5993
F1:                0.2704
Balanced Accuracy: 0.6217
```

45-feature reference:

```text
ROC-AUC:           0.6805
PR-AUC:            0.2240
Precision:         0.1902
Recall:            0.5388
F1:                0.2812
Balanced Accuracy: 0.6253
```

The selected model trades a small amount of ROC-AUC/F1 for higher recall, slightly higher PR-AUC, and substantially lower dimensionality.

---

## Final Held-Out Test Evaluation

The frozen 9-feature model was evaluated once on the untouched test set.

```text
ROC-AUC:           0.6726
PR-AUC:            0.2198
Precision:         0.1763
Recall:            0.6193
F1:                0.2745
Balanced Accuracy: 0.6279
```

### Comparison with 45-Feature Reference

| Metric | 45-feature test | 9-feature test |
|---|---:|---:|
| ROC-AUC | 0.6770 | 0.6726 |
| PR-AUC | 0.2088 | **0.2198** |
| Precision | 0.1868 | 0.1763 |
| Recall | 0.5299 | **0.6193** |
| F1 | 0.2762 | 0.2745 |
| Balanced Accuracy | 0.6200 | **0.6279** |

The **9-feature model is the final portfolio model**.

It provides substantially higher recall, slightly higher PR-AUC and balanced accuracy, while reducing raw features by 80% and processed features by approximately 94.6%.

The model should be viewed as a risk-screening/ranking model rather than a definitive clinical classifier.

---

## Error Analysis

Final test-set error counts:

```text
TN: 12,840
FP:  5,242
TP:  1,204
FN:  1,068
```

Key findings:

- prior inpatient utilization was a major predictive signal
- true positives generally had higher prior inpatient and emergency utilization than false negatives
- false positives and true positives had similar overall measures of hospital complexity
- age did not show a strong systematic error pattern
- SNF and home-health discharges had higher observed readmission rates but also produced substantial false positives
- observed readmission rates increased consistently across prediction-risk bands
- the highest risk band had an observed readmission rate of approximately 26%, compared with approximately 3.4% in the lowest risk band
- predicted probabilities should not be interpreted as calibrated probabilities

These observations describe associations and do not establish causality.

---

## Model Artifact and Reproducibility

Final artifact:

```text
models/final_random_forest_9_features.joblib
```

The artifact was reloaded successfully as a scikit-learn `Pipeline`.

Reload verification:

```text
Maximum absolute prediction difference: 4.44e-16
Mean absolute prediction difference:    3.75e-17
```

Predictions were numerically equivalent within `1e-12`.

The original 45-feature reference model remains preserved:

```text
models/final_random_forest.joblib
```

### Reproducibility Record

```text
Dataset:
UCI Diabetes 130-US Hospitals for Years 1999-2008

Target:
readmitted_30d = (readmitted == "<30").astype(int)

Prediction point:
Hospital discharge

Initial candidate features:
45

Final selected features:
9

Initial processed features:
2,328

Final processed features:
126

Train/test split:
StratifiedGroupKFold, 5 folds, random_state=42, groups=patient_nbr

Training:
81,412 encounters before validation subdivision

Training subset:
65,129 encounters

Validation:
16,283 encounters

Test:
20,354 encounters

Model:
Random Forest

n_estimators:
500

max_depth:
None

min_samples_leaf:
5

class_weight:
balanced

random_state:
42

Threshold:
0.50

Primary ranking metric:
PR-AUC

Secondary ranking metric:
ROC-AUC
```

Future experiments should record dataset/version, sampling strategy, split, feature construction, preprocessing, algorithm, hyperparameters, seed, evaluation protocol, and code version/commit.

---

## Current Project Status

### Completed

- environment initialized
- reproducible requirements recorded
- Git repository initialized
- UCI dataset downloaded and validated
- target created
- EDA completed
- healthcare-domain context reviewed
- diagnosis and HbA1c analyses completed
- ID mappings decoded
- feature taxonomy completed
- prediction point defined
- leakage audit completed
- repeated-patient structure analyzed
- patient-aware train/test and validation splits implemented
- preprocessing implemented
- class imbalance handled with class weighting
- Logistic Regression evaluated
- Random Forest evaluated
- threshold analysis completed
- Random Forest tuning completed
- permutation feature importance completed
- sequential forward selection completed
- final 9-feature model selected
- final held-out test evaluation completed
- final model artifact saved and verified

### Current Phase

```text
Phase 10 — Final Model / Application
```

The final feature-selected model has been evaluated and saved. The next stage is to build a user-facing application around the saved model while keeping application code separate from research/evaluation code where practical.

---

## Research Fidelity

The project distinguishes between:

```text
A = paper-stated facts and domain definitions
B = project-specific data-science decisions
C = engineering improvements
D = experimental observations
E = model-generated suggestions
```

The paper is used for healthcare-domain context, particularly HbA1c terminology and research motivation.

Project-specific decisions include:

- hospital-discharge prediction point
- patient-grouped evaluation
- candidate and selected feature sets
- exclusion of `weight` and `payer_code`
- preprocessing
- class weighting
- model comparison
- Random Forest tuning
- permutation importance
- sequential forward selection
- threshold selection
- final test protocol

---

## Limitations

- observational healthcare data
- data collected from 1999-2008
- historical clinical practice may differ from current practice
- substantial missingness
- rare categorical values
- repeated encounters for some patients
- limited clinical and social context
- de-identified public data
- observed readmission is not equivalent to all possible readmission risk
- moderate predictive performance
- precision remains limited at the selected threshold
- model performance does not establish clinical utility
- external validation has not been performed
- no temporal holdout evaluation has been performed
- feature selection used validation data during development, so selected-model validation metrics are not independent estimates

The model should not be used for clinical decision-making.

---

## Intended Use

This project is intended as a reproducible data-science portfolio project demonstrating:

- healthcare data analysis
- exploratory data analysis
- feature engineering
- feature selection
- leakage prevention
- patient-aware evaluation
- imbalanced classification
- model comparison
- hyperparameter tuning
- threshold analysis
- model interpretation
- error analysis
- reproducible experimentation
- model artifact creation

It is not intended for clinical deployment or medical decision-making.
