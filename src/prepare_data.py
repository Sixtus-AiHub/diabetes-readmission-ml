from pathlib import Path

import pandas as pd


# ============================================================
# File paths
# ============================================================

DATA_FILE = Path("data") / "raw" / "diabetic_data.csv"

MAPPING_FILE = (
    Path("data")
    / "raw"
    / "IDS_mapping.csv"
)


# ============================================================
# Load raw dataset
# ============================================================

print("Loading raw dataset...")

data = pd.read_csv(DATA_FILE)


# ============================================================
# Basic dataset information
# ============================================================

print("\nDataset shape:")
print(data.shape)


# ============================================================
# Define 30-day readmission target
# ============================================================

data["readmitted_30d"] = (
    data["readmitted"] == "<30"
).astype(int)

print("\nTarget definition:")
print("1 = readmitted within 30 days (<30)")
print("0 = not readmitted within 30 days (NO or >30)")


# ============================================================
# Target distribution
# ============================================================

print("\n30-day readmission distribution:")

print(
    data["readmitted_30d"]
    .value_counts()
)


print("\n30-day readmission proportions:")

print(
    data["readmitted_30d"]
    .value_counts(normalize=True)
)


print("\nTarget values:")

print(
    sorted(data["readmitted_30d"].unique())
)


# ============================================================
# Missing-value analysis
# ============================================================

question_mark_counts = (
    data == "?"
).sum()

nan_counts = (
    data.isna()
).sum()

missing_counts = (
    question_mark_counts
    + nan_counts
)

missing_percentages = (
    missing_counts
    / len(data)
    * 100
)


missing_summary = pd.DataFrame(
    {
        "missing_count": missing_counts,
        "missing_percentage": missing_percentages,
    }
)


missing_summary = (
    missing_summary[
        missing_summary["missing_count"] > 0
    ]
    .sort_values(
        "missing_percentage",
        ascending=False,
    )
)


print("\nMissing-value analysis:")

print(
    missing_summary
)


print("\nColumns with missing values:")

print(
    len(missing_summary)
)


# ============================================================
# Data types
# ============================================================

print("\nNumeric columns:")

numeric_columns = (
    data
    .select_dtypes(include="number")
    .columns
    .tolist()
)

print(
    numeric_columns
)


print("\nCategorical/string columns:")

categorical_columns = (
    data
    .select_dtypes(exclude="number")
    .columns
    .tolist()
)

print(
    categorical_columns
)


# ============================================================
# Unique values per column
# ============================================================

print("\nUnique values per column:")

unique_summary = (
    data
    .nunique(dropna=False)
    .sort_values(ascending=False)
)

print(
    unique_summary
)


# ============================================================
# Identifier analysis
# ============================================================

print("\nIdentifier columns:")

for column in [
    "encounter_id",
    "patient_nbr",
]:
    print(
        f"{column}: "
        f"{data[column].nunique():,} unique values "
        f"out of {len(data):,} rows"
    )


# ============================================================
# Patient encounter frequency
# ============================================================

print("\nPatient encounter frequency:")

patient_encounter_counts = (
    data["patient_nbr"]
    .value_counts()
)


print(
    "\nEncounter-count distribution per patient:"
)

print(
    patient_encounter_counts
    .value_counts()
    .sort_index()
    .head(20)
)


multiple_encounter_patients = (
    patient_encounter_counts > 1
).sum()


multiple_encounter_percentage = (
    patient_encounter_counts > 1
).mean()


print(
    f"\nPatients with multiple encounters: "
    f"{multiple_encounter_patients:,}"
)


print(
    f"Percentage of patients with multiple encounters: "
    f"{multiple_encounter_percentage:.2%}"
)


# ============================================================
# Temporal information check
# ============================================================

print("\nTemporal information check:")

date_like_columns = [
    column
    for column in data.columns
    if any(
        keyword in column.lower()
        for keyword in [
            "date",
            "time",
            "admit",
            "discharge",
        ]
    )
]


print(
    "Potential temporal columns:"
)

print(
    date_like_columns
)


# ============================================================
# Potential data leakage audit
# ============================================================

print("\nPotential data leakage audit:")

leakage_candidates = [
    "encounter_id",
    "patient_nbr",
    "discharge_disposition_id",
    "readmitted",
    "readmitted_30d",
]


print(
    "\nColumns requiring leakage review:"
)


for column in leakage_candidates:
    if column in data.columns:
        print(
            f"- {column}"
        )


# ============================================================
# Discharge disposition distribution
# ============================================================

print("\nDischarge disposition distribution:")

print(
    data["discharge_disposition_id"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# Inspect IDS mapping file
# ============================================================

print("\nIDS mapping file:")

with open(
    MAPPING_FILE,
    "r",
    encoding="utf-8",
) as mapping_file:

    mapping_lines = [
        line.strip()
        for line in mapping_file
        if line.strip()
    ]


print("\nMapping sections:")


for line in mapping_lines:
    print(line)


# ============================================================
# Admission-related categorical variables
# ============================================================

print("\nAdmission-related variables:")

for column in [
    "admission_type_id",
    "admission_source_id",
]:
    print(
        f"\n{column} distribution:"
    )

    print(
        data[column]
        .value_counts(dropna=False)
        .sort_index()
    )


# ============================================================
# Preparation summary
# ============================================================

print("\nPreparation summary:")

print(
    f"Rows: {len(data):,}"
)

print(
    f"Columns: {len(data.columns):,}"
)

print(
    f"30-day readmissions: "
    f"{data['readmitted_30d'].sum():,}"
)

print(
    f"30-day readmission rate: "
    f"{data['readmitted_30d'].mean():.2%}"
)