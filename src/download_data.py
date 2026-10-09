from pathlib import Path
import urllib.request
import zipfile

import pandas as pd


# ============================================================
# Official UCI dataset
# ============================================================

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)


# ============================================================
# Project paths
# ============================================================

DATA_DIR = Path("data") / "raw"

DATASET_FILE = DATA_DIR / "diabetes_130_us_hospitals.zip"

DATA_FILE = DATA_DIR / "diabetic_data.csv"

MAPPING_FILE = DATA_DIR / "IDS_mapping.csv"


# ============================================================
# Create raw-data directory
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Download the official UCI dataset
# ============================================================

if DATASET_FILE.exists():

    print(f"Dataset already exists: {DATASET_FILE}")

else:

    print("Downloading UCI Diabetes 130-US Hospitals dataset...")

    urllib.request.urlretrieve(
        DATASET_URL,
        DATASET_FILE
    )

    print(f"Dataset downloaded to: {DATASET_FILE}")


# ============================================================
# Inspect archive contents
# ============================================================

with zipfile.ZipFile(DATASET_FILE, "r") as archive:

    print("\nFiles contained in the UCI archive:")

    for file_name in archive.namelist():
        print(file_name)


# ============================================================
# Inspect IDS mapping file
# ============================================================

with zipfile.ZipFile(DATASET_FILE, "r") as archive:

    with archive.open("IDS_mapping.csv") as mapping_file:

        mapping_preview = mapping_file.read(3000).decode("utf-8")

    print("\nPreview of IDS_mapping.csv:")
    print(mapping_preview)


# ============================================================
# Inspect main patient-level dataset
# ============================================================

with zipfile.ZipFile(DATASET_FILE, "r") as archive:

    with archive.open("diabetic_data.csv") as data_file:

        data_preview = data_file.read(5000).decode("utf-8")

    print("\nPreview of diabetic_data.csv:")
    print(data_preview)


# ============================================================
# Extract main patient-level dataset
# ============================================================

if DATA_FILE.exists():

    print(f"\nDataset already extracted: {DATA_FILE}")

else:

    with zipfile.ZipFile(DATASET_FILE, "r") as archive:

        with archive.open("diabetic_data.csv") as source_file:

            with open(DATA_FILE, "wb") as destination_file:

                destination_file.write(source_file.read())

    print(f"\nDataset extracted to: {DATA_FILE}")


# ============================================================
# Extract ID mapping file
# ============================================================

if MAPPING_FILE.exists():

    print(f"Mapping file already extracted: {MAPPING_FILE}")

else:

    with zipfile.ZipFile(DATASET_FILE, "r") as archive:

        with archive.open("IDS_mapping.csv") as source_file:

            with open(MAPPING_FILE, "wb") as destination_file:

                destination_file.write(source_file.read())

    print(f"Mapping file extracted to: {MAPPING_FILE}")


# ============================================================
# Load the extracted patient-level dataset
# ============================================================

print("\nLoading patient-level dataset...")

data = pd.read_csv(DATA_FILE)


# ============================================================
# Basic dataset structure
# ============================================================

print("\nDataset shape:")
print(data.shape)


# ============================================================
# Column names
# ============================================================

print("\nColumn names:")

for column_number, column_name in enumerate(data.columns, start=1):

    print(f"{column_number:02d}. {column_name}")


# ============================================================
# First five rows
# ============================================================

print("\nFirst five rows:")
print(data.head())


# ============================================================
# Data types
# ============================================================

print("\nData types:")
print(data.dtypes)


# ============================================================
# Missing-value markers
# ============================================================

print("\nMissing-value markers:")

missing_markers = (
    data == "?"
).sum().sort_values(ascending=False)

print(missing_markers.head(15))


# ============================================================
# Target distribution
# ============================================================

print("\nTarget distribution:")

print(
    data["readmitted"].value_counts(
        dropna=False
    )
)


# ============================================================
# Target proportions
# ============================================================

print("\nTarget proportions:")

print(
    data["readmitted"].value_counts(
        normalize=True,
        dropna=False
    )
)