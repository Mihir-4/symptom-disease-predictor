"""
preprocessing.py

Dataset cleaning pipeline for the
Symptom-to-Disease Predictor project.
"""

import pandas as pd
import numpy as np

from src import config




def load_dataset():

    df = pd.read_csv(config.RAW_DATA_FILE)

    return df




def dataset_overview(df):

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nData Types:")
    print(df.dtypes.value_counts())

    print("\nTarget Column:")
    print(config.TARGET_COL)

    print("\nUnique Diseases:")
    print(df[config.TARGET_COL].nunique())




def analyze_missing_values(df):

    missing = df.isnull().sum()

    total_missing = missing.sum()

    print("\nMissing Values Analysis")
    print("-" * 40)

    print(f"Total Missing Values: {total_missing}")

    if total_missing > 0:
        print(
            missing[missing > 0]
            .sort_values(ascending=False)
        )

    return missing




def analyze_duplicates(df):

    duplicates = df.duplicated().sum()

    percentage = (
        duplicates / len(df)
    ) * 100

    print("\nDuplicate Analysis")
    print("-" * 40)

    print(
        f"Duplicate Rows: "
        f"{duplicates:,}"
    )

    print(
        f"Duplicate Percentage: "
        f"{percentage:.2f}%"
    )

    return duplicates



def get_symptom_columns(df):

    return [
        col
        for col in df.columns
        if col != config.TARGET_COL
    ]



def find_zero_variance_features(
    df,
    symptom_columns
):

    zero_variance = []

    for column in symptom_columns:

        if df[column].sum() == 0:
            zero_variance.append(column)

    print("\nZero Variance Features")
    print("-" * 40)

    print(
        f"Features To Remove: "
        f"{len(zero_variance)}"
    )

    return zero_variance


def remove_zero_variance_features(
    df,
    columns_to_remove
):

    df = df.drop(
        columns=columns_to_remove
    )

    return df


#

def filter_rare_diseases(df):

    disease_counts = (
        df[config.TARGET_COL]
        .value_counts()
    )

    valid_diseases = disease_counts[
        disease_counts
        >= config.MIN_SAMPLES_PER_DISEASE
    ].index

    removed_count = (
        len(disease_counts)
        - len(valid_diseases)
    )

    print("\nRare Disease Filtering")
    print("-" * 40)

    print(
        f"Removed Diseases: "
        f"{removed_count}"
    )

    print(
        f"Remaining Diseases: "
        f"{len(valid_diseases)}"
    )

    df = df[
        df[config.TARGET_COL]
        .isin(valid_diseases)
    ]

    return df



def optimize_memory(df):

    symptom_columns = (
        get_symptom_columns(df)
    )

    for column in symptom_columns:

        df[column] = (
            df[column]
            .astype(np.int8)
        )

    return df





def save_cleaned_dataset(df):

    df.to_csv(
        config.CLEANED_DATA_FILE,
        index=False
    )

    print(
        f"\nSaved Cleaned Dataset:"
    )
    print(
        config.CLEANED_DATA_FILE
    )



def generate_cleaning_report(
    original_rows,
    final_rows,
    zero_variance_count
):

    report = f"""
# Data Cleaning Report

## Original Dataset

Rows: {original_rows}

## Final Dataset

Rows: {final_rows}

## Features Removed

Zero Variance Features Removed:
{zero_variance_count}

## Notes

- Missing values checked.
- Duplicate rows analyzed.
- Zero variance features removed.
- Rare diseases filtered.
- Dataset optimized for memory.
"""

    with open(
        config.DATA_CLEANING_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print(
        "\nData Cleaning Report Saved."
    )




def run_preprocessing():

    print(
        "\nStarting Preprocessing..."
    )

    df = load_dataset()

    original_rows = len(df)

    dataset_overview(df)

    analyze_missing_values(df)

    analyze_duplicates(df)

    symptom_columns = (
        get_symptom_columns(df)
    )

    zero_variance_columns = (
        find_zero_variance_features(
            df,
            symptom_columns
        )
    )

    df = remove_zero_variance_features(
        df,
        zero_variance_columns
    )

    df = filter_rare_diseases(df)

    df = optimize_memory(df)

    save_cleaned_dataset(df)

    generate_cleaning_report(
        original_rows=original_rows,
        final_rows=len(df),
        zero_variance_count=len(
            zero_variance_columns
        )
    )

    print(
        "\nPreprocessing Complete."
    )


if __name__ == "__main__":

    config.create_directories()

    run_preprocessing()