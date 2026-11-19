
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "synthetic_pregnancy_data.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "hypertensive_complication"

sns.set_theme(style="whitegrid")

# -----------------------------------------
# 2. LOAD DATA
# -----------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run: python src/generate_dataset.py"
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "pregnancy_id",
    "facility_id",
    "region",
    "maternal_age",
    "parity",
    "gestational_age_weeks",
    "anc_visits",
    "bmi",
    "previous_hypertension",
    "previous_adverse_outcome",
    "systolic_bp",
    "diastolic_bp",
    "haemoglobin_g_dl",
    TARGET,
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

if not df[TARGET].dropna().isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

print("\n" + "=" * 60)
print("PROJECT 2: HIGH-RISK PREGNANCY PREDICTION")
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# -----------------------------------------
# 3. DATASET OVERVIEW
# -----------------------------------------

print("\n1. DATASET OVERVIEW")
print("-" * 40)

print(f"Total pregnancy records: {len(df):,}")
print(f"Total variables: {df.shape[1]}")
print(f"Health facilities: {df['facility_id'].nunique()}")
print(f"Regions: {df['region'].nunique()}")

print("\nColumn data types:")
print(df.dtypes.to_string())

print("\nDescriptive statistics:")
summary = df.describe(include="all").transpose()
summary.to_csv(REPORT_DIR / "descriptive_statistics.csv")

# -----------------------------------------
# 4. DATA QUALITY ASSESSMENT
# -----------------------------------------

print("\n2. DATA QUALITY ASSESSMENT")
print("-" * 40)

missing = pd.DataFrame({
    "variable": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (
        df.isna().mean().values * 100
    ).round(2),
})

missing.to_csv(
    REPORT_DIR / "missing_values_summary.csv",
    index=False
)

print("\nMissing values:")
print(
    missing[missing["missing_count"] > 0].to_string(index=False)
)

duplicate_ids = df["pregnancy_id"].duplicated().sum()
duplicate_rows = df.duplicated().sum()

print(f"\nDuplicate pregnancy IDs: {duplicate_ids}")
print(f"Fully duplicated rows: {duplicate_rows}")

# Illustrative plausibility checks, not clinical diagnosis.
validation_rules = {
    "maternal_age": (15, 49),
    "gestational_age_weeks": (20, 42),
    "anc_visits": (0, 20),
    "bmi": (12, 60),
    "systolic_bp": (70, 250),
    "diastolic_bp": (40, 160),
    "haemoglobin_g_dl": (4, 20),
}

quality_results = []

for column, (minimum, maximum) in validation_rules.items():
    invalid = (
        df[column].notna()
        & ~df[column].between(minimum, maximum)
    )

    quality_results.append({
        "variable": column,
        "minimum_allowed": minimum,
        "maximum_allowed": maximum,
        "invalid_records": int(invalid.sum()),
    })

pd.DataFrame(quality_results).to_csv(
    REPORT_DIR / "data_quality_checks.csv",
    index=False
)

# -----------------------------------------
# 5. OUTCOME DISTRIBUTION
# -----------------------------------------

print("\n3. OUTCOME DISTRIBUTION")
print("-" * 40)

outcome_counts = df[TARGET].value_counts().reindex(
    [0, 1], fill_value=0
)

outcome_percentages = (
    outcome_counts / len(df) * 100
)

print(f"No complication: {outcome_counts[0]:,}")
print(f"Complication: {outcome_counts[1]:,}")
print(
    "Simulated complication prevalence:",
    f"{outcome_percentages[1]:.2f}%"
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=df,
    x=TARGET,
    order=[0, 1],
    color="#3274A1",
    ax=ax
)

ax.set_xticks([0, 1])
ax.set_xticklabels([
    "No complication",
    "Complication"
])

ax.set_title("Distribution of Simulated Pregnancy Outcomes")
ax.set_xlabel("Outcome")
ax.set_ylabel("Number of pregnancies")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "01_outcome_distribution.png",
    dpi=200
)
plt.close()

# -----------------------------------------
# 6. MATERNAL CHARACTERISTICS
# -----------------------------------------

print("\n4. MATERNAL CHARACTERISTICS")
print("-" * 40)

numeric_features = [
    "maternal_age",
    "parity",
    "gestational_age_weeks",
    "anc_visits",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "haemoglobin_g_dl",
]

fig, axes = plt.subplots(
    2, 4, figsize=(17, 9)
)

for ax, feature in zip(axes.flat, numeric_features):
    sns.histplot(
        df[feature].dropna(),
        bins=25,
        kde=True,
        ax=ax,
        color="#3274A1"
    )

    ax.set_title(feature.replace("_", " ").title())
    ax.set_xlabel("")

plt.suptitle(
    "Distribution of Maternal and Clinical Characteristics",
    fontsize=16
)

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "02_maternal_characteristics.png",
    dpi=200
)
plt.close()

# -----------------------------------------
# 7. OUTCOME COMPARISONS
# -----------------------------------------

print("\n5. COMPARISON BY PREGNANCY OUTCOME")
print("-" * 40)

grouped_summary = (
    df.groupby(TARGET)[numeric_features]
    .agg(["mean", "median", "std", "count"])
)

grouped_summary.to_csv(
    REPORT_DIR / "outcome_group_comparison.csv"
)

print(
    df.groupby(TARGET)[numeric_features]
    .mean()
    .round(2)
    .to_string()
)

comparison_features = [
    "maternal_age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
]

fig, axes = plt.subplots(
    2, 2, figsize=(12, 9)
)

for ax, feature in zip(
    axes.flat, comparison_features
):
    sns.boxplot(
        data=df,
        x=TARGET,
        y=feature,
        order=[0, 1],
        color="#3274A1",
        ax=ax
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_title(
        feature.replace("_", " ").title()
    )
    ax.set_xlabel("Simulated complication")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "03_outcome_comparisons.png",
    dpi=200
)
plt.close()

# -----------------------------------------
# 8. OUTCOME BY PREVIOUS HISTORY
# -----------------------------------------

history_features = [
    "previous_hypertension",
    "previous_adverse_outcome",
]

history_rows = []

for feature in history_features:
    rates = df.groupby(feature)[TARGET].agg(
        ["mean", "count", "sum"]
    )

    for value, row in rates.iterrows():
        history_rows.append({
            "feature": feature,
            "value": value,
            "records": int(row["count"]),
            "complications": int(row["sum"]),
            "complication_rate_pct": round(
                row["mean"] * 100, 2
            ),
        })

history_summary = pd.DataFrame(history_rows)

history_summary.to_csv(
    REPORT_DIR / "pregnancy_history_comparison.csv",
    index=False
)

print("\nPregnancy history comparisons:")
print(history_summary.to_string(index=False))

# -----------------------------------------
# 9. CORRELATION ANALYSIS
# -----------------------------------------

print("\n6. CORRELATION ANALYSIS")
print("-" * 40)

correlation_features = (
    numeric_features
    + history_features
    + [TARGET]
)

correlation_matrix = df[
    correlation_features
].corr(numeric_only=True)

correlation_matrix.to_csv(
    REPORT_DIR / "correlation_matrix.csv"
)

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax
)

ax.set_title(
    "Correlation Matrix: Maternal Characteristics and Outcome"
)

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "04_correlation_heatmap.png",
    dpi=200
)
plt.close()

target_correlations = (
    correlation_matrix[TARGET]
    .drop(TARGET)
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print("\nCorrelations with simulated outcome:")
print(target_correlations.round(3).to_string())

# -----------------------------------------
# 10. FACILITY-LEVEL EXPLORATION
# -----------------------------------------

facility_summary = (
    df.groupby("facility_id")
    .agg(
        pregnancies=("pregnancy_id", "count"),
        complications=(TARGET, "sum"),
        complication_rate=(TARGET, "mean"),
    )
    .reset_index()
)

facility_summary["complication_rate_pct"] = (
    facility_summary["complication_rate"] * 100
).round(2)

facility_summary.to_csv(
    REPORT_DIR / "facility_outcome_summary.csv",
    index=False
)

# -----------------------------------------
# 11. AUTOMATED EDA REPORT
# -----------------------------------------

report = f"""# Exploratory Data Analysis Report

## Project
Predicting High-Risk Pregnancy Outcomes

## Dataset
Synthetic pregnancy dataset generated for research training.

## Dataset Overview
- Pregnancy records: {len(df):,}
- Variables: {df.shape[1]}
- Facilities: {df['facility_id'].nunique()}
- Regions: {df['region'].nunique()}

## Outcome
- Outcome definition: Simulated hypertensive pregnancy complication
- No complication: {outcome_counts[0]:,}
- Complication: {outcome_counts[1]:,}
- Simulated complication prevalence: {outcome_percentages[1]:.2f}%

## Data Quality
- Duplicate pregnancy IDs: {duplicate_ids}
- Duplicate rows: {duplicate_rows}
- Total missing cells: {int(df.isna().sum().sum())}

## Strongest Absolute Correlations with Outcome
{target_correlations.head(5).round(3).to_string()}

## Important Interpretation
All records and outcome labels are synthetic.
The data-generating relationships were deliberately specified.
Observed correlations do not establish causation or
demonstrate clinical predictive validity.

## Next Step
Develop preprocessing and feature-engineering pipelines,
then evaluate models using facility-aware train/test splits.
"""

(REPORT_DIR / "EDA_REPORT.md").write_text(
    report, encoding="utf-8"
)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Reports saved to: {REPORT_DIR}")
print(f"Charts saved to: {FIGURE_DIR}")
print("Generated 4 visualizations and summary CSV reports.")
