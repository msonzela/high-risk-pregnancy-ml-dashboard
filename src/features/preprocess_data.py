
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "synthetic_pregnancy_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "src" / "models" / "artifacts"
REPORT_DIR = ROOT / "reports"

for directory in [PROCESSED_DIR, MODEL_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "hypertensive_complication"
GROUP = "facility_id"

RANDOM_STATE = 42


# =====================================================
# 2. LOAD AND VALIDATE DATA
# =====================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_PATH}\n"
        "Run: python src/generate_dataset.py"
    )

df = pd.read_csv(RAW_PATH)

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
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["pregnancy_id"].isna().any():
    raise ValueError("Missing pregnancy IDs detected.")

if df["pregnancy_id"].duplicated().any():
    raise ValueError("Duplicate pregnancy IDs detected.")

if df[GROUP].isna().any():
    raise ValueError("Missing facility IDs detected.")

if df[TARGET].isna().any():
    raise ValueError("Missing target values detected.")

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

print("=" * 60)
print("STEP 4: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("=" * 60)

print(f"\nOriginal records: {len(df):,}")
print(f"Original columns: {df.shape[1]}")
print(f"Facilities: {df[GROUP].nunique()}")


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

def engineer_features(data):
    """
    Create features using information available at the
    antenatal assessment.

    No outcome-derived variables are used.
    """

    result = data.copy()

    # Maternal age group
    result["maternal_age_group"] = pd.cut(
        result["maternal_age"],
        bins=[0, 19, 34, 100],
        labels=[
            "Under 20",
            "20-34",
            "35+"
        ]
    ).astype("object")

    # BMI group
    result["bmi_group"] = pd.cut(
        result["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obese"
        ],
        right=False
    ).astype("object")

    # Blood pressure difference
    result["pulse_pressure"] = (
        result["systolic_bp"]
        - result["diastolic_bp"]
    )

    # Previous pregnancy history indicator
    result["any_previous_complication"] = (
        (
            (result["previous_hypertension"] == 1)
            |
            (result["previous_adverse_outcome"] == 1)
        )
        .astype(int)
    )

    # ANC visits relative to gestational age
    result["anc_visits_per_week"] = (
        result["anc_visits"]
        / result["gestational_age_weeks"].replace(0, np.nan)
    )

    return result


df = engineer_features(df)

print("\nFeature engineering completed.")
print("Created:")
print(" - maternal_age_group")
print(" - bmi_group")
print(" - pulse_pressure")
print(" - any_previous_complication")
print(" - anc_visits_per_week")


# =====================================================
# 4. DEFINE MODEL FEATURES
# =====================================================

numeric_features = [
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
    "pulse_pressure",
    "any_previous_complication",
    "anc_visits_per_week",
]

categorical_features = [
    "region",
    "maternal_age_group",
    "bmi_group",
]

feature_columns = numeric_features + categorical_features

# Identifiers and the target are deliberately excluded.
X = df[feature_columns].copy()
y = df[TARGET].astype(int)
groups = df[GROUP]

print(f"\nModel input features: {len(feature_columns)}")


# =====================================================
# 5. FACILITY-GROUPED DATA SPLITTING
# =====================================================

# First split: 80% train+validation, 20% test
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

# Second split: 25% of remaining facilities for
# validation, giving approximately 60/20/20 overall.
second_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE + 1
)

relative_train_idx, relative_val_idx = next(
    second_splitter.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups=groups.iloc[train_val_idx]
    )
)

train_idx = train_val_idx[relative_train_idx]
val_idx = train_val_idx[relative_val_idx]

X_train = X.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_val = y.iloc[val_idx].copy()
y_test = y.iloc[test_idx].copy()

train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

# Check that both outcome classes exist in each split.
for name, labels in [
    ("train", y_train),
    ("validation", y_val),
    ("test", y_test)
]:
    if labels.nunique() != 2:
        raise ValueError(
            f"{name} split does not contain both classes."
        )

print("\nFACILITY-GROUPED SPLIT")
print("-" * 40)

for name, indices in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Testing", test_idx)
]:
    subset = df.iloc[indices]

    print(
        f"{name}: {len(subset):,} records | "
        f"{subset[GROUP].nunique()} facilities | "
        f"Outcome rate: {subset[TARGET].mean():.2%}"
    )


# =====================================================
# 6. PREPROCESSING PIPELINES
# =====================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

# Fit on training data ONLY.
preprocessor.fit(X_train)

# Transform the other splits using training-fitted rules.
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("\nPREPROCESSING RESULTS")
print("-" * 40)

print("Training shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Testing shape:", X_test_processed.shape)


# =====================================================
# 7. SAVE RAW FEATURE SPLITS
# =====================================================

# Save original feature values for model pipelines.
# Do not use already-transformed arrays to refit
# a separate preprocessing pipeline.

def save_split(indices, filename):
    subset = df.iloc[indices][
        ["pregnancy_id", "facility_id"]
        + feature_columns
        + [TARGET]
    ]

    subset.to_csv(
        PROCESSED_DIR / filename,
        index=False
    )


save_split(train_idx, "train.csv")
save_split(val_idx, "validation.csv")
save_split(test_idx, "test.csv")


# =====================================================
# 8. SAVE PREPROCESSOR
# =====================================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.joblib"
)

# Feature names after one-hot encoding
processed_feature_names = (
    preprocessor.get_feature_names_out().tolist()
)

metadata = {
    "project": "High-Risk Pregnancy Prediction",
    "dataset_type": "Synthetic",
    "target": TARGET,
    "target_definition": (
        "Simulated hypertensive pregnancy complication"
    ),
    "grouping_variable": GROUP,
    "random_state": RANDOM_STATE,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "processed_feature_names": processed_feature_names,
    "training_records": int(len(train_idx)),
    "validation_records": int(len(val_idx)),
    "testing_records": int(len(test_idx)),
    "training_facilities": len(train_groups),
    "validation_facilities": len(val_groups),
    "testing_facilities": len(test_groups),
    "training_prevalence": float(y_train.mean()),
    "validation_prevalence": float(y_val.mean()),
    "testing_prevalence": float(y_test.mean()),
    "disclaimer": (
        "Synthetic research demonstration only. "
        "Not clinically validated or suitable for patient care."
    )
}

with open(
    MODEL_DIR / "preprocessing_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)


# =====================================================
# 9. SAVE PREPROCESSING REPORT
# =====================================================

report = f"""# Data Preprocessing Report

## Project
Predicting High-Risk Pregnancy Outcomes

## Dataset
Synthetic maternal health records.

## Target
{TARGET}

## Feature Engineering
- Maternal age group
- BMI group
- Pulse pressure
- Previous complication indicator
- ANC visits per gestational week

## Missing Data
- Numeric variables: median imputation
- Categorical variables: most frequent imputation

## Encoding
- Categorical variables: one-hot encoding

## Scaling
- Numeric variables: StandardScaler

## Data Splitting
Facility-grouped train, validation and test splits.

| Dataset | Records | Facilities | Outcome Rate |
|---|---:|---:|---:|
| Training | {len(train_idx)} | {len(train_groups)} | {y_train.mean():.2%} |
| Validation | {len(val_idx)} | {len(val_groups)} | {y_val.mean():.2%} |
| Testing | {len(test_idx)} | {len(test_groups)} | {y_test.mean():.2%} |

## Leakage Prevention
- The target is excluded from model inputs.
- Facility identifiers are used for grouping, not prediction.
- Imputation, encoding and scaling are fitted on training data only.
- Validation and test facilities do not overlap with training facilities.

## Limitations
- Synthetic data-generating relationships are invented.
- The outcome is not clinically validated.
- The split does not represent prospective validation.
- No model should be used for clinical decisions.
"""

(REPORT_DIR / "PREPROCESSING_REPORT.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Processed datasets: {PROCESSED_DIR}")
print(f"Preprocessor: {MODEL_DIR / 'preprocessor.joblib'}")
print(f"Metadata: {MODEL_DIR / 'preprocessing_metadata.json'}")
print(f"Report: {REPORT_DIR / 'PREPROCESSING_REPORT.md'}")
