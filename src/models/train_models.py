
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "src" / "models" / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for directory in [ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "hypertensive_complication"

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "region",
    "maternal_age_group",
    "bmi_group",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =====================================================
# 2. LOAD TRAIN / VALIDATION / TEST
# =====================================================

def load_split(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing first."
        )

    return pd.read_csv(path)


train = load_split("train.csv")
validation = load_split("validation.csv")
test = load_split("test.csv")

for name, data in [
    ("training", train),
    ("validation", validation),
    ("testing", test),
]:
    required = FEATURES + [TARGET, "facility_id"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            f"{name} data is missing columns: {missing}"
        )

    if data[TARGET].isna().any():
        raise ValueError(f"Missing targets in {name}.")

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(f"Invalid targets in {name}.")

    if data[TARGET].nunique() != 2:
        raise ValueError(
            f"Both outcome classes required in {name}."
        )

# Confirm facilities do not overlap.
train_facilities = set(train["facility_id"])
val_facilities = set(validation["facility_id"])
test_facilities = set(test["facility_id"])

assert train_facilities.isdisjoint(val_facilities)
assert train_facilities.isdisjoint(test_facilities)
assert val_facilities.isdisjoint(test_facilities)

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_val = validation[FEATURES]
y_val = validation[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

print("=" * 65)
print("STEP 5: MACHINE LEARNING MODEL DEVELOPMENT")
print("=" * 65)

for name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test),
]:
    print(
        f"{name}: {len(data):,} records | "
        f"{data['facility_id'].nunique()} facilities | "
        f"Positive rate: {data[TARGET].mean():.2%}"
    )


# =====================================================
# 3. BUILD LEAKAGE-SAFE PREPROCESSING
# =====================================================

def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        ),
    ])


# =====================================================
# 4. DEFINE MACHINE LEARNING MODELS
# =====================================================

models = {
    "Dummy Baseline": DummyClassifier(
        strategy="prior"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4
    ),
}


# =====================================================
# 5. EVALUATION FUNCTION
# =====================================================

def evaluate_model(y_true, probabilities, threshold=0.5):
    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid probabilities.")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    }


# =====================================================
# 6. TRAIN AND VALIDATE
# =====================================================

trained_pipelines = {}
validation_results = {}
validation_probabilities = {}

for name, estimator in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", estimator)
    ])

    # The pipeline fits imputation, scaling and
    # encoding using training records only.
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    metrics = evaluate_model(
        y_val,
        probabilities,
        threshold=0.5
    )

    trained_pipelines[name] = pipeline
    validation_results[name] = metrics
    validation_probabilities[name] = probabilities

    print(f"  PR-AUC: {metrics['average_precision']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")


# =====================================================
# 7. SELECT BEST MODEL USING VALIDATION PR-AUC
# =====================================================

candidate_names = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]

best_model_name = max(
    candidate_names,
    key=lambda name: validation_results[name][
        "average_precision"
    ]
)

best_pipeline = trained_pipelines[best_model_name]

print("\n" + "=" * 65)
print("VALIDATION MODEL SELECTION")
print("=" * 65)
print(f"Selected candidate: {best_model_name}")
print(
    "Validation PR-AUC:",
    round(
        validation_results[best_model_name][
            "average_precision"
        ], 4
    )
)

baseline_pr_auc = validation_results[
    "Dummy Baseline"
]["average_precision"]

if (
    validation_results[best_model_name]["average_precision"]
    <= baseline_pr_auc
):
    warnings.warn(
        "Selected model did not outperform the "
        "dummy baseline on validation PR-AUC."
    )


# =====================================================
# 8. HELD-OUT TEST EVALUATION
# =====================================================

# The test set is used only after model selection.
# Keep the model trained on training data only.
test_probabilities = best_pipeline.predict_proba(
    X_test
)[:, 1]

test_metrics = evaluate_model(
    y_test,
    test_probabilities,
    threshold=0.5
)

# Evaluate dummy baseline on test for comparison.
dummy_probabilities = trained_pipelines[
    "Dummy Baseline"
].predict_proba(X_test)[:, 1]

dummy_test_metrics = evaluate_model(
    y_test,
    dummy_probabilities,
    threshold=0.5
)

print("\n" + "=" * 65)
print("HELD-OUT TEST PERFORMANCE")
print("=" * 65)

for key in [
    "average_precision",
    "roc_auc",
    "brier_score",
    "accuracy",
    "precision",
    "recall",
    "f1",
]:
    print(f"{key}: {test_metrics[key]:.4f}")

print("\nConfusion matrix:")
print(json.dumps(test_metrics["confusion_matrix"], indent=2))

print(
    "\nDummy test PR-AUC:",
    round(dummy_test_metrics["average_precision"], 4)
)


# =====================================================
# 9. SAVE MODEL AND METRICS
# =====================================================

model_path = ARTIFACT_DIR / "selected_model.joblib"

joblib.dump(best_pipeline, model_path)

results = {
    "project": "Predicting High-Risk Pregnancy Outcomes",
    "dataset": "Synthetic demonstration dataset",
    "target": TARGET,
    "selected_model": best_model_name,
    "selection_metric": "validation_average_precision",
    "decision_threshold": 0.5,
    "training_records": len(train),
    "validation_records": len(validation),
    "testing_records": len(test),
    "training_prevalence": float(y_train.mean()),
    "validation_prevalence": float(y_val.mean()),
    "testing_prevalence": float(y_test.mean()),
    "validation_results": validation_results,
    "selected_model_test_results": test_metrics,
    "dummy_test_results": dummy_test_metrics,
    "limitations": [
        "All data and outcomes are synthetic.",
        "Data-generating relationships are invented.",
        "Model performance does not establish clinical validity.",
        "No independent real-world external validation.",
        "Probability calibration has not been corrected.",
        "Threshold 0.5 is illustrative, not clinically optimized.",
        "Not suitable for patient care or clinical decisions."
    ]
}

metrics_path = ARTIFACT_DIR / "model_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

comparison = pd.DataFrame([
    {
        "model": name,
        **{
            k: v for k, v in metrics.items()
            if k != "confusion_matrix"
        }
    }
    for name, metrics in validation_results.items()
])

comparison = comparison.sort_values(
    "average_precision",
    ascending=False
)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)


# =====================================================
# 10. MODEL COMPARISON CHART
# =====================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    comparison["model"],
    comparison["average_precision"]
)

ax.axhline(
    y_val.mean(),
    linestyle="--",
    color="gray",
    label="Validation prevalence (random ranking reference)"
)

ax.set_ylabel("Average Precision (PR-AUC)")
ax.set_title("Validation Model Comparison")
ax.tick_params(axis="x", rotation=15)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "05_model_comparison.png",
    dpi=200
)
plt.close()


# =====================================================
# 11. PRECISION-RECALL CURVES
# =====================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name in candidate_names:
    precision, recall, _ = precision_recall_curve(
        y_val,
        validation_probabilities[name]
    )

    ap = validation_results[name]["average_precision"]

    ax.plot(
        recall,
        precision,
        label=f"{name} (AP={ap:.3f})"
    )

ax.axhline(
    y_val.mean(),
    linestyle="--",
    color="gray",
    label="Validation prevalence"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Validation Precision-Recall Curves")
ax.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "06_precision_recall_curves.png",
    dpi=200
)
plt.close()


# =====================================================
# 12. ROC CURVES
# =====================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name in candidate_names:
    fpr, tpr, _ = roc_curve(
        y_val,
        validation_probabilities[name]
    )

    auc = validation_results[name]["roc_auc"]

    ax.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random ranking"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Validation ROC Curves")
ax.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "07_roc_curves.png",
    dpi=200
)
plt.close()


# =====================================================
# 13. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    (test_probabilities >= 0.5).astype(int),
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No", "Yes"])
ax.set_yticklabels(["No", "Yes"])

ax.set_xlabel("Predicted complication")
ax.set_ylabel("Simulated outcome")
ax.set_title(f"Test Confusion Matrix: {best_model_name}")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "08_confusion_matrix.png",
    dpi=200
)
plt.close()


# =====================================================
# 14. SAVE TECHNICAL REPORT
# =====================================================

report = f"""# Machine Learning Evaluation Report

## Project
Predicting High-Risk Pregnancy Outcomes

## Dataset
Synthetic antenatal pregnancy records.

## Prediction Target
Simulated hypertensive pregnancy complication.

## Model Development
- Logistic Regression
- Random Forest
- XGBoost
- Dummy Classifier baseline

## Validation Strategy
Facility-grouped train, validation and test datasets.

## Selected Model
{best_model_name}

## Selection Criterion
Validation Average Precision (PR-AUC).

## Validation Performance
PR-AUC: {validation_results[best_model_name]['average_precision']:.4f}

## Held-Out Test Performance
- PR-AUC: {test_metrics['average_precision']:.4f}
- ROC-AUC: {test_metrics['roc_auc']:.4f}
- Brier score: {test_metrics['brier_score']:.4f}
- Accuracy: {test_metrics['accuracy']:.4f}
- Precision at threshold 0.5: {test_metrics['precision']:.4f}
- Recall at threshold 0.5: {test_metrics['recall']:.4f}
- F1-score at threshold 0.5: {test_metrics['f1']:.4f}

## Dummy Baseline
Test PR-AUC: {dummy_test_metrics['average_precision']:.4f}

## Interpretation
Results measure performance on synthetic records.
They do not demonstrate clinical prediction capability.

## Limitations
- Synthetic data only
- Invented relationships and outcome labels
- No prospective or external clinical validation
- No calibration correction
- Threshold 0.5 is not clinically optimized
- Not suitable for individual patient care

## Next Step
Explainable AI and model interpretation using SHAP.
"""

(REPORT_DIR / "MODEL_EVALUATION_REPORT.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Selected model: {best_model_name}")
print(f"Model saved: {model_path}")
print(f"Metrics saved: {metrics_path}")
print(f"Reports saved: {REPORT_DIR}")
print(f"Charts saved: {FIGURE_DIR}")
