
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "src" / "models" / "artifacts"
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "hypertensive_complication"

FEATURES = [
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
    "region",
    "maternal_age_group",
    "bmi_group",
]

print("=" * 65)
print("STEP 7: MODEL VALIDATION AND THRESHOLD ANALYSIS")
print("=" * 65)

# =====================================================
# 1. LOAD SAVED MODEL AND DATA
# =====================================================

model = joblib.load(
    MODEL_DIR / "selected_model.joblib"
)

with open(
    MODEL_DIR / "model_metrics.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

validation = pd.read_csv(
    DATA_DIR / "validation.csv"
)

test = pd.read_csv(
    DATA_DIR / "test.csv"
)

X_val = validation[FEATURES]
y_val = validation[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

val_probabilities = model.predict_proba(X_val)[:, 1]

test_probabilities = model.predict_proba(X_test)[:, 1]

print("Selected model:", metadata["selected_model"])
print("Validation records:", len(validation))
print("Testing records:", len(test))


# =====================================================
# 2. THRESHOLD EVALUATION
# =====================================================

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive": int(
            predictions.sum()
        ),
    }


thresholds = np.round(
    np.arange(0.05, 0.96, 0.05),
    2
)

validation_results = pd.DataFrame([
    calculate_threshold_metrics(
        y_val,
        val_probabilities,
        threshold
    )
    for threshold in thresholds
])

validation_results.to_csv(
    REPORT_DIR / "validation_threshold_analysis.csv",
    index=False
)

# Select threshold on validation data only.
# F1 is used here as a demonstration objective,
# not as a clinically justified operating rule.
best_row = validation_results.sort_values(
    ["f1", "threshold"],
    ascending=[False, True]
).iloc[0]

selected_threshold = float(
    best_row["threshold"]
)

print("\nTHRESHOLD ANALYSIS")
print("-" * 45)

print(
    "Illustrative F1-selected threshold:",
    selected_threshold
)

print(
    "Validation F1:",
    round(float(best_row["f1"]), 4)
)


# =====================================================
# 3. HELD-OUT TEST EVALUATION
# =====================================================

default_test = calculate_threshold_metrics(
    y_test,
    test_probabilities,
    0.5
)

selected_test = calculate_threshold_metrics(
    y_test,
    test_probabilities,
    selected_threshold
)

print("\nHELD-OUT TEST COMPARISON")
print("-" * 45)

for label, result in [
    ("Default threshold", default_test),
    ("Validation-selected threshold", selected_test)
]:
    print(f"\n{label}:")
    print("Threshold:", result["threshold"])
    print("Precision:", round(result["precision"], 4))
    print("Recall:", round(result["recall"], 4))
    print("F1:", round(result["f1"], 4))
    print("False positives:", result["false_positive"])
    print("False negatives:", result["false_negative"])


# =====================================================
# 4. CALIBRATION DIAGNOSTICS
# =====================================================

brier = brier_score_loss(
    y_test,
    test_probabilities
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        test_probabilities,
        n_bins=8,
        strategy="quantile"
    )
)

print("\nCALIBRATION DIAGNOSTICS")
print("-" * 45)
print("Brier score:", round(brier, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))
print(
    "Observed outcome prevalence:",
    round(float(y_test.mean()), 4)
)
print(
    "Mean predicted probability:",
    round(float(test_probabilities.mean()), 4)
)


# =====================================================
# 5. THRESHOLD COMPARISON CHART
# =====================================================

plt.figure(figsize=(9, 6))

plt.plot(
    validation_results["threshold"],
    validation_results["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    validation_results["threshold"],
    validation_results["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    validation_results["threshold"],
    validation_results["f1"],
    marker="o",
    label="F1-score"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    color="gray",
    label="Validation-selected threshold"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Validation Threshold Analysis")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "12_threshold_analysis.png",
    dpi=200
)
plt.close()


# =====================================================
# 6. CALIBRATION CURVE
# =====================================================

plt.figure(figsize=(8, 6))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration"
)

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Selected model"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Positive Fraction")
plt.title("Calibration Curve — Synthetic Test Data")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "13_calibration_curve.png",
    dpi=200
)
plt.close()


# =====================================================
# 7. SAVE VALIDATION RESULTS
# =====================================================

results = {
    "selected_model": metadata["selected_model"],
    "dataset": "Synthetic",
    "threshold_selection_dataset": "validation",
    "threshold_selection_method": (
        "Maximum validation F1 (illustrative only)"
    ),
    "default_threshold": 0.5,
    "selected_threshold": selected_threshold,
    "default_test_metrics": default_test,
    "selected_threshold_test_metrics": selected_test,
    "test_roc_auc": float(roc_auc),
    "test_pr_auc": float(pr_auc),
    "test_brier_score": float(brier),
    "test_prevalence": float(y_test.mean()),
    "mean_test_predicted_probability": float(
        test_probabilities.mean()
    ),
    "limitations": [
        "Synthetic dataset only",
        "Threshold not clinically validated",
        "No calibration correction performed",
        "No independent external validation",
        "Not suitable for clinical decision-making"
    ]
}

with open(
    MODEL_DIR / "validation_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)


# =====================================================
# 8. GENERATE REPORT
# =====================================================

report = f"""# Model Validation and Threshold Analysis

## Project
Predicting High-Risk Pregnancy Outcomes

## Selected Model
{metadata['selected_model']}

## Dataset
Synthetic pregnancy records.

## Threshold Selection
The threshold was selected using validation F1-score.

This is an illustrative statistical objective and
does not represent a clinically approved threshold.

## Selected Threshold
{selected_threshold:.2f}

## Held-Out Test Performance

| Metric | Threshold 0.50 | Threshold {selected_threshold:.2f} |
|---|---:|---:|
| Precision | {default_test['precision']:.4f} | {selected_test['precision']:.4f} |
| Recall | {default_test['recall']:.4f} | {selected_test['recall']:.4f} |
| F1 | {default_test['f1']:.4f} | {selected_test['f1']:.4f} |
| False Positives | {default_test['false_positive']} | {selected_test['false_positive']} |
| False Negatives | {default_test['false_negative']} | {selected_test['false_negative']} |

## Calibration Diagnostics
- Brier score: {brier:.4f}
- Observed prevalence: {y_test.mean():.4f}
- Mean predicted probability: {test_probabilities.mean():.4f}

## Interpretation
The calibration curve compares predicted probabilities
with observed simulated outcome frequencies.

A calibration curve is diagnostic and does not itself
recalibrate model outputs.

## Limitations
- Synthetic dataset
- Simulated complication outcomes
- No real-world clinical validation
- No independent external validation
- No clinical threshold optimization
- Not suitable for patient care
"""

(REPORT_DIR / "VALIDATION_REPORT.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Saved:")
print("- validation_threshold_analysis.csv")
print("- validation_metrics.json")
print("- 12_threshold_analysis.png")
print("- 13_calibration_curve.png")
print("- VALIDATION_REPORT.md")
