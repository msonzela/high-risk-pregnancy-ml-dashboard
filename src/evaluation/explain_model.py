
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "src" / "models" / "artifacts"
    / "selected_model.joblib"
)

METRICS_PATH = (
    ROOT / "src" / "models" / "artifacts"
    / "model_metrics.json"
)

TEST_PATH = ROOT / "data" / "processed" / "test.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load model and evaluation metadata
pipeline = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

selected_model = metadata["selected_model"]

print("=" * 60)
print("STEP 6: EXPLAINABLE AI")
print("=" * 60)
print("Selected model:", selected_model)

if selected_model != "Logistic Regression":
    raise ValueError(
        "This script expects the selected Logistic "
        "Regression model."
    )

test = pd.read_csv(TEST_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

feature_columns = (
    metadata.get("features")
    or [
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
)

X_test = test[feature_columns].copy()

X_transformed = preprocessor.transform(X_test)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(
    X_transformed,
    dtype=float
)

feature_names = (
    preprocessor.get_feature_names_out().tolist()
)

# =====================================================
# 1. MODEL COEFFICIENTS
# =====================================================

coefficients = classifier.coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_coefficient": np.abs(coefficients),
})

importance = importance.sort_values(
    "absolute_coefficient",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance.head(15).iloc[::-1]

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["feature"],
    top_features["coefficient"]
)

plt.axvline(0, color="gray", linewidth=1)
plt.xlabel("Logistic Regression Coefficient")
plt.title("Model Coefficients — Synthetic Dataset")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_feature_coefficients.png",
    dpi=200
)
plt.close()

# =====================================================
# 2. SHAP EXPLANATIONS
# =====================================================

# For a linear model, explain the model's log-odds.
# Background data must come from training records only.

train = pd.read_csv(
    ROOT / "data" / "processed" / "train.csv"
)

X_train = train[feature_columns]

background = preprocessor.transform(X_train)

if hasattr(background, "toarray"):
    background = background.toarray()

background = np.asarray(background, dtype=float)

rng = np.random.default_rng(42)

background_idx = rng.choice(
    len(background),
    size=min(200, len(background)),
    replace=False
)

background_sample = background[background_idx]

explainer = shap.LinearExplainer(
    classifier,
    background_sample
)

shap_values = explainer.shap_values(X_transformed)

if isinstance(shap_values, list):
    shap_values = shap_values[-1]

shap_values = np.asarray(shap_values)

# =====================================================
# 3. GLOBAL SHAP FEATURE IMPORTANCE
# =====================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": mean_abs_shap,
})

shap_importance = shap_importance.sort_values(
    "mean_absolute_shap",
    ascending=False
)

shap_importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

top_shap = shap_importance.head(15).iloc[::-1]

plt.figure(figsize=(10, 7))
plt.barh(
    top_shap["feature"],
    top_shap["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value (Log-Odds)")
plt.title("Global SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "10_shap_importance.png",
    dpi=200
)
plt.close()

# =====================================================
# 4. SHAP SUMMARY PLOT
# =====================================================

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False,
    max_display=15
)

plt.title("SHAP Summary — Synthetic Dataset")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "11_shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# =====================================================
# 5. INTERPRETATION REPORT
# =====================================================

top_names = shap_importance.head(10)["feature"].tolist()

report = """# Explainable AI Report

## Project
Predicting High-Risk Pregnancy Outcomes

## Dataset
Synthetic pregnancy records.

## Selected Model
Logistic Regression

## Explanation Method
SHAP LinearExplainer.

SHAP values explain contributions to the model's
log-odds output, not direct clinical probabilities.

## Top Model Features
"""

for index, feature in enumerate(top_names, start=1):
    report += f"\n{index}. {feature}"

report += """

## Interpretation

Feature importance indicates how strongly a variable
contributes to model predictions across the synthetic
test dataset.

Positive and negative SHAP values indicate movement
of the model output relative to the background
reference prediction.

SHAP does not establish causality.

## Limitations

- Synthetic records only
- Simulated outcomes
- No real-world clinical validation
- No independent external validation
- Model probability calibration not established
- Not suitable for clinical decision-making
"""

(REPORT_DIR / "SHAP_EXPLANATION_REPORT.md").write_text(
    report,
    encoding="utf-8"
)

print("\nSTEP 6 COMPLETED SUCCESSFULLY")
print("Generated:")
print("- feature_importance.csv")
print("- shap_feature_importance.csv")
print("- 09_feature_coefficients.png")
print("- 10_shap_importance.png")
print("- 11_shap_summary.png")
print("- SHAP_EXPLANATION_REPORT.md")
