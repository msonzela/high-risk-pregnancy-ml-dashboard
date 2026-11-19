# Explainable AI Report

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

1. numeric__anc_visits_per_week
2. numeric__anc_visits
3. numeric__previous_hypertension
4. numeric__diastolic_bp
5. numeric__any_previous_complication
6. numeric__systolic_bp
7. numeric__bmi
8. numeric__previous_adverse_outcome
9. numeric__gestational_age_weeks
10. numeric__parity

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
