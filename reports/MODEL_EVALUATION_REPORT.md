# Machine Learning Evaluation Report

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
Logistic Regression

## Selection Criterion
Validation Average Precision (PR-AUC).

## Validation Performance
PR-AUC: 0.1316

## Held-Out Test Performance
- PR-AUC: 0.1469
- ROC-AUC: 0.7651
- Brier score: 0.1942
- Accuracy: 0.7039
- Precision at threshold 0.5: 0.0926
- Recall at threshold 0.5: 0.7317
- F1-score at threshold 0.5: 0.1644

## Dummy Baseline
Test PR-AUC: 0.0398

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
