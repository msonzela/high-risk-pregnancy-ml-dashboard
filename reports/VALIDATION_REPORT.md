# Model Validation and Threshold Analysis

## Project
Predicting High-Risk Pregnancy Outcomes

## Selected Model
Logistic Regression

## Dataset
Synthetic pregnancy records.

## Threshold Selection
The threshold was selected using validation F1-score.

This is an illustrative statistical objective and
does not represent a clinically approved threshold.

## Selected Threshold
0.65

## Held-Out Test Performance

| Metric | Threshold 0.50 | Threshold 0.65 |
|---|---:|---:|
| Precision | 0.0926 | 0.1500 |
| Recall | 0.7317 | 0.4390 |
| F1 | 0.1644 | 0.2236 |
| False Positives | 294 | 102 |
| False Negatives | 11 | 23 |

## Calibration Diagnostics
- Brier score: 0.1942
- Observed prevalence: 0.0398
- Mean predicted probability: 0.4082

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
