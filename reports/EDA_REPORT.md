# Exploratory Data Analysis Report

## Project
Predicting High-Risk Pregnancy Outcomes

## Dataset
Synthetic pregnancy dataset generated for research training.

## Dataset Overview
- Pregnancy records: 5,000
- Variables: 14
- Facilities: 30
- Regions: 5

## Outcome
- Outcome definition: Simulated hypertensive pregnancy complication
- No complication: 4,798
- Complication: 202
- Simulated complication prevalence: 4.04%

## Data Quality
- Duplicate pregnancy IDs: 0
- Duplicate rows: 0
- Total missing cells: 750

## Strongest Absolute Correlations with Outcome
previous_hypertension    0.154
systolic_bp              0.130
diastolic_bp             0.116
maternal_age             0.061
bmi                      0.051

## Important Interpretation
All records and outcome labels are synthetic.
The data-generating relationships were deliberately specified.
Observed correlations do not establish causation or
demonstrate clinical predictive validity.

## Next Step
Develop preprocessing and feature-engineering pipelines,
then evaluate models using facility-aware train/test splits.
