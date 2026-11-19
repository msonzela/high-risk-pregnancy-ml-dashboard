# Data Preprocessing Report

## Project
Predicting High-Risk Pregnancy Outcomes

## Dataset
Synthetic maternal health records.

## Target
hypertensive_complication

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
| Training | 2987 | 18 | 4.05% |
| Validation | 983 | 6 | 4.07% |
| Testing | 1030 | 6 | 3.98% |

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
