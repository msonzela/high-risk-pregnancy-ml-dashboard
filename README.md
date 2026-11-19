# High-Risk Pregnancy Prediction Using Explainable Machine Learning

**An End-to-End Data Science and Machine Learning Research Project**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-149ECA)
![Research](https://img.shields.io/badge/Data-Synthetic-yellow)

## 1. Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting a simulated hypertensive pregnancy complication outcome using synthetic maternal health data.

It integrates data generation, exploratory data analysis, feature engineering, machine learning, model evaluation, explainable AI, and interactive dashboard development.

The project is an **independent Data Scientist portfolio demonstration**, designed to showcase the application of data science techniques to maternal and newborn health research.

**Disclaimer:** The dataset is entirely synthetic. The models have not been clinically validated and must not be used for diagnosis, triage, treatment, or patient care.

## 2. Project Objectives

- Generate and analyze synthetic maternal health data.
- Identify patterns and relationships associated with the simulated outcome.
- Develop and compare machine-learning classification models.
- Evaluate predictive performance using appropriate metrics.
- Apply explainable AI techniques to understand model behavior.
- Assess model validation, generalization, and limitations.
- Develop an interactive research dashboard.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python, TypeScript |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Backend | FastAPI, Pydantic |
| Frontend | React, Vite |
| Data Visualization | Recharts |
| Development | VS Code, Git, GitHub |

## 4. Machine Learning Workflow

The project follows these stages:

1. **Synthetic Data Generation:** Creation of simulated pregnancy records.
2. **Exploratory Data Analysis:** Assessment of distributions, correlations, and data quality.
3. **Data Preprocessing:** Preparation of numerical and categorical variables.
4. **Feature Engineering:** Development of additional predictors.
5. **Model Development:** Training and comparison of classification algorithms.
6. **Model Evaluation:** Assessment of discrimination, precision, recall, and classification errors.
7. **Model Validation:** Examination of model performance on separate validation and testing data.
8. **Explainable AI:** Interpretation of model behavior using SHAP.
9. **Dashboard Development:** Integration of machine learning outputs into a React application.

## 5. Model Performance

The selected model is **Logistic Regression**.

The following results were obtained from the synthetic demonstration dataset:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.765 |
| Average Precision | 0.147 |
| Recall | 73.2% |

These results describe experimental performance on simulated data. They do not establish real-world clinical predictive performance.

## 6. Interactive Dashboard

The application contains six modules.

### Overview

Displays the selected model, key evaluation metrics, project information, and research disclaimers.

### Model Performance

Provides model comparison and classification performance analysis, including confusion matrices and evaluation metrics.

### Explainable AI

Presents model interpretation outputs to help explain the relative contribution of predictive features.

### Model Validation

Displays validation results and information about the model's ability to generalize within the synthetic dataset.

### Prediction Simulator

Allows users to enter simulated maternal health characteristics and obtain an experimental model score.

**The displayed score is not a clinically validated probability of complications.**

### Research Limitations

Documents the limitations of synthetic data, experimental modelling, and the absence of real-world clinical validation.

## 7. Project Structure

```text
IHI_High_Risk_Pregnancy_Prediction/
├── backend/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.tsx
│   └── package.json
├── notebooks/
├── reports/
├── src/
│   ├── evaluation/
│   ├── features/
│   └── models/
├── requirements.txt
└── README.md
```

## 8. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/msonzela/high-risk-pregnancy-ml-dashboard.git
cd high-risk-pregnancy-ml-dashboard
```

### Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the FastAPI backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

API documentation:

http://127.0.0.1:8000/docs

### Start the React frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, normally:

http://localhost:5173

**Note:** The model prediction endpoint requires locally available trained model artifacts. If these are excluded from Git, they must be regenerated using the project's training scripts before running predictions.

## 9. Research Limitations

- The dataset is entirely synthetic.
- Relationships between predictors and outcomes were artificially generated.
- Model performance may not generalize to real maternal health datasets.
- No external clinical validation was performed.
- Probability calibration has not been established for clinical use.
- The model is not suitable for individual patient risk assessment.

## 10. Future Improvements

Potential future development includes:

- More comprehensive feature engineering.
- Additional model comparison and hyperparameter optimization.
- Improved evaluation of model calibration.
- Fairness and subgroup performance assessment.
- Reproducible model training pipelines.
- Deployment of the research dashboard.
- Validation on appropriately governed real-world data, subject to ethical and institutional approvals.

## 11. Project Context

This project was independently developed as a Data Science portfolio exercise exploring the application of machine learning and explainable AI to simulated maternal health outcomes.

It is not affiliated with, endorsed by, or deployed by Ifakara Health Institute.

## 12. Author

**Wilson Vicent**

Data Scientist | Data Analyst

[GitHub](https://github.com/msonzela) | [LinkedIn](https://www.linkedin.com/in/vicent-wilson/)

---

*Independent research demonstration · Synthetic data only · Not for clinical use*
