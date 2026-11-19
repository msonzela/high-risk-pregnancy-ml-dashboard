
from pathlib import Path
from typing import Any, Dict

import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT / "src" / "models" / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "selected_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
VALIDATION_PATH = ARTIFACT_DIR / "validation_metrics.json"

MODEL = None
MODEL_LOAD_ERROR = None


# ==========================================
# INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="High-Risk Pregnancy Prediction API",
    description=(
        "Experimental research API using synthetic "
        "pregnancy data. Not clinically validated."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==========================================
# MODEL LOADING
# ==========================================

def get_model():
    global MODEL, MODEL_LOAD_ERROR

    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Saved model file was not found."
        )

    try:
        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOAD_ERROR = None
        return MODEL

    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        raise HTTPException(
            status_code=503,
            detail="Unable to load saved model."
        )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} is not available."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, ValueError):
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read {path.name}."
        )


DISCLAIMER = (
    "Experimental model trained on synthetic data. "
    "The score is not a clinically validated "
    "probability of pregnancy complications. "
    "Not suitable for diagnosis, triage, "
    "treatment, or patient care."
)


# ==========================================
# INPUT SCHEMA
# ==========================================

class PregnancyInput(BaseModel):
    maternal_age: float = Field(..., ge=10, le=60)
    parity: int = Field(..., ge=0, le=20)
    gestational_age_weeks: float = Field(
        ..., ge=20, le=45
    )
    anc_visits: int = Field(..., ge=0, le=30)
    bmi: float = Field(..., ge=10, le=65)

    previous_hypertension: int = Field(
        ..., ge=0, le=1
    )
    previous_adverse_outcome: int = Field(
        ..., ge=0, le=1
    )

    systolic_bp: float = Field(..., ge=60, le=250)
    diastolic_bp: float = Field(..., ge=30, le=160)

    haemoglobin_g_dl: float = Field(
        ..., ge=3, le=22
    )

    region: str
    maternal_age_group: str
    bmi_group: str


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "project": "High-Risk Pregnancy Prediction",
        "version": "1.0.0",
        "status": "API running",
        "data_type": "Synthetic",
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_file_exists": MODEL_PATH.exists(),
        "metrics_file_exists": METRICS_PATH.exists(),
        "validation_file_exists": VALIDATION_PATH.exists(),
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/api/model/info")
def model_info():
    return {
        "project": "Predicting High-Risk Pregnancy Outcomes",
        "model_type": "Logistic Regression",
        "target": "Simulated hypertensive pregnancy complication",
        "dataset": "Synthetic demonstration data",
        "clinical_validation": False,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/model/metrics")
def model_metrics():
    return read_json(METRICS_PATH)


@app.get("/api/model/validation")
def model_validation():
    return read_json(VALIDATION_PATH)


# ==========================================
# MODEL FEATURES
# ==========================================

@app.get("/api/model/features")
def model_features():
    return {
        "numeric_inputs": [
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
        ],
        "categorical_inputs": [
            "region",
            "maternal_age_group",
            "bmi_group",
        ],
        "derived_features": [
            "pulse_pressure",
            "any_previous_complication",
            "anc_visits_per_week",
        ],
    }


# ==========================================
# EXPERIMENTAL PREDICTION
# ==========================================

@app.post("/api/model/predict")
def predict(data: PregnancyInput):

    model = get_model()

    record = data.dict()

    # Recreate derived features used during training.
    record["pulse_pressure"] = (
        record["systolic_bp"] -
        record["diastolic_bp"]
    )

    record["any_previous_complication"] = int(
        record["previous_hypertension"] == 1
        or record["previous_adverse_outcome"] == 1
    )

    record["anc_visits_per_week"] = (
        record["anc_visits"] /
        max(record["gestational_age_weeks"], 1)
    )

    frame = pd.DataFrame([record])

    try:
        if not hasattr(model, "predict_proba"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Saved model does not support "
                    "probability scoring."
                )
            )

        score = float(
            model.predict_proba(frame)[0][1]
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=422,
            detail=(
                "Prediction failed. Verify that "
                "the input features and categories "
                "match the training pipeline."
            )
        )

    return {
        "model": "Logistic Regression",
        "experimental_model_score": round(score, 4),
        "score_percentage": round(score * 100, 2),
        "decision_threshold": 0.5,
        "threshold_exceeded": bool(score >= 0.5),
        "clinical_risk_classification": None,
        "disclaimer": DISCLAIMER,
    }
# Add these imports near the top of backend/main.py:
# import csv
# import math

# Add this endpoint after /api/model/features in backend/main.py.
# The CSV was generated in Step 6; this endpoint reads it, without retraining.

@app.get('/api/model/shap')
def model_shap():
    path = ROOT / 'reports' / 'shap_feature_importance.csv'
    if not path.exists():
        raise HTTPException(status_code=404, detail='SHAP importance CSV not found in reports/.')

    import csv
    import math
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            feature_col = next((c for c in columns if c.lower().strip() in (
                'feature', 'feature_name', 'features', 'variable')), None)
            value_col = next((c for c in columns if c.lower().strip() in (
                'mean_abs_shap', 'mean_absolute_shap', 'mean_abs_shap_value',
                'mean_abs_shap_values', 'mean(|shap_value|)', 'importance',
                'shap_importance', 'mean_shap_importance')), None)
            if not feature_col or not value_col:
                raise HTTPException(
                    status_code=422,
                    detail='Unexpected SHAP CSV columns: ' + ', '.join(columns) +
                           '. Expected a feature column and a mean absolute SHAP column.'
                )
            features = []
            for row in reader:
                name = (row.get(feature_col) or '').strip()
                try:
                    value = float(row.get(value_col, ''))
                except (ValueError, TypeError):
                    continue
                if name and math.isfinite(value) and value >= 0:
                    features.append({'feature': name, 'mean_abs_shap': value})
    except HTTPException:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail='Could not read SHAP CSV.') from exc

    features.sort(key=lambda row: row['mean_abs_shap'], reverse=True)
    return {
        'source': 'reports/shap_feature_importance.csv',
        'features': features,
        'value_scale': 'Depends on SHAP explainer configuration; check Step 6 report',
        'disclaimer': DISCLAIMER,
    }
