from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import base64

from cla_ca.preprocessing import normalize_dataset
from cla_ca.feature_weighting import set_weight
from cla_ca.discretization import discretize_dataset
from cla_ca.valley_detection import create_split
from cla_ca.scoring import calculate_scores
from cla_ca.prediction import predict_labels
from cla_ca.metrics import calculate_metrics
from cla_ca.enhancements.e13_calibrated_ensemble import run_e13_calibrated_ensemble_from_data

app = FastAPI(title="CLA-CA Research Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExperimentRequest(BaseModel):
    csv_base64: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the CLA-CA Research Platform API"}

@app.post("/api/run-experiment")
async def run_experiment(req: ExperimentRequest, bim: int = None, alpha: float = 0.8):
    """
    Accepts JSON body matching ExperimentRequest schema.
    Uses bim=40 for PROMISE, bim=155 for AEEEM if not specified.
    """
    try:
        # Decode base64 CSV data
        decoded = base64.b64decode(req.csv_base64)
        df = pd.read_csv(io.BytesIO(decoded))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV data: {e}")

    # Identify target column
    target_col = None
    for col in ['bug', 'defects', 'isDefective', 'class', 'Defective', 'buggy']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    # Convert target
    y = df[target_col]
    is_aeeem = False
    if y.dtype == object:
        # Check if it's AEEEM format (buggy/clean strings)
        unique_vals = set(str(val).strip().lower() for val in y.unique())
        if 'buggy' in unique_vals or 'clean' in unique_vals:
            is_aeeem = True
        y = y.apply(lambda val: 1 if str(val).strip().lower() in ['buggy', 'y', 'true', '1', 'yes', 'defective'] else 0)
    else:
        y = y.apply(lambda val: 1 if val > 0 else 0)

    feature_cols = []
    for c in df.columns:
        if c == target_col: continue
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].notna().sum() > 0:
            feature_cols.append(c)

    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    X_df = df[feature_cols]

    # Use optimal BIM values from the paper if not specified.
    if bim is None:
        bim = 155 if is_aeeem else 40

    retained_features = feature_cols
    excluded_features = []
    normalized_for_display = normalize_dataset(df.copy(), feature_cols)
    weights_for_display = set_weight(normalized_for_display, feature_cols, alpha=alpha)

    result = run_e13_calibrated_ensemble_from_data(
        X_df,
        y.values,
        is_aeeem=is_aeeem,
        dataset_name="uploaded",
        alpha=alpha,
        bim=bim,
    )
    y_pred = result["predictions"]
    probabilities = result["probabilities"]
    metrics = result["metrics"]
    total_modules = len(df)
    defective_count = int(np.sum(y_pred == 1))
    non_defective_count = int(np.sum(y_pred == 0))

    # Prepare module list preview
    modules_preview = []
    for i in range(min(100, total_modules)):
        modules_preview.append({
            'id': i + 1,
            'score': float(probabilities[i]),
            'actual': int(y.values[i]),
            'predicted': int(y_pred[i]),
            'status': 'Defective' if y_pred[i] == 1 else 'Non-Defective'
        })

    return {
        "total_modules": total_modules,
        "defective_count": defective_count,
        "non_defective_count": non_defective_count,
        "retained_features_count": len(retained_features),
        "excluded_features_count": len(excluded_features),
        "label_point": float(result["threshold_mean"]),
        "metrics": metrics,
        "weights": {k: float(v) for k, v in list(weights_for_display.items())[:20]},
        "algorithm": "E13_Calibrated_Ensemble",
        "bim": bim,
        "alpha": alpha,
        "modules": modules_preview
    }