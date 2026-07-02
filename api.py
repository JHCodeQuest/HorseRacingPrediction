import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from horse_racing.config import Config
from horse_racing.data_loader import load_dataset
from horse_racing.predictor import Predictor
from horse_racing.preprocessor import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Horse Racing Prediction Dashboard")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# --- Global state (loaded at startup) ---
df_raw: Optional[pd.DataFrame] = None
df_processed: Optional[pd.DataFrame] = None
y_series: Optional[pd.Series] = None
predictor: Optional[Predictor] = None
training_metrics: dict = {}


class PredictInput(BaseModel):
    age: float = Field(..., ge=0)
    weight: float = Field(..., ge=0)
    official_rating: float = Field(..., ge=0)
    distance: float = Field(..., ge=0)
    going: str = Field(..., min_length=1)
    odds: float = Field(..., ge=0)


@app.on_event("startup")
def startup():
    global df_raw, df_processed, y_series, predictor, training_metrics

    config = Config()

    model_path = config.model_output_path
    if not os.path.exists(model_path):
        logger.info("No saved model found. Training…")
        df_raw = load_dataset(config.dataset_slug)
        df_processed, y_series, metadata = preprocess(
            df_raw, config.drop_cols, config.categorical_cols, config.target_col
        )
        from horse_racing.trainer import train_and_evaluate
        _, acc, roc = train_and_evaluate(df_processed, y_series, config, metadata)
        training_metrics = {"accuracy": round(acc, 4), "roc_auc": round(roc, 4)}
    else:
        logger.info("Loading saved model from %s", model_path)
        df_raw = load_dataset(config.dataset_slug)
        df_processed, y_series, _ = preprocess(
            df_raw, config.drop_cols, config.categorical_cols, config.target_col
        )
        training_metrics = {"accuracy": None, "roc_auc": None}

    predictor = Predictor(model_path)


# ---------- Routes ----------


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    n_rows = len(df_raw) if df_raw is not None else 0
    n_cols = len(df_raw.columns) if df_raw is not None else 0
    sample = df_raw.head(20).to_dict(orient="records") if df_raw is not None else []
    cols = list(df_raw.columns) if df_raw is not None else []

    return templates.TemplateResponse("index.html", {
        "request": request,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "metrics": training_metrics,
        "sample": sample,
        "cols": cols,
    })


@app.get("/data", response_class=HTMLResponse)
def data_view(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
):
    if df_raw is None:
        return templates.TemplateResponse("data.html", {"request": request, "rows": [], "cols": [], "page": 1, "total": 0, "size": 50})
    total = len(df_raw)
    start = (page - 1) * size
    end = start + size
    rows = df_raw.iloc[start:end].to_dict(orient="records")
    cols = list(df_raw.columns)
    return templates.TemplateResponse("data.html", {
        "request": request,
        "rows": rows,
        "cols": cols,
        "page": page,
        "total": total,
        "size": size,
    })


@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    going_values = sorted(df_raw["going"].dropna().unique().tolist()) if df_raw is not None and "going" in df_raw.columns else []
    return templates.TemplateResponse("predict.html", {"request": request, "going_values": going_values})


@app.post("/predict", response_class=HTMLResponse)
def predict_result(input: PredictInput):
    if predictor is None:
        return '<p class="error">Model not loaded.</p>'

    pred, prob = predictor.predict(input.model_dump())
    return templates.TemplateResponse("result.html", {
        "request": {},
        "prediction": pred,
        "probability": round(prob, 4),
    })


@app.get("/api/metrics")
def api_metrics():
    return training_metrics
