import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: str):
        artifacts = joblib.load(model_path)
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.train_columns: List[str] = artifacts.get("train_columns", [])
        self.medians: Dict[str, float] = artifacts.get("medians", {})
        self.categorical_cols: List[str] = artifacts.get("categorical_cols", [])
        logger.info("Predictor loaded from %s", model_path)

    def _preprocess_input(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])

        for cat_col in self.categorical_cols:
            if cat_col in df.columns:
                dummies = pd.get_dummies(df[[cat_col]], columns=[cat_col])
                for col in dummies.columns:
                    df[col] = dummies[col]

        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0.0

        for col, val in self.medians.items():
            if col in df.columns and pd.isna(df[col]).any():
                df[col] = df[col].fillna(val)

        df = df[self.train_columns]
        return df

    def predict(self, data: dict) -> Tuple[int, float]:
        X = self._preprocess_input(data)
        X_scaled = self.scaler.transform(X)
        pred = int(self.model.predict(X_scaled)[0])
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        return pred, prob

    def predict_batch(self, records: List[dict]) -> List[Tuple[int, float]]:
        return [self.predict(r) for r in records]
