import logging
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from horse_racing.config import Config

logger = logging.getLogger(__name__)


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, config: Config, metadata: dict | None = None
) -> Tuple[XGBClassifier, float, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        random_state=config.random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logger.info("Accuracy:  %.4f", accuracy)
    logger.info("ROC AUC:   %.4f", roc_auc)

    artifacts = {"model": model, "scaler": scaler}
    if metadata:
        artifacts.update(metadata)
    joblib.dump(artifacts, config.model_output_path)
    logger.info("Model saved to %s", config.model_output_path)

    return model, accuracy, roc_auc
