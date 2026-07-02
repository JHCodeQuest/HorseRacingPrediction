import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


PreprocessResult = Tuple[pd.DataFrame, pd.Series, dict]


def preprocess(
    df: pd.DataFrame,
    drop_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
) -> PreprocessResult:
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    existing_cat_cols = [c for c in categorical_cols if c in df.columns]
    if existing_cat_cols:
        df = pd.get_dummies(df, columns=existing_cat_cols)

    numeric_cols = df.select_dtypes(include="number").columns
    medians = df[numeric_cols].median().to_dict()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    metadata = {
        "train_columns": X.columns.tolist(),
        "medians": medians,
        "categorical_cols": existing_cat_cols,
        "numeric_cols": numeric_cols.tolist(),
    }

    logger.info("Preprocessed data: X=%s, y=%s", X.shape, y.shape)
    return X, y, metadata
