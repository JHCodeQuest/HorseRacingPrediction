import logging
from pathlib import Path

import kagglehub
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(dataset_slug: str) -> pd.DataFrame:
    path = kagglehub.dataset_download(dataset_slug)
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    df = pd.read_csv(csv_files[0])
    logger.info("Data fetched successfully! Shape: %s", df.shape)
    return df
