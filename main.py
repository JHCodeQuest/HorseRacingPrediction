import logging
import sys

from horse_racing.config import Config
from horse_racing.data_loader import load_dataset
from horse_racing.preprocessor import preprocess
from horse_racing.trainer import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)


def main() -> None:
    config = Config()
    df = load_dataset(config.dataset_slug)
    X, y, metadata = preprocess(df, config.drop_cols, config.categorical_cols, config.target_col)
    train_and_evaluate(X, y, config, metadata)


if __name__ == "__main__":
    main()
