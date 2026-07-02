from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    dataset_slug: str = "deltaromeo/horse-racing-results-ukireland-2015-2025"
    target_col: str = "winner"
    drop_cols: List[str] = field(default_factory=lambda: ["horse_name"])
    categorical_cols: List[str] = field(default_factory=lambda: ["going"])
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    model_output_path: str = "horse_racing_model.pkl"
