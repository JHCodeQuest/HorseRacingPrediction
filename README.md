# HorseRacingPrediction

Horse Racing Prediction UK — XGBoost classifier trained on historical UK/Ireland racing data (2015–2025).

## Structure

```
main.py                   # Entry point
horse_racing/
├── config.py             # Configuration dataclass
├── data_loader.py        # Kaggle dataset download
├── preprocessor.py       # Feature engineering & cleaning
└── trainer.py            # Model training, evaluation, saving
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```
