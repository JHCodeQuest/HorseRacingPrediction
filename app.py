import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import kagglehub
import os
import requests
import joblib

# load data

#use CSV or API
API_KEY = os.getenv("KAGGLE_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set KAGGLEHUB_API_KEY in Codespaces secrets.")

#fetch data
BASE_URL = "https://api.kaggle.com/v1/datasets/"
DATASET_SLUG = "deltaromeo/horse-racing-results-ukireland-2015-2025"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

#Get dataset files
response = requests.get(f"{BASE_URL}/{DATASET_SLUG}/files", headers=headers)

if response.status_code == 200:
    files = response.json().get("files", [])
    #Assume first CSV file is the main dataset
    csv_file_url = files[0]["downloadURL"]
    print(f"Downloading: {csv_file_url}")
    df = pd.read_csv(csv_file_url)
    print("Data fetched successfully!")
else:
    raise Exception(f"API Error: {response.status_code}, {response.text}")

#Preprocessing
# Example columns: ['horse_name', 'age', 'weight', 'official_rating', 'distance', 'going', 'odds', 'winner']
df = df.drop(columns=['horse_name'], errors="ignore")

categorical_cols = ['going']
df = pd.get_dummies(df, columns=categorical_cols)

df = df.fillna(df.median())

#feature and target split
X = df.drop(columns=['winner'], errors="ignore")
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train model
model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# Save Model
joblib.dump(model, "horse_racing_model.pkl")
print("Model saved as horse_racing_model.pkl")
