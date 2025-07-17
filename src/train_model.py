# src/train_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # or RandomForestClassifier

# Load data
def train():
    df = pd.read_csv("data/segmented/segmented_data.csv")

# Features & target
    X = df.drop(columns=["Churn", "Segment", "HighValue"])
    y = df["Churn"]

# Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# Train model
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

# Save model & test data
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/churn_model.pkl")

    os.makedirs("Data/test", exist_ok=True)
    X_test.to_csv("Data/test/X_test.csv", index=False)
    y_test.to_csv("Data/test/y_test.csv", index=False)

#print("✅ Model trained and saved.")
    return
