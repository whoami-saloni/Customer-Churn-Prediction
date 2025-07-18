# src/train_model.py

import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # You can replace with RandomForestClassifier if needed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train():
    # Paths
    segmented_path = "Data/segmented/segmented_data.csv"
    model_path = "models/churn_model.pkl"
    test_data_dir = "Data/test"

    # Check file existence
    if not os.path.exists(segmented_path):
        logging.error(f"❌ Segmented data not found at {segmented_path}")
        return

    # Load segmented data
    df = pd.read_csv(segmented_path)
    logging.info(f"📥 Loaded {len(df)} rows for training.")

    # Features & target
    X = df.drop(columns=["Churn", "Segment", "HighValue"])
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"🧪 Split data into {len(X_train)} training and {len(X_test)} test samples.")

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    logging.info("🤖 Model training complete.")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"✅ Model saved to {model_path}")

    # Save test data
    os.makedirs(test_data_dir, exist_ok=True)
    X_test.to_csv(os.path.join(test_data_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(test_data_dir, "y_test.csv"), index=False)
    logging.info(f"📁 Test data saved to {test_data_dir}")

# Uncomment below if you want to run this file independently
# if __name__ == "__main__":
#     train()
