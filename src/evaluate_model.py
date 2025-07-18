# src/evaluate_model.py

import os
import json
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate():
    model_path = "models/churn_model.pkl"
    x_test_path = "Data/test/X_test.csv"
    y_test_path = "Data/test/y_test.csv"
    output_dir = "outputs"

    # Check file existence
    if not all(os.path.exists(p) for p in [model_path, x_test_path, y_test_path]):
        logging.error("❌ Required files missing. Ensure model and test data exist.")
        return

    # Load model and data
    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    logging.info(f"📊 Loaded test data: {len(X_test)} samples")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4)
    }

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"✅ Metrics saved to {output_dir}/metrics.json")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    logging.info(f"📈 ROC curve saved to {output_dir}/roc_curve.png")

# Uncomment to run directly
# if __name__ == "__main__":
#     evaluate()

logging.info("✅ Evaluation script is ready.")
