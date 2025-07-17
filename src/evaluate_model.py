# src/evaluate_model.py

import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

# Load model
def evaluate():
    model = joblib.load("models/churn_model.pkl")

# Load test data
    X_test = pd.read_csv("data/test/X_test.csv")
    y_test = pd.read_csv("data/test/y_test.csv")

# Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

# Save metrics
    metrics = {
     "accuracy": round(accuracy, 4),
     "f1_score": round(f1, 4),
     "roc_auc": round(roc_auc, 4)
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

# ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    return

#print("✅ Evaluation complete. Metrics saved.")
