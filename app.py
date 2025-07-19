import os
from flask import Flask, request, redirect, url_for, render_template
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.main import run_pipeline  # Update this to your actual function/module

app = Flask(__name__)
UPLOAD_FOLDER = "Data"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            uploaded_file = request.files.get("dataset")
            if uploaded_file and uploaded_file.filename.endswith(".csv"):
                uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], "churn.csv")
                uploaded_file.save(uploaded_path)
                print("✅ File uploaded successfully.", uploaded_path)

                run_pipeline()  # ← your full flow runs here

                return redirect(url_for("index"))
            else:
                return "❌ Only .csv files are accepted", 400
        except Exception as e:
            print(f"❌ Exception during POST: {e}")
            return "⚠️ Internal server error", 500
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    eda_dir = os.path.join("static", "eda")
    eda_images = [
        f"eda/{img}" for img in os.listdir(eda_dir)
        if img.endswith(".png") and img != "roc_curve.png"
    ]
    return render_template("Dashboard.html", eda_images=eda_images)

@app.route("/evaluation")
def evaluation():
    os.makedirs('static/eda', exist_ok=True) 
    metrics_path = "static/eda/metrics.json"  # Adjust path if needed
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
            print("✅ Metrics loaded successfully.")
    except FileNotFoundError:
        metrics = {
            "accuracy": "N/A",
            "f1_score": "N/A",
            "roc_auc": "N/A"
        }
    return render_template("evaluation.html", metrics=metrics)
if __name__ == "__main__":
    app.run(debug=True)
