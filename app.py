from flask import Flask, jsonify
import logging
from src.main import run_pipeline  # import your main pipeline function

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return jsonify({"status": "Server running", "message": "Welcome to Churn Prediction API"})

@app.route("/run-pipeline", methods=["GET"])
def run_pipeline_route():
    try:
        run_pipeline()
        return jsonify({"status": "Pipeline executed successfully"}), 200
    except Exception as e:
        logging.exception("Pipeline failed")
        return jsonify({"status": "Pipeline failed", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
