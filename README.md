# 📊 Smart Customer Churn Prediction

An end-to-end MLOps pipeline to predict customer churn using machine learning, containerization (Docker), data versioning (DVC), and web deployment (Render). The project features interactive visualizations and model performance evaluation through a Flask-based interface.

---

### 📌 Project-Overview

This project leverages telecom customer data to predict churn using an XGBoost model, visualizes insights through exploratory data analysis (EDA), and enables web-based interaction using Flask. The complete pipeline is Dockerized and deployed on Render.

---

### 🚀 Features

- ✅ Data preprocessing and cleaning  
- 📊 Exploratory Data Analysis (EDA) with visualizations  
- 📦 Customer segmentation using KMeans  
- 🤖 XGBoost model training with performance metrics (Accuracy, F1, ROC-AUC)  
- 🌐 Flask web interface for data upload, EDA, and evaluation  
- 🐳 Dockerized for reproducibility  
- 📁 DVC for data and model versioning  
- ☁️ Deployed via Render

---

## Project Structure

```plaintext
Customer-Churn-Prediction/
├── app.py                  # Flask web app
├── Dockerfile              # Docker build configuration
├── requirements.txt        # Required Python packages
├── dvc.yaml                # DVC pipeline definition
├── .dvc/                   # DVC metadata
├── Data/
│   ├── processed/          # Cleaned dataset
│   ├── segmented/          # Clustering results
│   └── test/               # Test data split
├── models/
│   └── churn_model.pkl     # Trained model
├── outputs/
│   ├── metrics.json        # Evaluation scores
│   ├── roc_curve.png       # ROC curve image
│   └── eda/                # EDA plots
├── src/
│   ├── preprocess.py       # Data cleaning
│   ├── segments.py         # Customer segmentation
│   ├── train_model.py      # Model training
│   ├── evaluate_model.py   # Model evaluation
│   └── main.py             # Run full pipeline
├── templates/
│   ├── index.html          # Upload UI
│   ├── dashboard.html      # EDA dashboard
│   └── evaluation.html     # Metrics display



---

## ⚙️ Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/whoami-saloni/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```


**2. Create virtual environment & install dependencies:**

```bash
python3 -m venv churn
source churn/bin/activate
pip install -r requirements.txt
```

**3. Restore tracked data using DVC:**

```bash
dvc pull
```
**4. Run full pipeline locally:**

```bash
python src/main.py
```

**5. To run the Flask app locally:**

```bash
python app.py
```
Open in browser: http://localhost:5000

---

### Features:

📤 Upload new churn dataset (CSV)

📊 View EDA dashboard

📈 Show model evaluation (Accuracy, F1, ROC AUC, ROC Curve)

---

### 🔧 Pipeline & Tools

**src/preprocess.py** - Data cleaning & feature engineering

**src/eda.py** - Data Visualisation

**src/segments.py** - KMeans segmentation

**src/train_model.py** - XGBoost training

**src/evaluate_model.py** - Evaluation & metrics generation

**src/main.py** - Pipeline orchestration

**static/** - Stores generated plots and metrics

---

**🐳 Docker & Deployment**

Build Docker Image:

```bash
docker build -t churn-pipeline .
```
Run Docker Container:
```bash
docker run --rm -p 5000:5000 churn-pipeline
```
Live Deployment:
The app is deployed on Render using a Docker container.

---

### 📦 DVC Integration**
Track data and models with DVC:

```bash
dvc init
dvc add Data/churn.csv
dvc add models/churn_model.pkl
git add .dvc/ *.dvc .gitignore
git commit -m "Added DVC tracking"
```
---
### 🤝 Contributing
We welcome contributions to enhance this project!
Feel free to open issues or submit pull requests.

---
### 📜 License
MIT License (or specify your license here)

👤 Author
Developed by Saloni Sahal




