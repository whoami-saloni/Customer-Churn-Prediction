import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def preprocess_data():
    RAW_PATH = "Data/Telco_Customer_Churn_Dataset  (3) (1).csv"
    PROCESSED_PATH = "Data/processed/data.csv"
    os.makedirs("data/processed", exist_ok=True)
# 1. Load data
    df = pd.read_csv(RAW_PATH)

# 2. Strip spaces and fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

# 3. Drop customerID (not useful for modeling)
    df.drop('customerID', axis=1, inplace=True)

# 4. Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']

    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# 5. Label encode multi-category features
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract',
                  'PaymentMethod']

    le = LabelEncoder()
    for col in multi_cat_cols:
        df[col] = le.fit_transform(df[col])

# 6. Save processed dataset
    df.to_csv(PROCESSED_PATH, index=False)
    return
#print("✅ Preprocessed data saved to:", PROCESSED_PATH)
