import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data():
    RAW_PATH = "Data/Churn.csv"
    PROCESSED_PATH = "Data/processed/data.csv"
    os.makedirs("Data/processed", exist_ok=True)

    if not os.path.exists(RAW_PATH):
        logging.error(f"❌ Raw file not found at {RAW_PATH}")
        return

    # 1. Load data
    df = pd.read_csv(RAW_PATH)
    logging.info(f"📥 Loaded raw data with {df.shape[0]} rows.")

    # 2. Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    logging.info(f"🧹 Cleaned TotalCharges column. Remaining rows: {df.shape[0]}")

    # 3. Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # 4. Encode binary categorical columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # 5. Label encode multi-category features
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    le = LabelEncoder()
    for col in multi_cat_cols:
        df[col] = le.fit_transform(df[col])

    # 6. Save processed data
    df.to_csv(PROCESSED_PATH, index=False)
    logging.info(f"✅ Preprocessed data saved to: {PROCESSED_PATH}")

    return df  # Optional: return dataframe if needed
