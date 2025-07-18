import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def segments():
    # Load processed data
    processed_path = "Data/processed/data.csv"
    if not os.path.exists(processed_path):
        logging.error(f"❌ Processed data not found at {processed_path}")
        return

    df = pd.read_csv(processed_path)
    logging.info(f"📥 Loaded processed data with {df.shape[0]} rows.")

    # Select features for segmentation
    features = ['tenure', 'MonthlyCharges']
    X = df[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("🔄 Scaled features for clustering.")

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(X_scaled)
    logging.info("🤖 KMeans clustering completed.")

    # Optional: Flag high-value customers
    df['HighValue'] = df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)

    # Save segmented data
    os.makedirs("Data/segmented", exist_ok=True)
    segmented_path = "Data/segmented/segmented_data.csv"
    df.to_csv(segmented_path, index=False)
    logging.info(f"✅ Customer segmentation completed and saved to: {segmented_path}")

    return df  # Optional return

# Uncomment for standalone run
# if __name__ == "__main__":
#     segments()
